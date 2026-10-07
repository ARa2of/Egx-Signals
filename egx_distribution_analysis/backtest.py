import numpy as np
import pandas as pd
from tvDatafeed import TvDatafeed, Interval
import json
import os
import time

TICKERS = ['MFPC', 'MASR', 'ETEL', 'EFIH', 'ORHD', 'CPCI', 'RMDA', 'ARCC', 'OBRI', 'EGAS', 'ADIB', 'EGAL', 'BONY', 'ENGC']
BUDGET = 10000
FEE_PER_SIDE = 0.003
ROUND_TRIP_FEE = FEE_PER_SIDE * 2
MAX_TRADES_PER_DAY = 3
MAX_PARTICIPATION = 0.10  # max share of a single bar's volume we may take
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(OUTPUT_DIR, '_cache')

_tv = None  # module-level client, reused across tickers


def get_tv():
    """Return a shared TvDatafeed client (one websocket per run, not per ticker)."""
    global _tv
    if _tv is None:
        _tv = TvDatafeed()
    return _tv


def _cache_path(ticker):
    return os.path.join(CACHE_DIR, '{}.csv'.format(ticker))


def _save_cache(ticker, data):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        data.to_csv(_cache_path(ticker))
    except Exception:
        pass  # cache is best-effort


def _load_cache(ticker):
    path = _cache_path(ticker)
    if not os.path.exists(path):
        return None
    try:
        data = pd.read_csv(path, index_col=0, parse_dates=True)
        return data if not data.empty else None
    except Exception:
        return None

STRATEGIES = {
    'Narrow (0.5s)': {'buy_mult': -0.5, 'sell_mult': 0.5},
    'Mid (1s)': {'buy_mult': -1.0, 'sell_mult': 1.0},
    'Loose (2s)': {'buy_mult': -2.0, 'sell_mult': 2.0},
}

# Corporate actions the raw TvDatafeed feed does NOT adjust for.
# Value = TOTAL shares multiplier applied from the ex-date onward
# (bonus issue of B shares/share -> 1 + B). Bars strictly before the ex-date
# get prices divided by the factor and volumes multiplied by it.
# ORHD (Orascom Development Egypt): 2.2288508184 bonus shares per share,
# effective from first trading session 2026-10-07 (EGX disclosure 2026-09-21).
CORPORATE_ACTIONS = {
    'ORHD': {'2026-10-07': 3.2288508184},
}


def adjust_for_corporate_actions(ticker, data):
    """Rescale pre-ex-date bars for known splits/bonus issues so stats and
    band levels are computed on one consistent price basis. Cache keeps raw
    prices; adjustment is re-applied on every load (idempotent)."""
    events = CORPORATE_ACTIONS.get(ticker)
    if not events or data is None or data.empty:
        return data
    for ex_date_str, factor in events.items():
        ex = pd.Timestamp(ex_date_str)
        mask = data.index < ex
        if not mask.any():
            continue
        data = data.copy()
        for col in ('open', 'high', 'low', 'close'):
            if col in data.columns:
                data.loc[mask, col] = data.loc[mask, col] / factor
        if 'volume' in data.columns:
            data.loc[mask, 'volume'] = data.loc[mask, 'volume'] * factor
        print("  {}: split-adjusted {} pre-{} bars (÷{:.6f})".format(
            ticker, int(mask.sum()), ex_date_str, factor))
    return data


def fetch_data(ticker, n_bars=1000):
    """Fetch 5-min bars for a ticker. Reuses one TvDatafeed client across
    tickers; on live failure falls back to the on-disk cache from the last
    successful run so a flaky fetch does not lose the whole report."""
    for attempt in range(3):
        try:
            tv = get_tv()
            data = tv.get_hist(ticker, exchange='EGX', interval=Interval.in_5_minute, n_bars=n_bars)
            if data is not None and not data.empty:
                _save_cache(ticker, data)  # cache keeps raw (unadjusted) prices
                return adjust_for_corporate_actions(ticker, data)
        except Exception:
            if attempt < 2:
                time.sleep(1)
                # drop the possibly-dead client so the next attempt reconnects
                global _tv
                _tv = None
            else:
                break
    cached = _load_cache(ticker)
    if cached is not None:
        print("  WARNING: {} live fetch failed; using cached data from {}".format(
            ticker, cached.index[-1]))
        return adjust_for_corporate_actions(ticker, cached)
    raise ValueError("No data found for {}".format(ticker))


ROLLING_WINDOW = 300  # ~5 EGX sessions of 5-min bars; matches the live 5d signal window


def backtest_strategy(data, strategy_name, buy_mult, sell_mult):
    close = data['close'].values
    volume = data['volume'].values
    dates = data.index

    # Rolling stats shifted by 1 bar: level at bar i uses only bars [i-W .. i-1].
    # Static full-sample mean/std leaked future data into every trade (look-ahead).
    close_s = data['close']
    roll_mean = close_s.rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean().shift(1)
    roll_std = close_s.rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).std().shift(1)

    budget = BUDGET
    shares = 0
    buy_price = 0
    buy_time = None
    sell_level = None  # frozen at entry, like the live trading plan
    trades = []
    daily_trades = {}
    in_trade = False
    last_m = np.nan
    last_s = np.nan

    for i in range(len(close)):
        price = close[i]
        vol = volume[i]
        day = dates[i].date()
        bar_time = dates[i]

        if day not in daily_trades:
            daily_trades[day] = 0

        m = roll_mean.iloc[i]
        s = roll_std.iloc[i]
        if pd.isna(m) or pd.isna(s) or s <= 0:
            continue  # warm-up: not enough history for honest levels

        last_m, last_s = m, s
        buy_level = m + buy_mult * s
        sell_level_now = m + sell_mult * s

        if not in_trade:
            if price <= buy_level and daily_trades[day] < MAX_TRADES_PER_DAY:
                affordable = int(budget / price)
                max_shares = min(affordable, int(vol * MAX_PARTICIPATION))
                if max_shares > 0:
                    shares = max_shares
                    buy_price = price
                    buy_time = bar_time
                    sell_level = sell_level_now  # frozen at entry
                    cost = shares * price
                    fee = cost * FEE_PER_SIDE
                    budget -= (cost + fee)
                    daily_trades[day] += 1  # cap counts ENTRIES per day
                    in_trade = True
        else:
            if price >= sell_level:
                revenue = shares * price
                fee = revenue * FEE_PER_SIDE
                net_revenue = revenue - fee
                profit = net_revenue - (shares * buy_price + shares * buy_price * FEE_PER_SIDE)
                budget += net_revenue

                trades.append({
                    'entry_time': buy_time.strftime('%Y-%m-%d %H:%M'),
                    'exit_time': bar_time.strftime('%Y-%m-%d %H:%M'),
                    'entry_price': round(buy_price, 4),
                    'exit_price': round(price, 4),
                    'shares': shares,
                    'profit': round(profit, 2),
                    'return_pct': round((price - buy_price) / buy_price * 100, 4),
                })

                shares = 0
                buy_price = 0
                buy_time = None
                in_trade = False

    if in_trade:
        final_price = close[-1]
        revenue = shares * final_price
        fee = revenue * FEE_PER_SIDE
        net_revenue = revenue - fee
        profit = net_revenue - (shares * buy_price + shares * buy_price * FEE_PER_SIDE)
        budget += net_revenue
        trades.append({
            'entry_time': buy_time.strftime('%Y-%m-%d %H:%M'),
            'exit_time': dates[-1].strftime('%Y-%m-%d %H:%M'),
            'entry_price': round(buy_price, 4),
            'exit_price': round(final_price, 4),
            'shares': shares,
            'profit': round(profit, 2),
            'return_pct': round((final_price - buy_price) / buy_price * 100, 4),
            'open_position': True,
        })
        shares = 0
        buy_time = None
        in_trade = False

    total_profit = sum(t['profit'] for t in trades)
    total_return = (budget - BUDGET) / BUDGET * 100
    winning_trades = [t for t in trades if t['profit'] > 0]
    losing_trades = [t for t in trades if t['profit'] <= 0]
    win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
    avg_profit = np.mean([t['profit'] for t in trades]) if trades else 0
    avg_return = np.mean([t['return_pct'] for t in trades]) if trades else 0
    max_drawdown = min([t['profit'] for t in trades]) if trades else 0
    trading_days = len([d for d, c in daily_trades.items() if c > 0])
    total_possible_days = len(set(dates.date))
    avg_trades_per_day = sum(daily_trades.values()) / total_possible_days if total_possible_days > 0 else 0

    if np.isnan(last_m):
        # Fewer bars than the rolling window: fall back to full-sample stats
        last_m = float(close.mean())
        last_s = float(close.std()) if close.std() > 0 else 1e-9
    disp_buy = last_m + buy_mult * last_s
    disp_sell = last_m + sell_mult * last_s
    band_width = disp_sell - disp_buy
    band_width_pct = band_width / last_m * 100 if last_m > 0 else 0

    # Band crossings vs the rolling (no-look-ahead) levels
    buy_series = (roll_mean + buy_mult * roll_std).values
    sell_series = (roll_mean + sell_mult * roll_std).values
    crosses_buy = 0
    crosses_sell = 0
    prev_above_buy = None
    prev_above_sell = None
    for i in range(len(close)):
        b = buy_series[i]
        sl = sell_series[i]
        if np.isnan(b) or np.isnan(sl):
            continue
        above_buy = close[i] > b
        above_sell = close[i] > sl
        if prev_above_buy is not None:
            if prev_above_buy and not above_buy:
                crosses_buy += 1
            if not prev_above_sell and above_sell:
                crosses_sell += 1
        prev_above_buy = above_buy
        prev_above_sell = above_sell

    ease_score = 0
    if crosses_buy >= 10 and crosses_sell >= 10:
        ease_score = 3
    elif crosses_buy >= 5 and crosses_sell >= 5:
        ease_score = 2
    elif crosses_buy >= 2 and crosses_sell >= 2:
        ease_score = 1

    return {
        'strategy': strategy_name,
        'buy_level': round(disp_buy, 4),
        'sell_level': round(disp_sell, 4),
        'band_width': round(band_width, 4),
        'band_width_pct': round(band_width_pct, 2),
        'total_trades': len(trades),
        'total_profit': round(total_profit, 2),
        'total_return_pct': round(total_return, 2),
        'final_budget': round(budget, 2),
        'win_rate': round(win_rate, 1),
        'avg_profit_per_trade': round(avg_profit, 2),
        'avg_return_per_trade': round(avg_return, 4),
        'max_loss': round(max_drawdown, 2),
        'trading_days': trading_days,
        'avg_trades_per_day': round(avg_trades_per_day, 2),
        'crosses_buy': crosses_buy,
        'crosses_sell': crosses_sell,
        'ease_score': ease_score,
        'trades': trades,
    }


def analyze_ticker(ticker):
    try:
        print("  Fetching {}...".format(ticker))
        data = fetch_data(ticker)
        print("  Got {} bars".format(len(data)))

        close = data['close'].values
        current = close[-1]
        # Display the CURRENT band basis (rolling 5-session stats), not full-sample
        roll_m = data['close'].rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean().iloc[-1]
        roll_s = data['close'].rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).std().iloc[-1]
        if pd.isna(roll_m):
            roll_m = data['close'].mean()
            roll_s = data['close'].std()
        mean = float(roll_m)
        std = float(roll_s)

        results = {}
        for name, params in STRATEGIES.items():
            r = backtest_strategy(data, name, params['buy_mult'], params['sell_mult'])
            results[name] = r

        best_strategy = max(results.keys(), key=lambda k: results[k]['total_return_pct'])

        ease_values = [r['ease_score'] for r in results.values()]
        avg_ease = np.mean(ease_values) if ease_values else 0
        if avg_ease >= 2.5:
            ease_label = 'Easy'
        elif avg_ease >= 1.5:
            ease_label = 'Moderate'
        elif avg_ease >= 0.5:
            ease_label = 'Difficult'
        else:
            ease_label = 'Very Difficult'

        return {
            'ticker': ticker,
            'current_price': round(current, 2),
            'mean': round(mean, 2),
            'std': round(std, 2),
            'strategies': results,
            'best_strategy': best_strategy,
            'ease_label': ease_label,
            'ease_score': round(avg_ease, 1),
        }
    except Exception as e:
        print("  ERROR: {}".format(e))
        return None


def generate_html(ticker_results):
    ticker_results = [r for r in ticker_results if r is not None]
    ticker_results.sort(key=lambda x: (
        0 if x['ease_label'] == 'Easy' else 1 if x['ease_label'] == 'Moderate' else 2 if x['ease_label'] == 'Difficult' else 3,
        -x['strategies'][x['best_strategy']]['total_return_pct']
    ))

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EGX Band Strategy Backtest</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f0f23; color: #e0e0e0; padding: 10px; }
  h1 { text-align: center; margin: 10px 0; font-size: 1.4em; color: #ffd93d; }
  .subtitle { text-align: center; color: #888; margin-bottom: 15px; font-size: 0.85em; }
  .summary-table { width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 0.85em; }
  .summary-table th { background: #16213e; color: #ffd93d; padding: 8px 6px; text-align: center; border-bottom: 2px solid #2a2a4a; }
  .summary-table td { padding: 6px; text-align: center; border-bottom: 1px solid #2a2a4a; }
  .summary-table tr:hover { background: #16213e; }
  .ticker-section { background: #1a1a2e; border: 1px solid #2a2a4a; border-radius: 8px; margin-bottom: 10px; overflow: hidden; }
  .ticker-header { display: flex; align-items: center; padding: 10px 15px; cursor: pointer; gap: 12px; }
  .ticker-header:hover { background: #16213e; }
  .ticker-name { font-weight: 700; font-size: 1.1em; min-width: 70px; }
  .ease-badge { padding: 3px 10px; border-radius: 4px; font-weight: 600; font-size: 0.8em; }
  .ease-Easy { background: rgba(107,203,119,0.2); color: #6bcb77; }
  .ease-Moderate { background: rgba(255,217,61,0.2); color: #ffd93d; }
  .ease-Difficult { background: rgba(255,159,67,0.2); color: #ff9f43; }
  .ease-Very-Difficult { background: rgba(238,90,36,0.2); color: #ee5a24; }
  .best-badge { padding: 3px 10px; border-radius: 4px; font-weight: 600; font-size: 0.8em; background: rgba(107,203,119,0.2); color: #6bcb77; }
  .accordion { display: none; padding: 0 15px 15px; }
  .strat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 10px; margin-top: 10px; }
  .strat-card { background: #16213e; border-radius: 6px; padding: 12px; border-left: 3px solid #2a2a4a; }
  .strat-card.best { border-left-color: #6bcb77; }
  .strat-name { font-weight: 700; color: #ffd93d; margin-bottom: 6px; }
  .strat-row { display: flex; justify-content: space-between; padding: 2px 0; font-size: 0.85em; }
  .strat-row .label { color: #888; }
  .strat-row .value { font-weight: 600; }
  .profit-pos { color: #6bcb77; }
  .profit-neg { color: #ee5a24; }
  .arrow { font-size: 0.8em; color: #888; margin-left: auto; }
  .params-box { background: #16213e; border-radius: 6px; padding: 10px 15px; margin: 15px 0; }
  .params-box h3 { color: #ffd93d; margin-bottom: 8px; font-size: 0.95em; }
  .params-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; font-size: 0.85em; }
  .params-grid div { padding: 6px; background: #1a1a2e; border-radius: 4px; text-align: center; }
  .params-grid .label { color: #888; font-size: 0.75em; }
  .params-grid .value { font-weight: 700; margin-top: 2px; }
</style>
</head>
<body>
<h1>EGX Band Strategy Backtest</h1>
<p class="subtitle">10,000 EGP budget | Max 3 trades/day | 20 sessions | 5-min data | Fees: 0.6% round trip</p>

<div class="params-box">
  <h3>Strategy Definitions</h3>
  <div class="params-grid">
    <div><div class="label">Narrow Band</div><div class="value" style="color:#6bcb77">Buy: Mean - 0.5\u03C3</div><div class="value" style="color:#6bcb77">Sell: Mean + 0.5\u03C3</div></div>
    <div><div class="label">Mid Band</div><div class="value" style="color:#ffd93d">Buy: Mean - 1\u03C3</div><div class="value" style="color:#ffd93d">Sell: Mean + 1\u03C3</div></div>
    <div><div class="label">Loose Band</div><div class="value" style="color:#ff9f43">Buy: Mean - 2\u03C3</div><div class="value" style="color:#ff9f43">Sell: Mean + 2\u03C3</div></div>
  </div>
</div>

<table class="summary-table">
  <tr>
    <th>Ticker</th><th>Price</th><th>Mean</th><th>Std</th>
    <th>Narrow Return</th><th>Mid Return</th><th>Loose Return</th>
    <th>Best Strategy</th><th>Ease</th>
  </tr>"""

    for r in ticker_results:
        s = r['strategies']
        best = r['best_strategy']
        html += """
  <tr>
    <td><strong>{ticker}</strong></td><td>{price:.2f}</td><td>{mean:.2f}</td><td>{std:.2f}</td>
    <td class="{narrow_cls}">{narrow_ret:+.2f}% ({narrow_trades} trades)</td>
    <td class="{mid_cls}">{mid_ret:+.2f}% ({mid_trades} trades)</td>
    <td class="{loose_cls}">{loose_ret:+.2f}% ({loose_trades} trades)</td>
    <td><span class="best-badge">{best}</span></td>
    <td><span class="ease-badge ease-{ease}">{ease}</span></td>
  </tr>""".format(
            ticker=r['ticker'], price=r['current_price'], mean=r['mean'], std=r['std'],
            narrow_ret=s['Narrow (0.5s)']['total_return_pct'],
            narrow_trades=s['Narrow (0.5s)']['total_trades'],
            narrow_cls='profit-pos' if s['Narrow (0.5s)']['total_return_pct'] > 0 else 'profit-neg',
            mid_ret=s['Mid (1s)']['total_return_pct'],
            mid_trades=s['Mid (1s)']['total_trades'],
            mid_cls='profit-pos' if s['Mid (1s)']['total_return_pct'] > 0 else 'profit-neg',
            loose_ret=s['Loose (2s)']['total_return_pct'],
            loose_trades=s['Loose (2s)']['total_trades'],
            loose_cls='profit-pos' if s['Loose (2s)']['total_return_pct'] > 0 else 'profit-neg',
            best=best, ease=r['ease_label'],
        )

    html += "\n</table>\n"

    for r in ticker_results:
        s = r['strategies']
        best = r['best_strategy']
        html += """
<div class="ticker-section">
  <div class="ticker-header" onclick="toggleSection(this)">
    <span class="ticker-name">{ticker}</span>
    <span class="ease-badge ease-{ease}">{ease}</span>
    <span class="best-badge">Best: {best} ({best_ret:+.2f}%)</span>
    <span class="arrow">&#9660;</span>
  </div>
  <div class="accordion">
    <div class="strat-grid">""".format(
            ticker=r['ticker'], ease=r['ease_label'], best=best,
            best_ret=s[best]['total_return_pct'],
        )

        for sname, sd in s.items():
            is_best = ' best' if sname == best else ''
            cls = 'profit-pos' if sd['total_return_pct'] > 0 else 'profit-neg'
            html += """
      <div class="strat-card{is_best}">
        <div class="strat-name">{sname}</div>
        <div class="strat-row"><span class="label">Buy Level</span><span class="value">{buy:.4f} EGP</span></div>
        <div class="strat-row"><span class="label">Sell Level</span><span class="value">{sell:.4f} EGP</span></div>
        <div class="strat-row"><span class="label">Band Width</span><span class="value">{bw:.4f} EGP ({bwp:.1f}%)</span></div>
        <div class="strat-row"><span class="label">Total Trades</span><span class="value">{trades}</span></div>
        <div class="strat-row"><span class="label">Trading Days</span><span class="value">{days}</span></div>
        <div class="strat-row"><span class="label">Avg Trades/Day</span><span class="value">{atd:.1f}</span></div>
        <div class="strat-row"><span class="label">Win Rate</span><span class="value">{wr:.1f}%</span></div>
        <div class="strat-row"><span class="label">Total Return</span><span class="value {cls}">{ret:+.2f}%</span></div>
        <div class="strat-row"><span class="label">Total Profit</span><span class="value {cls}">{profit:+.2f} EGP</span></div>
        <div class="strat-row"><span class="label">Final Budget</span><span class="value">{fb:.2f} EGP</span></div>
        <div class="strat-row"><span class="label">Avg Profit/Trade</span><span class="value {cls}">{apt:+.2f} EGP</span></div>
        <div class="strat-row"><span class="label">Max Single Loss</span><span class="value profit-neg">{ml:+.2f} EGP</span></div>
        <div class="strat-row"><span class="label">Buy Crosses</span><span class="value">{cb}</span></div>
        <div class="strat-row"><span class="label">Sell Crosses</span><span class="value">{cs}</span></div>
      </div>""".format(
                is_best=is_best, sname=sname,
                buy=sd['buy_level'], sell=sd['sell_level'],
                bw=sd['band_width'], bwp=sd['band_width_pct'],
                trades=sd['total_trades'], days=sd['trading_days'],
                atd=sd['avg_trades_per_day'], wr=sd['win_rate'],
                ret=sd['total_return_pct'], profit=sd['total_profit'],
                fb=sd['final_budget'], apt=sd['avg_profit_per_trade'],
                ml=sd['max_loss'], cb=sd['crosses_buy'], cs=sd['crosses_sell'],
                cls=cls,
            )

        html += """
    </div>
  </div>
</div>"""

    html += """
<script>
function toggleSection(header) {
  var content = header.nextElementSibling;
  var arrow = header.querySelector('.arrow');
  if (content.style.display === 'block') {
    content.style.display = 'none';
    arrow.innerHTML = '&#9660;';
  } else {
    content.style.display = 'block';
    arrow.innerHTML = '&#9650;';
  }
}
</script>
</body>
</html>"""

    return html


def main():
    print("EGX Band Strategy Backtest")
    print("=" * 50)
    print("Budget: {} EGP | Max {} trades/day | Fees: {}% per side".format(BUDGET, MAX_TRADES_PER_DAY, FEE_PER_SIDE * 100))

    results = []
    for ticker in TICKERS:
        r = analyze_ticker(ticker)
        if r:
            results.append(r)
        time.sleep(0.3)

    if not results:
        print("\nNo results from any ticker - all fetches failed.")
        return

    results.sort(key=lambda x: (
        0 if x['ease_label'] == 'Easy' else 1 if x['ease_label'] == 'Moderate' else 2 if x['ease_label'] == 'Difficult' else 3,
        -x['strategies'][x['best_strategy']]['total_return_pct']
    ))

    print("\n" + "=" * 80)
    print("RESULTS (sorted by ease of trading)")
    print("=" * 80)
    print("{:6} | {:>8} | {:>12} | {:>10} | {:>10} | {:>10} | {:8} | {}".format(
        'TICKER', 'PRICE', 'NARROW', 'MID', 'LOOSE', 'BEST RET', 'BEST', 'EASE'))
    print("-" * 100)
    narrow_key = [k for k in results[0]['strategies'] if 'Narrow' in k][0]
    mid_key = [k for k in results[0]['strategies'] if 'Mid' in k][0]
    loose_key = [k for k in results[0]['strategies'] if 'Loose' in k][0]
    for r in results:
        s = r['strategies']
        best = r['best_strategy']
        print("{:6} | {:>8.2f} | {:>+7.2f}%  | {:>+7.2f}%  | {:>+7.2f}%  | {:>+7.2f}%  | {:8} | {}".format(
            r['ticker'], r['current_price'],
            s[narrow_key]['total_return_pct'],
            s[mid_key]['total_return_pct'],
            s[loose_key]['total_return_pct'],
            s[best]['total_return_pct'], best, r['ease_label']))

    html = generate_html(results)
    html_path = os.path.join(OUTPUT_DIR, "backtest.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print("\nHTML report: {}".format(html_path))

    json_data = {
        'generated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
        'budget': BUDGET,
        'max_trades_per_day': MAX_TRADES_PER_DAY,
        'fees_per_side': FEE_PER_SIDE * 100,
        'tickers': []
    }
    for r in results:
        ticker_data = {
            'ticker': r['ticker'],
            'current_price': r['current_price'],
            'mean': r['mean'],
            'std': r['std'],
            'best_strategy': r['best_strategy'],
            'ease_label': r['ease_label'],
            'strategies': {}
        }
        for sname, sd in r['strategies'].items():
            ticker_data['strategies'][sname] = {
                'buy_level': sd['buy_level'],
                'sell_level': sd['sell_level'],
                'band_width_pct': sd['band_width_pct'],
                'total_trades': sd['total_trades'],
                'total_return_pct': sd['total_return_pct'],
                'total_profit': sd['total_profit'],
                'win_rate': sd['win_rate'],
                'avg_trades_per_day': sd['avg_trades_per_day'],
                'crosses_buy': sd['crosses_buy'],
                'crosses_sell': sd['crosses_sell'],
            }
        json_data['tickers'].append(ticker_data)
    json_path = os.path.join(OUTPUT_DIR, "backtest.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    print("JSON report: {}".format(json_path))


def run_backtest(tickers=None, fetched_data=None):
    if tickers is None:
        tickers = TICKERS
    results = []
    for ticker in tickers:
        try:
            if fetched_data and ticker in fetched_data:
                print("  Backtesting {} (using cached data)...".format(ticker))
                data = fetched_data[ticker]
            else:
                print("  Fetching {}...".format(ticker))
                data = fetch_data(ticker)
                print("  Got {} bars".format(len(data)))

            close = data['close'].values
            current = close[-1]
            roll_m = data['close'].rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean().iloc[-1]
            roll_s = data['close'].rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).std().iloc[-1]
            if pd.isna(roll_m):
                roll_m = data['close'].mean()
                roll_s = data['close'].std()
            mean = float(roll_m)
            std = float(roll_s)

            strat_results = {}
            for name, params in STRATEGIES.items():
                r = backtest_strategy(data, name, params['buy_mult'], params['sell_mult'])
                strat_results[name] = r

            best_strategy = max(strat_results.keys(), key=lambda k: strat_results[k]['total_return_pct'])

            ease_values = [r['ease_score'] for r in strat_results.values()]
            avg_ease = np.mean(ease_values) if ease_values else 0
            if avg_ease >= 2.5:
                ease_label = 'Easy'
            elif avg_ease >= 1.5:
                ease_label = 'Moderate'
            elif avg_ease >= 0.5:
                ease_label = 'Difficult'
            else:
                ease_label = 'Very Difficult'

            results.append({
                'ticker': ticker,
                'current_price': round(current, 2),
                'mean': round(mean, 2),
                'std': round(std, 2),
                'strategies': strat_results,
                'best_strategy': best_strategy,
                'ease_label': ease_label,
                'ease_score': round(avg_ease, 1),
            })
        except Exception as e:
            print("  ERROR: {}".format(e))
        time.sleep(0.3)
    strategy_keys = list(STRATEGIES.keys())
    narrow_key, mid_key, loose_key = strategy_keys[0], strategy_keys[1], strategy_keys[2]
    if not results:
        # every fetch failed: return empty results with valid keys instead of
        # crashing on results[0] (IndexError) so callers can render a stub.
        print("  WARNING: no ticker data fetched; backtest empty")
        return [], narrow_key, mid_key, loose_key
    results.sort(key=lambda x: (
        0 if x['ease_label'] == 'Easy' else 1 if x['ease_label'] == 'Moderate' else 2 if x['ease_label'] == 'Difficult' else 3,
        -x['strategies'][x['best_strategy']]['total_return_pct']
    ))
    return results, narrow_key, mid_key, loose_key


def format_trades_list(trades, color):
    if not trades:
        return '<div class="bt-trades-empty">No completed trades</div>'
    html = '<div class="bt-trades-list">'
    for i, t in enumerate(trades):
        open_tag = ' <span style="color:#ffd93d;font-size:0.75em">(OPEN)</span>' if t.get('open_position') else ''
        profit_cls = 'profit-pos' if t['profit'] > 0 else 'profit-neg'
        html += """
      <div class="bt-trade-row">
        <span class="bt-trade-num">#{}</span>
        <span class="bt-trade-time">{} &rarr; {}</span>
        <span class="bt-trade-prices">{} &rarr; {}</span>
        <span class="bt-trade-profit {}">{:+,.2f} EGP{}</span>
      </div>""".format(
            i + 1, t['entry_time'], t['exit_time'],
            t['entry_price'], t['exit_price'],
            profit_cls, t['profit'], open_tag)
    html += '</div>'
    return html


def backtest_fragment(results, narrow_key, mid_key, loose_key):
    narrow_total = sum(r['strategies'][narrow_key]['total_profit'] for r in results)
    mid_total = sum(r['strategies'][mid_key]['total_profit'] for r in results)
    loose_total = sum(r['strategies'][loose_key]['total_profit'] for r in results)
    best_total = sum(r['strategies'][r['best_strategy']]['total_profit'] for r in results)

    html = """
<div class="backtest-section">
  <div class="backtest-header" onclick="toggleSection(this)">
    <span style="font-size:1.1em;font-weight:700;color:#ffd93d">Backtest: Band Strategy Comparison</span>
    <span style="color:#888;font-size:0.85em">10,000 EGP per stock | Max 3 trades/day | 0.6% fees</span>
    <span class="arrow">&#9660;</span>
  </div>
  <div class="accordion" style="padding:0 15px 15px">
    <div class="bt-summary">
      <div class="bt-box"><div class="bt-label">Narrow (0.5s) Total</div><div class="bt-value" style="color:{narrow_color}">{narrow_str} EGP</div></div>
      <div class="bt-box"><div class="bt-label">Mid (1s) Total</div><div class="bt-value" style="color:{mid_color}">{mid_str} EGP</div></div>
      <div class="bt-box"><div class="bt-label">Loose (2s) Total</div><div class="bt-value" style="color:{loose_color}">{loose_str} EGP</div></div>
      <div class="bt-box" style="border-color:#6bcb77"><div class="bt-label" style="color:#6bcb77">Best Combo Total (hindsight)</div><div class="bt-value" style="color:#6bcb77;font-size:1.3em">{best_str} EGP</div></div>
    </div>
    <p style="color:#888;font-size:0.75em;margin:4px 0 8px">*Best-per-ticker band chosen on the same data it was scored on — in-sample hindsight, not a forward-tested allocation.</p>
    <table class="bt-table">
      <tr><th></th><th>Ticker</th><th>Price</th><th>Narrow</th><th>Mid</th><th>Loose</th><th>Best</th><th>Best Strategy</th><th>Ease</th></tr>""".format(
            narrow_str='{:+,.0f}'.format(narrow_total), narrow_color='#6bcb77' if narrow_total >= 0 else '#ee5a24',
            mid_str='{:+,.0f}'.format(mid_total), mid_color='#ffd93d' if mid_total >= 0 else '#ee5a24',
            loose_str='{:+,.0f}'.format(loose_total), loose_color='#ff9f43' if loose_total >= 0 else '#ee5a24',
            best_str='{:+,.0f}'.format(best_total))

    for r in results:
        s = r['strategies']
        best = r['best_strategy']
        narrow_ret = s[narrow_key]['total_return_pct']
        mid_ret = s[mid_key]['total_return_pct']
        loose_ret = s[loose_key]['total_return_pct']
        best_ret = s[best]['total_return_pct']
        narrow_profit = s[narrow_key]['total_profit']
        mid_profit = s[mid_key]['total_profit']
        loose_profit = s[loose_key]['total_profit']
        best_profit = s[best]['total_profit']
        narrow_trades = s[narrow_key]['total_trades']
        mid_trades = s[mid_key]['total_trades']
        loose_trades = s[loose_key]['total_trades']
        narrow_wr = s[narrow_key]['win_rate']
        mid_wr = s[mid_key]['win_rate']
        loose_wr = s[loose_key]['win_rate']
        narrow_atd = s[narrow_key]['avg_trades_per_day']
        mid_atd = s[mid_key]['avg_trades_per_day']
        loose_atd = s[loose_key]['avg_trades_per_day']

        nc = 'profit-pos' if narrow_ret > 0 else 'profit-neg'
        mc = 'profit-pos' if mid_ret > 0 else 'profit-neg'
        lc = 'profit-pos' if loose_ret > 0 else 'profit-neg'
        bc = 'profit-pos' if best_ret > 0 else 'profit-neg'

        html += """
      <tr class="bt-row" onclick="toggleBtDetail(this)" style="cursor:pointer">
        <td style="color:#888;font-size:0.8em">&#9654;</td>
        <td><strong>{ticker}</strong></td><td>{price:.2f}</td>
        <td class="{nc}">{nr:+.1f}%</td>
        <td class="{mc}">{mr:+.1f}%</td>
        <td class="{lc}">{lr:+.1f}%</td>
        <td class="{bc}">{br:+.1f}%</td>
        <td>{best}</td>
        <td><span class="ease-badge ease-{ease}">{ease}</span></td>
      </tr>
      <tr class="bt-detail" style="display:none">
        <td colspan="9">
          <div class="bt-detail-grid">
            <div class="bt-detail-card">
              <div class="bt-detail-title" style="color:#6bcb77">Narrow (0.5s)</div>
              <div class="bt-detail-row"><span>Buy Level:</span><span>{nb:.4f}</span></div>
              <div class="bt-detail-row"><span>Sell Level:</span><span>{ns:.4f}</span></div>
              <div class="bt-detail-row"><span>Completed Trades:</span><span>{nt}</span></div>
              <div class="bt-detail-row"><span>Win Rate:</span><span>{nw:.1f}%</span></div>
              <div class="bt-detail-row"><span>Band Width:</span><span>{nbw:.2f} EGP</span></div>
              <div class="bt-detail-row"><span>Profit:</span><span class="{nc}">{np:+,.2f} EGP</span></div>
              {narrow_trades}
            </div>
            <div class="bt-detail-card">
              <div class="bt-detail-title" style="color:#ffd93d">Mid (1s)</div>
              <div class="bt-detail-row"><span>Buy Level:</span><span>{mb:.4f}</span></div>
              <div class="bt-detail-row"><span>Sell Level:</span><span>{ms:.4f}</span></div>
              <div class="bt-detail-row"><span>Completed Trades:</span><span>{mt}</span></div>
              <div class="bt-detail-row"><span>Win Rate:</span><span>{mw:.1f}%</span></div>
              <div class="bt-detail-row"><span>Band Width:</span><span>{mbw:.2f} EGP</span></div>
              <div class="bt-detail-row"><span>Profit:</span><span class="{mc}">{mp:+,.2f} EGP</span></div>
              {mid_trades}
            </div>
            <div class="bt-detail-card">
              <div class="bt-detail-title" style="color:#ff9f43">Loose (2s)</div>
              <div class="bt-detail-row"><span>Buy Level:</span><span>{lb:.4f}</span></div>
              <div class="bt-detail-row"><span>Sell Level:</span><span>{ls:.4f}</span></div>
              <div class="bt-detail-row"><span>Completed Trades:</span><span>{lt}</span></div>
              <div class="bt-detail-row"><span>Win Rate:</span><span>{lw:.1f}%</span></div>
              <div class="bt-detail-row"><span>Band Width:</span><span>{lbw:.2f} EGP</span></div>
              <div class="bt-detail-row"><span>Profit:</span><span class="{lc}">{lp:+,.2f} EGP</span></div>
              {loose_trades}
            </div>
          </div>
        </td>
      </tr>""".format(
            ticker=r['ticker'], price=r['current_price'],
            nc=nc, nr=narrow_ret, mc=mc, mr=mid_ret, lc=lc, lr=loose_ret, bc=bc, br=best_ret,
            best=best, ease=r['ease_label'],
            nb=s[narrow_key]['buy_level'], ns=s[narrow_key]['sell_level'],
            nt=narrow_trades, nw=narrow_wr, nbw=s[narrow_key]['band_width'], np=narrow_profit,
            narrow_trades=format_trades_list(s[narrow_key]['trades'], '#6bcb77'),
            mb=s[mid_key]['buy_level'], ms=s[mid_key]['sell_level'],
            mt=mid_trades, mw=mid_wr, mbw=s[mid_key]['band_width'], mp=mid_profit,
            mid_trades=format_trades_list(s[mid_key]['trades'], '#ffd93d'),
            lb=s[loose_key]['buy_level'], ls=s[loose_key]['sell_level'],
            lt=loose_trades, lw=loose_wr, lbw=s[loose_key]['band_width'], lp=loose_profit,
            loose_trades=format_trades_list(s[loose_key]['trades'], '#ff9f43'))

    html += """
    </table>
  </div>
</div>""".format(
            narrow_str=('{:+,.0f}'.format(narrow_total)), narrow_color='#6bcb77' if narrow_total >= 0 else '#ee5a24',
            mid_str=('{:+,.0f}'.format(mid_total)), mid_color='#ffd93d' if mid_total >= 0 else '#ee5a24',
            loose_str=('{:+,.0f}'.format(loose_total)), loose_color='#ff9f43' if loose_total >= 0 else '#ee5a24',
            best_str=('{:+,.0f}'.format(best_total)))
    return html


if __name__ == '__main__':
    main()
