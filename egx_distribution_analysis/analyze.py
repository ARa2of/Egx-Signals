import numpy as np
import pandas as pd
from tvDatafeed import TvDatafeed, Interval
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest import run_backtest, backtest_fragment, fetch_data

TICKERS = ['MFPC', 'MASR', 'ETEL', 'EFIH', 'ORHD', 'CPCI', 'RMDA', 'ARCC', 'OBRI', 'EGAS', 'ADIB', 'EGAL', 'BONY', 'ENGC']
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
FEE_PER_SIDE = 0.003
ROUND_TRIP_FEE = FEE_PER_SIDE * 2
DARK_BG = '#1a1a2e'
DARK_BG2 = '#16213e'


def fetch_data(ticker, n_bars=1000):
    # kept for compatibility; the shared implementation lives in backtest.py
    # (reuses one TvDatafeed client + falls back to the on-disk cache)
    from backtest import fetch_data as _fetch
    return _fetch(ticker, n_bars=n_bars)


def last_n_sessions(data, n=5):
    """Return the last n trading sessions (calendar dates present in the data).

    Fixes the old calendar-day Timedelta(days=7) window, which silently shrank
    when sessions were missed (e.g. EGX holidays).
    """
    dates = sorted(set(data.index.date))
    keep = set(dates[-n:])
    return data[[d in keep for d in data.index.date]]


def compute_stats(data):
    close = data['close']
    volume = data['volume']
    recent = last_n_sessions(data, 5)
    close_5d = recent['close']
    volume_5d = recent['volume']

    vol_at_mean_5d = 0
    if len(recent) > 0:
        bin_edges = np.linspace(close.min(), close.max(), 51)
        bin_idx = np.clip(np.digitize(close_5d.values, bin_edges) - 1, 0, 49)
        mean_bin = np.clip(np.digitize([close_5d.mean()], bin_edges)[0] - 1, 0, 49)
        vol_per_bin = np.zeros(50)
        for i, v in zip(bin_idx, volume_5d.values):
            vol_per_bin[i] += v
        vol_at_mean_5d = vol_per_bin[mean_bin]

    return {
        'ticker': data['symbol'].iloc[0].replace('EGX:', ''),
        'period': "{} to {}".format(data.index[0].strftime('%Y-%m-%d'), data.index[-1].strftime('%Y-%m-%d')),
        'period_5d': "{} to {}".format(recent.index[0].strftime('%Y-%m-%d'), recent.index[-1].strftime('%Y-%m-%d')),
        'sessions': len(set(data.index.date)),
        'sessions_5d': len(set(recent.index.date)),
        'bars': len(data),
        'bars_5d': len(recent),
        'mean': close.mean(),
        'std': close.std(),
        'median': close.median(),
        'min': close.min(),
        'max': close.max(),
        'range': close.max() - close.min(),
        'cv': (close.std() / close.mean()) * 100,
        'mean_5d': close_5d.mean(),
        'std_5d': close_5d.std(),
        'min_5d': close_5d.min(),
        'max_5d': close_5d.max(),
        'range_5d': close_5d.max() - close_5d.min(),
        'cv_5d': (close_5d.std() / close_5d.mean()) * 100,
        'total_volume': volume.sum(),
        'avg_volume': volume.mean(),
        'max_volume': volume.max(),
        'current': close.iloc[-1],
        'pct_from_mean': ((close.iloc[-1] - close.mean()) / close.mean()) * 100,
        'pct_from_mean_5d': ((close_5d.iloc[-1] - close_5d.mean()) / close_5d.mean()) * 100 if len(close_5d) > 0 else 0,
        'vol_at_mean_5d': vol_at_mean_5d,
    }



def count_oscillations(close_series, threshold):
    cross_up = (close_series > threshold).astype(int)
    diffs = cross_up.diff()
    return int(diffs.sum())


def compute_trading_plan(stats, data):
    mean = stats['mean']
    std = stats['std']
    mean_5d = stats['mean_5d']
    std_5d = stats['std_5d']
    current = stats['current']
    cv = stats['cv']
    cv_5d = stats['cv_5d']

    dist = current - mean_5d
    dist_sigma = dist / std_5d if std_5d > 0 else 0

    support_1 = mean_5d - std_5d
    support_2 = mean_5d - 2 * std_5d
    resistance_1 = mean_5d + std_5d
    resistance_2 = mean_5d + 2 * std_5d

    close_5d = last_n_sessions(data, 5)['close']
    close_1m = data['close']

    # Count how many times price crosses into/out of the buy zone in the last month
    buy_zone_upper = resistance_1
    buy_zone_lower = support_1
    crossings_1m = count_oscillations(close_1m, buy_zone_upper) + count_oscillations(close_1m, buy_zone_lower)
    swings_per_month = max(1, crossings_1m // 2)

    avg_vol = stats['avg_volume']
    if avg_vol > 100000:
        liquidity_bonus = 5
    elif avg_vol > 50000:
        liquidity_bonus = 3
    elif avg_vol > 20000:
        liquidity_bonus = 1
    else:
        liquidity_bonus = -3

    # Swing trade entry logic
    if dist_sigma <= -2:
        action = 'STRONG BUY'
        action_color = '#00ff88'
        strategy = 'Deep Value - Buy at -2 Sigma Support'
        confidence_base = 80
        entry = current
        exit_target = mean_5d
    elif dist_sigma <= -1:
        action = 'BUY'
        action_color = '#6bcb77'
        strategy = 'Buy at -1 Sigma Support'
        confidence_base = 72
        entry = current
        exit_target = mean_5d
    elif dist_sigma <= -0.3:
        action = 'BUY'
        action_color = '#6bcb77'
        strategy = 'Below Mean - Good Entry'
        confidence_base = 65
        entry = current
        exit_target = resistance_1
    elif dist_sigma <= 0.3:
        action = 'WAIT'
        action_color = '#ffd93d'
        strategy = 'At Mean - Wait for Pullback'
        confidence_base = 40
        entry = support_1
        exit_target = resistance_1
    elif dist_sigma <= 1:
        action = 'WAIT'
        action_color = '#ffd93d'
        strategy = 'Above Mean - Wait for Pullback'
        confidence_base = 38
        entry = support_1
        exit_target = resistance_2
    else:
        action = 'AVOID'
        action_color = '#ee5a24'
        strategy = 'Overbought - Too Expensive'
        confidence_base = 30
        entry = support_1
        exit_target = resistance_1

    # Confidence adjustments
    if cv_5d < 1.5:
        confidence_base += 8
    elif cv_5d < 2.5:
        confidence_base += 4
    elif cv_5d > 6:
        confidence_base -= 6
    elif cv_5d > 4:
        confidence_base -= 3

    if swings_per_month >= 6:
        confidence_base += 8
    elif swings_per_month >= 4:
        confidence_base += 4

    confidence_base += liquidity_bonus

    if abs(((mean_5d - mean) / mean) * 100) < 1:
        confidence_base += 5

    confidence = min(max(confidence_base, 25), 92)

    # Duration: how long to hold from entry to exit
    if 'BUY' in action:
        dist_to_target = abs(exit_target - entry)
        dist_sigma_to_target = dist_to_target / std_5d if std_5d > 0 else 1
        base_duration = max(1, min(5, int(dist_sigma_to_target * 1.5)))
        if cv_5d < 2:
            base_duration = max(1, base_duration - 1)
        elif cv_5d > 5:
            base_duration = min(8, base_duration + 1)
        est_sessions = base_duration
    else:
        est_sessions = 0

    gross_gain_pct = abs(exit_target - entry) / entry * 100 if entry > 0 else 0
    net_gain_pct = gross_gain_pct - (ROUND_TRIP_FEE * 100)

    stop_loss = support_2
    if stop_loss >= current:
        # Deep-value entries (-2σ or lower): band stop would sit at/above entry.
        # Keep 1σ of risk below entry, consistent with the band logic.
        stop_loss = current - std_5d
    stop_loss_pct = abs(entry - stop_loss) / entry * 100 if entry > 0 else 0

    risk = abs(entry - stop_loss) if entry > 0 else 0
    risk_pct = risk / entry * 100 if entry > 0 else 0
    reward = abs(exit_target - entry)
    rr_ratio = reward / risk if risk > 0 else 0

    return {
        'action': action, 'action_color': action_color, 'strategy': strategy,
        'entry': entry, 'exit_target': exit_target,
        'stop_loss': stop_loss, 'stop_loss_pct': stop_loss_pct,
        'gross_gain_pct': gross_gain_pct, 'net_gain_pct': net_gain_pct,
        'fees_pct': ROUND_TRIP_FEE * 100, 'confidence': confidence,
        'est_sessions': est_sessions,
        'risk_pct': risk_pct, 'rr_ratio': rr_ratio,
        'support_1': support_1, 'support_2': support_2,
        'resistance_1': resistance_1, 'resistance_2': resistance_2,
        'dist_sigma': dist_sigma,
        'swings_per_month': swings_per_month,
        'trade_range': resistance_1 - buy_zone_lower,
        'trade_range_pct': ((resistance_1 - buy_zone_lower) / mean_5d * 100) if mean_5d > 0 else 0,
    }


def create_price_chart(data, stats):
    close_all = data['close'].values
    mean = stats['mean']
    std = stats['std']
    n_sess = stats.get('sessions', len(set(data.index.date)))
    recent = last_n_sessions(data, 5)
    close_5d = recent['close'].values
    mean_5d = stats['mean_5d']
    std_5d = stats['std_5d']

    n_bins = 50
    bin_edges = np.linspace(close_all.min(), close_all.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_idx_all = np.clip(np.digitize(close_all, bin_edges) - 1, 0, n_bins - 1)
    count_1m = np.zeros(n_bins)
    for i in bin_idx_all:
        count_1m[i] += 1

    bin_idx_5d = np.clip(np.digitize(close_5d, bin_edges) - 1, 0, n_bins - 1)
    count_5d = np.zeros(n_bins)
    for i in bin_idx_5d:
        count_5d[i] += 1

    colors_5d = []
    for bc in bin_centers:
        dist = abs(bc - mean_5d)
        if dist <= std_5d:
            colors_5d.append('#6bcb77')
        elif dist <= 2 * std_5d:
            colors_5d.append('#ffd93d')
        elif dist <= 3 * std_5d:
            colors_5d.append('#ff9f43')
        else:
            colors_5d.append('#ee5a24')

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=bin_centers, y=count_1m, name='Full ({} sess)'.format(n_sess),
        marker_color='#00d4ff', opacity=0.25,
        hovertemplate='Price: %{x:.2f}<br>Count: %{y:.0f}<extra>Full Window</extra>'
    ))

    fig.add_trace(go.Bar(
        x=bin_centers, y=count_5d, name='Last 5 Sessions',
        marker_color=colors_5d, opacity=0.85,
        hovertemplate='Price: %{x:.2f}<br>Count: %{y:.0f}<extra>5 Sessions</extra>'
    ))

    x = np.linspace(close_all.min() - std, close_all.max() + std, 300)
    bin_width = (np.max(close_all) - np.min(close_all)) / n_bins

    bell_1m = len(close_all) * bin_width * (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    fig.add_trace(go.Scatter(
        x=x, y=bell_1m, name='Full Normal Fit',
        line=dict(color='#ff6b6b', width=1.5, dash='dash'), opacity=0.6,
        hovertemplate='Price: %{x:.2f}<br>Expected: %{y:.1f}<extra>Full Fit</extra>'
    ))

    bell_5d = len(close_5d) * bin_width * (1 / (std_5d * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_5d) / std_5d) ** 2)
    fig.add_trace(go.Scatter(
        x=x, y=bell_5d, name='5D Normal Fit',
        line=dict(color='#ff6b6b', width=2.5), opacity=1.0,
        hovertemplate='Price: %{x:.2f}<br>Expected: %{y:.1f}<extra>5D Fit</extra>'
    ))

    for val, label, color, dash in [
        (mean_5d, 'Mean {:.2f}'.format(mean_5d), '#ffd93d', 'solid'),
        (mean_5d - std_5d, '-1\u03C3 {:.2f}'.format(mean_5d - std_5d), '#6bcb77', 'dash'),
        (mean_5d + std_5d, '+1\u03C3 {:.2f}'.format(mean_5d + std_5d), '#6bcb77', 'dash'),
        (mean_5d - 2 * std_5d, '-2\u03C3 {:.2f}'.format(mean_5d - 2 * std_5d), '#ff9f43', 'dot'),
        (mean_5d + 2 * std_5d, '+2\u03C3 {:.2f}'.format(mean_5d + 2 * std_5d), '#ff9f43', 'dot'),
    ]:
        fig.add_vline(x=val, line=dict(color=color, width=1.5, dash=dash), opacity=0.8,
                      annotation=dict(text=label, font=dict(color=color, size=10), yshift=15, showarrow=False))

    current = stats['current']
    fig.add_vline(x=current, line=dict(color='white', width=2, dash='solid'), opacity=0.9,
                  annotation=dict(text='Current {:.2f}'.format(current), font=dict(color='white', size=11, family='Arial Black'), yshift=-15, showarrow=False))

    fig.update_layout(
        title=dict(text='{} - Price Distribution (Full {} Sessions + 5D)'.format(stats['ticker'], n_sess), font=dict(color='white', size=14)),
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG2,
        xaxis=dict(title='Price (EGP)', color='white', gridcolor='#2a2a4a'),
        yaxis=dict(title='Number of Occurrences', color='white', gridcolor='#2a2a4a'),
        legend=dict(bgcolor='rgba(22,33,62,0.8)', bordercolor='#2a2a4a', font=dict(color='white', size=10), x=0.99, y=0.99, xanchor='right', yanchor='top'),
        barmode='overlay', height=400, margin=dict(l=60, r=65, t=50, b=40),
        dragmode=False,
    )

    return fig.to_html(full_html=False, include_plotlyjs=False, config={'scrollZoom': True, 'displayModeBar': False})


def create_volume_chart(data, stats):
    close_all = data['close'].values
    volume_all = data['volume'].values
    mean_5d = stats['mean_5d']
    std_5d = stats['std_5d']
    n_sess = stats.get('sessions', len(set(data.index.date)))
    recent = last_n_sessions(data, 5)
    close_5d = recent['close'].values
    volume_5d = recent['volume'].values

    n_bins = 50
    bin_edges = np.linspace(close_all.min(), close_all.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_idx_all = np.clip(np.digitize(close_all, bin_edges) - 1, 0, n_bins - 1)
    vol_1m = np.zeros(n_bins)
    for i, v in zip(bin_idx_all, volume_all):
        vol_1m[i] += v

    bin_idx_5d = np.clip(np.digitize(close_5d, bin_edges) - 1, 0, n_bins - 1)
    vol_5d = np.zeros(n_bins)
    for i, v in zip(bin_idx_5d, volume_5d):
        vol_5d[i] += v

    colors_5d = []
    for bc in bin_centers:
        dist = abs(bc - mean_5d)
        if dist <= std_5d:
            colors_5d.append('#6bcb77')
        elif dist <= 2 * std_5d:
            colors_5d.append('#ffd93d')
        elif dist <= 3 * std_5d:
            colors_5d.append('#ff9f43')
        else:
            colors_5d.append('#ee5a24')

    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(go.Bar(
        x=bin_centers, y=vol_1m, name='Full ({} sess)'.format(n_sess),
        marker_color='#00d4ff', opacity=0.25,
        hovertemplate='Price: %{x:.2f}<br>Volume: %{y:,.0f}<extra>Full Window</extra>'
    ))

    fig.add_trace(go.Bar(
        x=bin_centers, y=vol_5d, name='5D Volume',
        marker_color=colors_5d, opacity=0.85,
        hovertemplate='Price: %{x:.2f}<br>Volume: %{y:,.0f}<extra>5 Sessions</extra>'
    ))

    for val, label, color, dash in [
        (mean_5d, 'Mean {:.2f}'.format(mean_5d), '#ffd93d', 'solid'),
        (mean_5d - std_5d, '-1\u03C3 {:.2f}'.format(mean_5d - std_5d), '#6bcb77', 'dash'),
        (mean_5d + std_5d, '+1\u03C3 {:.2f}'.format(mean_5d + std_5d), '#6bcb77', 'dash'),
        (mean_5d - 2 * std_5d, '-2\u03C3 {:.2f}'.format(mean_5d - 2 * std_5d), '#ff9f43', 'dot'),
        (mean_5d + 2 * std_5d, '+2\u03C3 {:.2f}'.format(mean_5d + 2 * std_5d), '#ff9f43', 'dot'),
    ]:
        fig.add_vline(x=val, line=dict(color=color, width=1.5, dash=dash), opacity=0.8,
                      annotation=dict(text=label, font=dict(color=color, size=10), yshift=15, showarrow=False))

    current = stats['current']
    fig.add_vline(x=current, line=dict(color='white', width=2, dash='solid'), opacity=0.9,
                  annotation=dict(text='Current {:.2f}'.format(current), font=dict(color='white', size=11, family='Arial Black'), yshift=-15, showarrow=False))

    fig.update_layout(
        title=dict(text='{} - Volume Concentration (Full {} Sessions + 5D)'.format(stats['ticker'], n_sess), font=dict(color='white', size=14)),
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG2,
        xaxis=dict(title='Price (EGP)', color='white', gridcolor='#2a2a4a'),
        yaxis=dict(title='Total Volume', color='white', gridcolor='#2a2a4a', tickformat=',.0f'),
        legend=dict(bgcolor='rgba(22,33,62,0.8)', bordercolor='#2a2a4a', font=dict(color='white', size=10), x=0.99, y=0.99, xanchor='right', yanchor='top'),
        barmode='overlay', height=350, margin=dict(l=50, r=65, t=50, b=40),
        dragmode=False,
    )

    return fig.to_html(full_html=False, include_plotlyjs=False, config={'scrollZoom': True, 'displayModeBar': False})



def build_ticker_section(stats, price_chart, volume_chart, plan, mean_shift):
    action = plan['action']
    action_color = plan['action_color']
    confidence = plan['confidence']
    conf_color = '#6bcb77' if confidence >= 70 else '#ffd93d' if confidence >= 50 else '#ee5a24'

    if 'BUY' in action:
        action_bg = 'rgba(107,203,119,0.15)'
    elif 'WAIT' in action or 'AVOID' in action:
        action_bg = 'rgba(238,90,36,0.15)'
    elif 'HOLD' in action:
        action_bg = 'rgba(0,212,255,0.15)'
    elif 'PROFIT' in action or 'REDUCE' in action:
        action_bg = 'rgba(255,159,67,0.15)'
    else:
        action_bg = 'rgba(255,255,255,0.1)'

    net_color = '#6bcb77' if plan['net_gain_pct'] > 0 else '#ee5a24'

    section = """  <div class="ticker-section">
    <div class="ticker-header" onclick="toggleSection(this)">
      <span class="ticker-name">{ticker}</span>
      <span class="sigma-badge" style="color:{action_color};background:{action_bg}">{action}</span>
      <span class="current-price">{current:.2f} EGP</span>
      <span class="pct-badge" style="color:{pos_color}">{pct:+.2f}%</span>
      <span class="arrow">&#9660;</span>
    </div>
    <div class="accordion-content">
      <div class="stats-grid">
        <div class="stat-box"><div class="stat-label">Period</div><div class="stat-value">{period}</div></div>
        <div class="stat-box"><div class="stat-label">5D Mean</div><div class="stat-value">{mean_5d:.2f}</div></div>
        <div class="stat-box"><div class="stat-label">5D Std</div><div class="stat-value">{std_5d_v:.2f}</div></div>
        <div class="stat-box"><div class="stat-label">5D Range</div><div class="stat-value">{min_5d:.2f} - {max_5d:.2f}</div></div>
      </div>
      <div class="stats-grid">
        <div class="stat-box"><div class="stat-label">Mean Shift (Full vs 5D)</div><div class="stat-value">{mean_shift:+.1f}%</div></div>
      </div>
      <div class="sigma-section">
        <div class="sigma-bar">
          <span class="level" style="color:#ee5a24">S2: {support_2:.2f}</span>
          <span class="level" style="color:#ff9f43">S1: {support_1:.2f}</span>
          <span class="level" style="color:#ffd93d">MEAN: {mean_5d:.2f}</span>
          <span class="level" style="color:#6bcb77">R1: {resistance_1:.2f}</span>
          <span class="level" style="color:#00ff88">R2: {resistance_2:.2f}</span>
        </div>
      </div>
      <div class="chart-container">
        {price_chart}
      </div>
      <div class="chart-container">
        {volume_chart}
      </div>
      <div class="plan-section">
        <h3>Swing Trade Plan</h3>
        <div class="plan-grid">
          <div class="plan-box action-box" style="border-color:{action_color};background:{action_bg}"><div class="plan-label">Signal</div><div class="plan-value" style="color:{action_color};font-size:24px">{action}</div></div>
          <div class="plan-box"><div class="plan-label">Strategy</div><div class="plan-value small">{strategy}</div></div>
          <div class="plan-box"><div class="plan-label">Confidence</div><div class="plan-value" style="color:{conf_color}">{confidence}%</div></div>
          <div class="plan-box"><div class="plan-label">Hold Period</div><div class="plan-value">{est_sessions} sessions</div></div>
          <div class="plan-box" style="grid-column:1/-1"><div class="plan-label">Confidence Basis</div><div class="plan-value small">{confidence_basis}</div></div>
        </div>
        <div class="plan-grid">
          <div class="plan-box"><div class="plan-label">Entry (Buy Zone)</div><div class="plan-value" style="color:#6bcb77">{entry:.2f}</div></div>
          <div class="plan-box"><div class="plan-label">Exit Target</div><div class="plan-value" style="color:#6bcb77">{exit_target:.2f}</div></div>
          <div class="plan-box"><div class="plan-label">Stop Loss</div><div class="plan-value" style="color:#ee5a24">{stop_loss:.2f}</div></div>
          <div class="plan-box"><div class="plan-label">Fees (Round Trip)</div><div class="plan-value">{fees_pct:.1f}%</div></div>
        </div>
        <div class="plan-grid">
          <div class="plan-box"><div class="plan-label">Trade Range</div><div class="plan-value">{trade_range:.2f} EGP ({trade_range_pct:.1f}%)</div></div>
          <div class="plan-box"><div class="plan-label">Net Gain %</div><div class="plan-value" style="color:#6bcb77">{net_gain_pct:.2f}%</div></div>
          <div class="plan-box"><div class="plan-label">Reward:Risk</div><div class="plan-value">{rr_ratio:.1f}</div></div>
        </div>
        <div class="calc-grid" data-entry="{entry:.4f}" data-exit="{exit_target:.4f}" data-stop="{stop_loss:.4f}" data-net-pct="{net_gain_pct:.4f}" data-fees="{fees_pct:.4f}">
          <div class="plan-box"><div class="plan-label">Shares Bought</div><div class="plan-value calc-shares">-</div></div>
          <div class="plan-box"><div class="plan-label">Net Gain (EGP)</div><div class="plan-value calc-gain" style="color:#6bcb77">-</div></div>
          <div class="plan-box"><div class="plan-label">Loss if Stop (EGP)</div><div class="plan-value calc-loss" style="color:#ee5a24">-</div></div>
          <div class="plan-box"><div class="plan-label">Win/Loss Ratio</div><div class="plan-value calc-ratio">-</div></div>
        </div>
      </div>
    </div>
  </div>""".format(
        ticker=stats['ticker'], current=stats['current'], pct=stats['pct_from_mean_5d'],
        pos_color='#6bcb77' if stats['pct_from_mean_5d'] < 0 else '#ee5a24',
        period=stats['period'],
        mean_5d=stats['mean_5d'], std_5d_v=stats['std_5d'],
        min_5d=stats['min_5d'], max_5d=stats['max_5d'],
        mean_shift=mean_shift,
        price_chart=price_chart, volume_chart=volume_chart,
        action=action, action_color=action_color, action_bg=action_bg,
        strategy=plan['strategy'], confidence=confidence, conf_color=conf_color,
        confidence_basis=plan.get('confidence_basis', 'heuristic'),
        est_sessions=plan['est_sessions'], entry=plan['entry'], exit_target=plan['exit_target'],
        stop_loss=plan['stop_loss'], fees_pct=plan['fees_pct'],
        gross_gain_pct=plan['gross_gain_pct'], net_gain_pct=plan['net_gain_pct'],
        rr_ratio=plan['rr_ratio'], net_color=net_color,
        support_1=plan['support_1'], support_2=plan['support_2'],
        resistance_1=plan['resistance_1'], resistance_2=plan['resistance_2'],
        trade_range=plan['trade_range'], trade_range_pct=plan['trade_range_pct'],
    )
    return section


def build_combined_html(ticker_results, data_date='', backtest_html=''):
    sections = []
    for r in ticker_results:
        sections.append(r['section'])

    sections_html = '\n'.join(sections)

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EGX Distribution Analysis - Combined Report</title>
<script src="plotly.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f0f23; color: #e0e0e0; padding: 10px; }}
  h1 {{ text-align: center; margin: 10px 0; font-size: 1.4em; color: #ffd93d; }}
  .subtitle {{ text-align: center; color: #888; margin-bottom: 15px; font-size: 0.9em; }}
  .ticker-section {{ background: #1a1a2e; border: 1px solid #2a2a4a; border-radius: 8px; margin-bottom: 8px; overflow: hidden; }}
  .ticker-header {{ display: flex; align-items: center; padding: 10px 15px; cursor: pointer; gap: 12px; user-select: none; }}
  .ticker-header:hover {{ background: #16213e; }}
  .ticker-name {{ font-weight: 700; font-size: 1.2em; min-width: 70px; }}
  .current-price {{ font-weight: 600; font-size: 1.1em; margin-left: auto; }}
  .sigma-badge {{ padding: 3px 10px; border-radius: 4px; font-weight: 600; font-size: 0.85em; }}
  .pct-badge {{ font-weight: 600; font-size: 0.9em; }}
  .arrow {{ font-size: 0.8em; transition: transform 0.2s; color: #888; }}
  .accordion-content {{ display: none; padding: 0 15px 15px; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 6px; margin: 8px 0; }}
  .stat-box {{ background: #16213e; padding: 6px 10px; border-radius: 4px; }}
  .stat-label {{ font-size: 0.7em; color: #888; text-transform: uppercase; }}
  .stat-value {{ font-size: 0.85em; font-weight: 600; }}
  .chart-container {{ margin: 10px 0; background: #1a1a2e; border-radius: 6px; overflow: hidden; }}
  .sigma-bar {{ display: flex; justify-content: space-between; padding: 8px 12px; background: #16213e; border-radius: 4px; margin: 8px 0; font-size: 0.85em; font-weight: 600; }}
  .level {{ padding: 2px 6px; }}
  .plan-section {{ background: #16213e; border: 1px solid #2a2a4a; border-radius: 6px; padding: 12px; margin-top: 10px; }}
  .plan-section h3 {{ color: #ffd93d; margin-bottom: 10px; font-size: 1em; }}
  .plan-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 8px; margin-bottom: 8px; }}
  .plan-box {{ background: #1a1a2e; padding: 10px; border-radius: 6px; border-left: 3px solid #2a2a4a; }}
  .plan-box.action-box {{ border-left-width: 4px; }}
  .plan-label {{ font-size: 0.7em; color: #888; text-transform: uppercase; margin-bottom: 2px; }}
  .plan-value {{ font-size: 1.1em; font-weight: 700; }}
  .plan-value.small {{ font-size: 0.85em; font-weight: 500; }}
  .invest-bar {{ background: #16213e; border: 1px solid #2a2a4a; border-radius: 8px; padding: 12px 20px; margin: 15px auto; max-width: 500px; display: flex; align-items: center; gap: 12px; justify-content: center; }}
  .invest-bar label {{ font-weight: 600; color: #ffd93d; font-size: 0.95em; }}
  .invest-bar input {{ background: #1a1a2e; border: 1px solid #2a2a4a; color: white; padding: 8px 14px; border-radius: 6px; font-size: 1.1em; font-weight: 700; width: 140px; text-align: center; }}
  .invest-bar input:focus {{ outline: none; border-color: #ffd93d; }}
  .invest-bar span {{ color: #888; font-size: 0.85em; }}
  .calc-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 8px; margin-top: 8px; padding-top: 8px; border-top: 1px solid #2a2a4a; }}
  .data-date {{ text-align: center; color: #ffd93d; font-size: 0.85em; margin-bottom: 10px; }}
  .backtest-section {{ background: #1a1a2e; border: 1px solid #2a2a4a; border-radius: 8px; margin: 15px 0; overflow: hidden; }}
  .backtest-header {{ display: flex; align-items: center; padding: 12px 15px; cursor: pointer; gap: 15px; }}
  .backtest-header:hover {{ background: #16213e; }}
  .bt-summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 8px; margin: 10px 0; }}
  .bt-box {{ background: #16213e; padding: 10px; border-radius: 6px; border-left: 3px solid #2a2a4a; text-align: center; }}
  .bt-label {{ font-size: 0.7em; color: #888; text-transform: uppercase; }}
  .bt-value {{ font-size: 1.1em; font-weight: 700; margin-top: 2px; }}
  .bt-table {{ width: 100%; border-collapse: collapse; font-size: 0.85em; margin-top: 10px; }}
  .bt-table th {{ background: #16213e; color: #ffd93d; padding: 6px 4px; text-align: center; border-bottom: 2px solid #2a2a4a; }}
  .bt-table td {{ padding: 5px 4px; text-align: center; border-bottom: 1px solid #2a2a4a; }}
  .bt-table tr:hover {{ background: #16213e; }}
  .ease-badge {{ padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.8em; }}
  .ease-Easy {{ background: rgba(107,203,119,0.2); color: #6bcb77; }}
  .ease-Moderate {{ background: rgba(255,217,61,0.2); color: #ffd93d; }}
  .ease-Difficult {{ background: rgba(255,159,67,0.2); color: #ff9f43; }}
  .ease-Very-Difficult {{ background: rgba(238,90,36,0.2); color: #ee5a24; }}
  .bt-detail-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 8px; padding: 8px 0; }}
  .bt-detail-card {{ background: #1a1a2e; border-radius: 6px; padding: 10px; }}
  .bt-detail-title {{ font-weight: 700; margin-bottom: 6px; font-size: 0.9em; }}
  .bt-detail-row {{ display: flex; justify-content: space-between; padding: 2px 0; font-size: 0.82em; }}
  .bt-detail-row span:first-child {{ color: #888; }}
  .bt-trades-list {{ margin-top: 8px; border-top: 1px solid #2a2a4a; padding-top: 6px; }}
  .bt-trades-empty {{ color: #888; font-style: italic; font-size: 0.8em; margin-top: 8px; }}
  .bt-trade-row {{ display: flex; gap: 8px; padding: 3px 0; font-size: 0.78em; align-items: center; flex-wrap: wrap; }}
  .bt-trade-num {{ color: #888; min-width: 20px; }}
  .bt-trade-time {{ color: #aaa; min-width: 220px; }}
  .bt-trade-prices {{ color: #ccc; min-width: 140px; }}
  .bt-trade-profit {{ font-weight: 600; min-width: 100px; text-align: right; }}
</style>
</head>
<body>
<h1>EGX Distribution Analysis Report</h1>
<p class="data-date">Data retrieved: {data_date}</p>
<p class="subtitle">Sorted by Signal: BUY (high conf) first, then WAIT, then AVOID | 5-min data | full-window + 5D overlay</p>
<div class="invest-bar">
  <label>Investment Amount:</label>
  <input type="number" id="investAmount" value="10000" min="1" oninput="calculateAll()">
  <span>EGP</span>
</div>
{sections}
{backtest_html}
<script>
function toggleSection(header) {{
  var content = header.nextElementSibling;
  var arrow = header.querySelector('.arrow');
  if (content.style.display === 'block') {{
    content.style.display = 'none';
    arrow.innerHTML = '&#9660;';
  }} else {{
    content.style.display = 'block';
    arrow.innerHTML = '&#9650;';
    content.querySelectorAll('.js-plotly-plot').forEach(function(fig) {{
      Plotly.Plots.resize(fig);
    }});
  }}
}}
function calculateAll() {{
  var amount = parseFloat(document.getElementById('investAmount').value) || 0;
  document.querySelectorAll('.calc-grid').forEach(function(grid) {{
    var entry = parseFloat(grid.dataset.entry);
    var exit = parseFloat(grid.dataset.exit);
    var stop = parseFloat(grid.dataset.stop);
    var fees = parseFloat(grid.dataset.fees);
    if (!entry || entry <= 0) return;
    var shares = Math.floor(amount / entry);
    var invested = shares * entry;
    var grossGain = shares * (exit - entry);
    var feeCost = invested * (fees / 100);
    var netGain = grossGain - feeCost;
    var loss = shares * (entry - stop);
    var ratio = loss > 0 ? (netGain / loss) : 0;
    grid.querySelector('.calc-shares').textContent = shares.toLocaleString();
    var gainEl = grid.querySelector('.calc-gain');
    gainEl.textContent = (netGain >= 0 ? '+' : '') + netGain.toFixed(2) + ' EGP';
    gainEl.style.color = netGain >= 0 ? '#6bcb77' : '#ee5a24';
    grid.querySelector('.calc-loss').textContent = '-' + loss.toFixed(2) + ' EGP';
    grid.querySelector('.calc-ratio').textContent = ratio.toFixed(1) + 'x';
  }});
}}
calculateAll();
function toggleBtDetail(row) {{
  var detail = row.nextElementSibling;
  var arrow = row.querySelector('td:first-child');
  if (detail && detail.classList.contains('bt-detail')) {{
    if (detail.style.display === 'table-row') {{
      detail.style.display = 'none';
      arrow.innerHTML = '&#9654;';
    }} else {{
      detail.style.display = 'table-row';
      arrow.innerHTML = '&#9660;';
    }}
  }}
}}
</script>
</body>
</html>""".format(sections=sections_html, data_date=data_date, backtest_html=backtest_html)
    return html


def analyze_ticker(ticker, data):
    stats = compute_stats(data)
    plan = compute_trading_plan(stats, data)
    mean_shift = ((stats['mean_5d'] - stats['mean']) / stats['mean']) * 100
    price_chart = create_price_chart(data, stats)
    volume_chart = create_volume_chart(data, stats)
    return {
        'stats': stats, 'plan': plan,
        'confidence': plan['confidence'], 'action': plan['action'],
        'data': data, 'price_chart': price_chart,
        'volume_chart': volume_chart, 'mean_shift': mean_shift,
    }


def main():
    print("EGX Distribution Analysis (Interactive)")
    print("=" * 50)

    results = []
    for ticker in TICKERS:
        try:
            print("  Fetching {}...".format(ticker))
            data = fetch_data(ticker)
            print("  Got {} bars, generating charts...".format(len(data)))
            r = analyze_ticker(ticker, data)
            results.append(r)
        except Exception as e:
            print("  ERROR: {}".format(e))
        time.sleep(0.3)

    def sort_key(r):
        action = r['action']
        if 'BUY' in action:
            group = 0
        elif 'WAIT' in action:
            group = 1
        else:
            group = 2
        return (group, -r['confidence'])

    results.sort(key=sort_key)

    data_date = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')

    print("\nRunning backtest...")
    fetched_data = {r['stats']['ticker']: r['data'] for r in results}
    bt_results, narrow_key, mid_key, loose_key = run_backtest(TICKERS, fetched_data=fetched_data)
    bt_html = backtest_fragment(bt_results, narrow_key, mid_key, loose_key)

    # P3d: anchor BUY-signal confidence to the matching band's backtest win
    # rate (50/50 blend). WAIT/AVOID and bands with too few trades stay
    # heuristic-only, with the reason recorded in confidence_basis.
    bt_by_ticker = {b['ticker']: b for b in bt_results}
    for r in results:
        p = r['plan']
        ds = p['dist_sigma']
        if 'BUY' not in p['action']:
            p['confidence_basis'] = 'heuristic only (no entry signal to anchor)'
            continue
        band_key = loose_key if ds <= -2 else mid_key if ds <= -1 else narrow_key
        bt = bt_by_ticker.get(r['stats']['ticker'])
        sd = bt['strategies'].get(band_key) if bt else None
        n_tr = sd['total_trades'] if sd else 0
        if not sd or n_tr < 3:
            p['confidence_basis'] = 'heuristic only ({} band: n={} trades)'.format(
                band_key.split(' ')[0].lower(), n_tr)
            continue
        heur = p['confidence']
        wr = sd['win_rate']
        blended = int(min(max(round(0.5 * heur + 0.5 * wr), 25), 92))
        p['confidence'] = blended
        p['confidence_basis'] = 'heur {:.0f} + {:.0f}% {} wr (n={}) -> {:.0f}'.format(
            heur, wr, band_key.split(' ')[0].lower(), n_tr, blended)

    # rebuild sections + re-sort now that confidences are anchored
    for r in results:
        r['section'] = build_ticker_section(
            r['stats'], r['price_chart'], r['volume_chart'], r['plan'], r['mean_shift'])
        r['confidence'] = r['plan']['confidence']
        r['action'] = r['plan']['action']
    results.sort(key=sort_key)

    print("\nGenerating combined HTML...")
    html = build_combined_html(results, data_date=data_date, backtest_html=bt_html)
    output_path = os.path.join(OUTPUT_DIR, "index.html")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print("Saved: {}".format(output_path))

    import json
    json_data = {
        'generated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
        'data_date': data_date,
        'fees_per_side_pct': 0.3,
        'round_trip_fees_pct': 0.6,
        'backtest_budget_egp': 10000,
        'backtest_max_trades_per_day': 3,
        'tickers': [],
        'backtest': {
            'strategy_definitions': {
                'narrow': {'buy': 'mean - 0.5*std', 'sell': 'mean + 0.5*std'},
                'mid': {'buy': 'mean - 1*std', 'sell': 'mean + 1*std'},
                'loose': {'buy': 'mean - 2*std', 'sell': 'mean + 2*std'},
                'note': 'Levels use rolling 300-bar stats (no look-ahead). best_combo picks the best band per ticker in-sample (hindsight), not a forward-tested portfolio.',
            },
            'portfolio_totals_egp': {},
            'tickers': []
        }
    }
    for r in results:
        s = r['stats']
        p = r['plan']
        json_data['tickers'].append({
            'ticker': s['ticker'],
            'current_price': round(s['current'], 2),
            'signal': p['action'],
            'strategy': p['strategy'],
            'confidence': p['confidence'],
            'confidence_basis': p.get('confidence_basis', ''),
            'entry': round(p['entry'], 2),
            'exit_target': round(p['exit_target'], 2),
            'stop_loss': round(p['stop_loss'], 2),
            'net_gain_pct': round(p['net_gain_pct'], 2),
            'rr_ratio': round(p['rr_ratio'], 1),
            'hold_sessions': p['est_sessions'],
            'fees_pct': p['fees_pct'],
            'stats': {
                'mean_5d': round(s['mean_5d'], 2),
                'std_5d': round(s['std_5d'], 2),
                'min_5d': round(s['min_5d'], 2),
                'max_5d': round(s['max_5d'], 2),
                'cv_5d': round(s['cv_5d'], 2),
                'cv_1m': round(s['cv'], 2),
                'mean_shift': round(((s['mean_5d'] - s['mean']) / s['mean']) * 100, 2),
                'period': s['period'],
                'sessions': s['sessions'],
                'bars': s['bars'],
            },
            'levels': {
                'support_2': round(p['support_2'], 2),
                'support_1': round(p['support_1'], 2),
                'mean': round(s['mean_5d'], 2),
                'resistance_1': round(p['resistance_1'], 2),
                'resistance_2': round(p['resistance_2'], 2),
            },
        })
    narrow_total = sum(r['strategies'][narrow_key]['total_profit'] for r in bt_results)
    mid_total = sum(r['strategies'][mid_key]['total_profit'] for r in bt_results)
    loose_total = sum(r['strategies'][loose_key]['total_profit'] for r in bt_results)
    best_total = sum(r['strategies'][r['best_strategy']]['total_profit'] for r in bt_results)
    json_data['backtest']['portfolio_totals_egp'] = {
        'narrow_0_5sigma': round(narrow_total, 2),
        'mid_1sigma': round(mid_total, 2),
        'loose_2sigma': round(loose_total, 2),
        'best_combo': round(best_total, 2),
    }
    for bt in bt_results:
        ticker_bt = {
            'ticker': bt['ticker'],
            'current_price': bt['current_price'],
            'mean': bt['mean'],
            'std': bt['std'],
            'best_strategy': bt['best_strategy'],
            'ease_of_trading': bt['ease_label'],
            'strategies': {}
        }
        for sname, sd in bt['strategies'].items():
            key = 'narrow' if 'Narrow' in sname else 'mid' if 'Mid' in sname else 'loose'
            ticker_bt['strategies'][key] = {
                'label': sname,
                'buy_level': sd['buy_level'],
                'sell_level': sd['sell_level'],
                'band_width': sd['band_width'],
                'completed_trades': sd['total_trades'],
                'total_profit_egp': sd['total_profit'],
                'total_return_pct': sd['total_return_pct'],
                'win_rate_pct': sd['win_rate'],
                'crosses_buy': sd['crosses_buy'],
                'crosses_sell': sd['crosses_sell'],
                'trades': [
                    {'entry_time': t['entry_time'], 'exit_time': t['exit_time'],
                     'entry_price': t['entry_price'], 'exit_price': t['exit_price'],
                     'shares': t['shares'], 'profit_egp': t['profit'], 'return_pct': t['return_pct'],
                     'open_position': t.get('open_position', False)}
                    for t in sd['trades']
                ]
            }
        json_data['backtest']['tickers'].append(ticker_bt)
    json_path = os.path.join(OUTPUT_DIR, "data.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    print("Saved: {}".format(json_path))

    print("\n" + "=" * 65)
    print("SUMMARY (BUY signals first, then WAIT, then AVOID)")
    print("=" * 65)
    print("  {:6} | {:>8} | {:12} | {:>10} | {:>5} | {:>4}".format(
        'TICKER', 'PRICE', 'SIGNAL', 'NET GAIN', 'CONF', 'R:R'))
    print("  " + "-" * 57)
    for r in results:
        s = r['stats']
        p = r['plan']
        print("  {:6} | {:>8.2f} | {:12} | {:>7.2f}%  | {:>3}%  | {:>4.1f}".format(
            s['ticker'], s['current'], p['action'], p['net_gain_pct'],
            p['confidence'], p['rr_ratio']))
    print("\nCombined report: {}".format(output_path))


if __name__ == '__main__':
    main()
