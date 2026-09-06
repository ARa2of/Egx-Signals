"""
HTML Daily Report Generator for EGX Signals
Generates a styled HTML report for GitHub Pages
"""
from pathlib import Path
from datetime import date
from typing import Dict, List
import logging

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Explanation Generator
# ─────────────────────────────────────────────────────────────
def _generate_explanation(row):
    """Generate human-readable explanation for the recommendation."""
    parts = []
    rec = row.get("Recommendation", row.get("recommendation", "Watch"))
    score = row.get("Score", row.get("score", 0)) or 0

    # Consensus breakdown
    base_rec = row.get("Base Rec", "")
    ml_rec = row.get("ML Rec", "")
    cs_rec = row.get("ChartScan Rec", "")
    consensus_basis = row.get("Recommendation Basis", "")
    if base_rec or ml_rec or cs_rec:
        parts.append(f"<b>Consensus ({rec}):</b> Base={base_rec}, ML={ml_rec}, ChartScan={cs_rec}. {consensus_basis}.")

    # Score breakdown
    trend = row.get("score_trend", 0) or 0
    macd = row.get("score_macd", 0) or 0
    rsi = row.get("score_rsi", 0) or 0
    vol = row.get("score_volume", 0) or 0
    adi = row.get("score_adi", 0) or 0
    support = row.get("score_support", 0) or 0
    vwap = row.get("score_vwap", 0) or 0
    vp = row.get("score_volume_profile", 0) or 0

    # Top contributing factors
    scores = {
        "trend": trend, "MACD": macd, "RSI": rsi,
        "volume": vol, "ADI": adi, "support": support,
        "VWAP": vwap, "Volume Profile": vp
    }
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_factors = [f"{k} ({v:.0f}/100)" for k, v in sorted_scores[:3] if v > 5]

    if top_factors:
        parts.append(f"<b>Strongest signals:</b> {', '.join(top_factors)}.")

    # Trend context
    close = row.get("Current EGP Price", row.get("close"))
    regime = row.get("regime", "")
    ema20 = row.get("ema_20")
    ema50 = row.get("ema_50")

    if close and ema20 and ema50:
        if close > ema20 > ema50:
            parts.append("Price is above both EMAs — strong uptrend alignment.")
        elif close < ema20 < ema50:
            parts.append("Price is below both EMAs — downtrend structure.")
        elif close > ema20 and ema20 < ema50:
            parts.append("Price recovered above short-term EMA — potential reversal in progress.")
        elif close < ema20 and ema20 > ema50:
            parts.append("Price fell below short-term EMA — short-term weakness developing.")

    # RSI context
    rsi_val = row.get("RSI (%)", row.get("rsi"))
    if rsi_val is not None:
        try:
            rsi_val = float(rsi_val)
        except (TypeError, ValueError):
            rsi_val = None
    if rsi_val is not None:
        if rsi_val < 30:
            parts.append(f"RSI at {rsi_val:.0f} — oversold territory, bounce potential.")
        elif rsi_val > 70:
            parts.append(f"RSI at {rsi_val:.0f} — overbought, watch for pullback.")
        elif 40 <= rsi_val <= 60:
            parts.append(f"RSI neutral at {rsi_val:.0f} — no extreme momentum signal.")

    # MACD context
    macd_bull = row.get("MACD Bullish (Yes/No)", row.get("macd_bullish"))
    if macd_bull in ("Yes", True):
        parts.append("MACD bullish crossover — upward momentum building.")
    elif macd_bull in ("No", False):
        parts.append("MACD bearish — momentum fading or negative.")

    # VWAP context
    dist_vwap = row.get("Dist VWAP %", row.get("dist_vwap_pct"))
    if dist_vwap is not None:
        dv = float(dist_vwap)
        if dv > 5:
            parts.append(f"Trading {dv:.1f}% above VWAP — premium to average cost, watch for mean reversion.")
        elif dv < -5:
            parts.append(f"Trading {dv:.1f}% below VWAP — discount to average cost, potential value zone.")

    # Volume Profile context
    above_poc = row.get("Above POC", row.get("above_poc"))
    vp_poc = row.get("Volume Profile POC", row.get("vp_poc"))
    if above_poc is not None and vp_poc:
        if above_poc:
            parts.append(f"Price above POC ({vp_poc:.2f}) — bullish volume structure.")
        else:
            parts.append(f"Price below POC ({vp_poc:.2f}) — bearish volume structure.")

    # ML context
    ml_signal = row.get("ml_signal", "N/A")
    ml_conv = row.get("ml_confidence", 0) or 0
    ml_med = row.get("ml_medium_price")
    if ml_med and close and close > 0:
        ml_pct = (ml_med - close) / close * 100
        if ml_signal == "Buy" and ml_conv > 0.3:
            parts.append(f"ML model forecasts {ml_pct:+.1f}% move in 5 days (conviction: {ml_conv*100:.0f}%).")
        elif ml_signal == "Avoid" and ml_conv > 0.3:
            parts.append(f"ML model warns of {ml_pct:+.1f}% downside risk in 5 days (conviction: {ml_conv*100:.0f}%).")

    # ChartScan AI context
    cs_signal = row.get("ChartScanAI Signal", row.get("chartscan_signal", "N/A"))
    cs_conf = row.get("ChartScanAI Confidence", row.get("chartscan_confidence"))
    candle = row.get("candlestick", "")
    if cs_signal and cs_signal != "N/A" and cs_conf and cs_conf > 0.25:
        parts.append(f"ChartScan AI detected <b>{cs_signal}</b> pattern ({cs_conf*100:.0f}% confidence).")

    # Fair value context
    fv = row.get("Implied Fair Value (EGP)", row.get("fair_value_egp"))
    if fv and close and close > 0:
        fv_pct = (fv - close) / close * 100
        if fv_pct > 30:
            parts.append(f"Fair value implies {fv_pct:.0f}% upside — appears undervalued on fundamentals.")
        elif fv_pct < -30:
            parts.append(f"Fair value implies {abs(fv_pct):.0f}% downside — appears overvalued.")

    # Trade plan context
    entry = row.get("Optimal Entry Price", row.get("entry_price"))
    stop = row.get("Stop Loss", row.get("stop_loss"))
    rr = row.get("TP1 Risk/Reward", row.get("tp1_rr"))
    entry_source = row.get("Entry Source", row.get("entry_source", ""))
    entry_action = row.get("Entry Action", row.get("entry_action", ""))
    entry_high = row.get("Entry High", row.get("entry_high"))
    tp1 = row.get("Take Profit 1", row.get("tp1"))
    tp2 = row.get("Take Profit 2", row.get("tp2"))
    if entry and stop and rr:
        parts.append(f"Trade plan: entry {entry:.2f}, stop {stop:.2f}, R/R {rr:.1f}x.")
    if entry_source:
        parts.append(f"Entry sourced from <b>{entry_source}</b>.")
    if entry_high and entry and entry_high > entry:
        parts.append(f"Max entry: do not exceed <b>{entry_high:.2f}</b> (entry zone upper bound).")
    if entry_action and "BUY NOW" in str(entry_action).upper():
        parts.append("Entry is within reach — ready to execute.")
    elif entry_action and "BUY ON BREAKOUT" in str(entry_action).upper():
        parts.append(f"Entry is at resistance ({entry:.2f}) — buy if price closes above with volume confirmation.")
    elif entry_action and "WAIT FOR BREAKOUT" in str(entry_action).upper():
        parts.append(f"Entry is above current price ({entry:.2f}) — wait for breakout confirmation before entering.")
    elif entry_action and "WAIT" in str(entry_action).upper():
        parts.append("Entry is below current price — wait for pullback to execute.")
    if tp1 and entry and entry > 0:
        tp1_pct = (tp1 - entry) / entry * 100
        parts.append(f"TP1 implies +{tp1_pct:.1f}% profit from entry.")

    if not parts:
        parts.append("Insufficient data for detailed analysis.")

    return " ".join(parts)


# ─────────────────────────────────────────────────────────────
# HTML Template
# ─────────────────────────────────────────────────────────────
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<title>EGX Signals — {report_date}</title>
<style>
  :root {{
    --bg: #0f1117;
    --card: #1a1d27;
    --border: #2a2d3a;
    --text: #e1e4ea;
    --muted: #8b8fa3;
    --green: #22c55e;
    --red: #ef4444;
    --yellow: #eab308;
    --blue: #3b82f6;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
    background: var(--bg); color: var(--text);
    padding: 24px; max-width: 1100px; margin: 0 auto;
  }}
  h1 {{ font-size: 1.5rem; margin-bottom: 4px; }}
  .subtitle {{ color: var(--muted); font-size: 0.85rem; margin-bottom: 24px; }}
  .summary {{
    display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap;
  }}
  .summary-card {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px 24px; min-width: 140px;
  }}
  .summary-card .label {{ color: var(--muted); font-size: 0.75rem; text-transform: uppercase; }}
  .summary-card .value {{ font-size: 1.8rem; font-weight: 700; margin-top: 4px; }}
  .summary-card .value.buy {{ color: var(--green); }}
  .summary-card .value.avoid {{ color: var(--red); }}
  .summary-card .value.watch {{ color: var(--yellow); }}

  .market-overview {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px; margin-bottom: 24px;
  }}
  .market-title {{ color: var(--muted); font-size: 0.75rem; text-transform: uppercase; margin-bottom: 10px; }}
  .market-indices {{ display: flex; gap: 16px; flex-wrap: wrap; }}
  .index-card {{
    flex: 1; min-width: 160px; background: rgba(255,255,255,0.03);
    border: 1px solid var(--border); border-radius: 6px; padding: 12px;
  }}
  .index-card .idx-name {{ font-weight: 700; font-size: 0.85rem; margin-bottom: 4px; }}
  .index-card .idx-sentiment {{ font-size: 0.8rem; font-weight: 600; margin-bottom: 4px; }}
  .index-card .idx-sentiment.bullish {{ color: var(--green); }}
  .index-card .idx-sentiment.bearish {{ color: var(--red); }}
  .index-card .idx-sentiment.neutral {{ color: var(--yellow); }}
  .index-card .idx-detail {{ font-size: 0.72rem; color: var(--muted); }}
  .index-card .idx-reason {{ font-size: 0.7rem; color: var(--muted); margin-top: 4px; line-height: 1.4; }}

  .section-title {{
    font-size: 1rem; font-weight: 600; margin: 24px 0 12px;
    padding-bottom: 8px; border-bottom: 1px solid var(--border);
    cursor: pointer; user-select: none; display: flex; align-items: center; gap: 8px;
  }}
  .section-title::before {{ content: "\\25B6"; font-size: 0.6rem; transition: transform 0.2s; }}
  .section-title.open::before {{ transform: rotate(90deg); }}
  .section-title.buy {{ color: var(--green); }}
  .section-title.strong-buy {{ color: #00ff88; font-weight: 700; }}
  .section-title.avoid {{ color: var(--red); }}
  .section-title.watch {{ color: var(--yellow); }}
  .section-body {{ display: none; }}
  .section-body.open {{ display: block; }}

  .ticker-card {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; margin-bottom: 10px; overflow: hidden;
  }}
  .ticker-header {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 14px 16px; cursor: pointer; user-select: none;
    transition: background 0.15s;
  }}
  .ticker-header:hover {{ background: rgba(255,255,255,0.03); }}
  .ticker-header .arrow {{ font-size: 0.6rem; color: var(--muted); transition: transform 0.2s; }}
  .ticker-card.open .ticker-header .arrow {{ transform: rotate(90deg); }}
  .ticker-left {{ display: flex; align-items: center; gap: 12px; }}
  .ticker-name {{ font-size: 1rem; font-weight: 700; min-width: 60px; }}
  .ticker-score {{ font-size: 0.85rem; color: var(--muted); }}
  .badge {{
    display: inline-block; padding: 2px 10px; border-radius: 4px;
    font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
  }}
  .badge.buy {{ background: rgba(34,197,94,0.15); color: var(--green); }}
  .badge.avoid {{ background: rgba(239,68,68,0.15); color: var(--red); }}
  .badge.watch {{ background: rgba(234,179,8,0.15); color: var(--yellow); }}
  .badge.index {{ background: rgba(139,143,163,0.12); color: var(--muted); font-size: 0.68rem; padding: 1px 6px; }}

  .ticker-body {{ display: none; padding: 0 16px 16px; }}
  .ticker-card.open .ticker-body {{ display: block; }}

  .details {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 12px; }}
  .detail-group .group-title {{ color: var(--muted); font-size: 0.7rem; text-transform: uppercase; margin-bottom: 6px; text-align: center; }}
  .detail-row {{ display: grid; grid-template-columns: 1fr auto; gap: 8px; font-size: 0.8rem; padding: 2px 0; align-items: center; }}
  .detail-row .label {{ color: var(--muted); text-align: left; }}
  .detail-row .val {{ font-weight: 600; text-align: right; white-space: nowrap; }}
  .detail-row .val.green {{ color: var(--green); }}
  .detail-row .val.red {{ color: var(--red); }}
  .detail-row .val.yellow {{ color: var(--yellow); }}
  .detail-row .val.blue {{ color: var(--blue); }}

  .explanation {{
    background: rgba(59,130,246,0.08); border: 1px solid rgba(59,130,246,0.2);
    border-radius: 6px; padding: 12px 14px; margin-top: 12px;
    font-size: 0.82rem; line-height: 1.55; color: var(--text);
  }}
  .explanation b {{ color: var(--blue); }}
  .explanation::before {{ content: "\\1F4A1 "; }}

  .chart-container {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 6px; padding: 8px; margin-top: 12px;
  }}
  .chart-container .chart-title {{ color: var(--muted); font-size: 0.7rem; text-transform: uppercase; margin-bottom: 4px; text-align: center; }}

  .basis {{ color: var(--muted); font-size: 0.75rem; margin-top: 10px; padding-top: 8px; border-top: 1px solid var(--border); }}
  .exit-strategy {{
    background: rgba(234,179,8,0.06); border: 1px solid rgba(234,179,8,0.15);
    border-radius: 6px; padding: 10px 14px; margin-top: 10px; font-size: 0.8rem;
  }}
  .exit-strategy .group-title {{ color: var(--yellow); }}
  .exit-text {{ margin: 6px 0; line-height: 1.5; }}
  .exit-triggers {{ margin-top: 8px; }}
  .exit-triggers ul {{ margin: 4px 0 0 16px; padding: 0; }}
  .exit-triggers li {{ margin-bottom: 3px; color: var(--muted); font-size: 0.78rem; }}

  .sr-levels {{ font-size: 0.8rem; }}
  .sr-row {{ display: grid; grid-template-columns: 1fr auto auto; gap: 8px; padding: 3px 0; border-bottom: 1px solid rgba(255,255,255,0.04); align-items: center; }}
  .sr-row:last-child {{ border-bottom: none; }}
  .sr-label {{ color: var(--muted); text-align: left; }}
  .sr-price {{ font-weight: 600; text-align: center; }}
  .sr-strength {{ font-size: 0.72rem; padding: 1px 5px; border-radius: 3px; text-align: right; }}
  .sr-strength.strong {{ background: rgba(34,197,94,0.15); color: var(--green); }}
  .sr-strength.medium {{ background: rgba(234,179,8,0.12); color: var(--yellow); }}
  .sr-strength.weak {{ background: rgba(139,143,163,0.1); color: var(--muted); }}
  .no-signals {{ color: var(--muted); font-size: 0.85rem; padding: 16px; }}
  footer {{ margin-top: 32px; color: var(--muted); font-size: 0.7rem; text-align: center; }}
</style>
</head>
<body>
<h1>EGX Daily Signals</h1>
<p class="subtitle">{report_date} — {total_stocks} stocks analyzed</p>
<p class="subtitle" style="font-size:0.75rem;color:#9ca3af;margin-top:-8px;">TradingView data: {ta_timestamp}</p>

<div class="summary">
  <div class="summary-card"><div class="label">Strong Buy</div><div class="value strong-buy">{strong_buy_count}</div></div>
  <div class="summary-card"><div class="label">Buy</div><div class="value buy">{buy_count}</div></div>
  <div class="summary-card"><div class="label">Watch</div><div class="value watch">{watch_count}</div></div>
  <div class="summary-card"><div class="label">Avoid</div><div class="value avoid">{avoid_count}</div></div>
</div>

<div class="market-overview">
  <div class="market-title">Market Overview</div>
  <div class="market-indices">
    {index_cards}
  </div>
</div>

{strong_buy_section}
{buy_section}
{watch_section}
{avoid_section}

<footer>Generated by EGX Signal Generator — {report_date} - A.A.Raouf</footer>

<script>
// Toggle sections
document.querySelectorAll('.section-title').forEach(el => {{
  el.addEventListener('click', () => {{
    el.classList.toggle('open');
    el.nextElementSibling.classList.toggle('open');
  }});
}});
// Open buy sections by default
document.querySelectorAll('.section-title.strong-buy, .section-title.buy').forEach(el => {{
  el.classList.add('open'); el.nextElementSibling.classList.add('open');
}});

// Toggle ticker cards
document.querySelectorAll('.ticker-header').forEach(el => {{
  el.addEventListener('click', () => {{
    el.closest('.ticker-card').classList.toggle('open');
  }});
}});
</script>
</body>
</html>"""

TICKER_CARD_TEMPLATE = """
<div class="ticker-card" id="card-{ticker}">
  <div class="ticker-header">
    <div class="ticker-left">
      <span class="ticker-name">{ticker}</span>
      <span class="badge index">{index_badge}</span>
      <span class="ticker-score">{score}</span>
      <span class="badge {rec_class}">{recommendation}</span>
    </div>
    <span class="arrow">&#9654;</span>
  </div>
  <div class="ticker-body">
    <div class="details">
      <div class="detail-group">
        <div class="group-title">Technical</div>
        <div class="detail-row"><span class="label">Price</span><span class="val">{price}</span></div>
        <div class="detail-row"><span class="label">RSI</span><span class="val {rsi_class}">{rsi}</span></div>
        <div class="detail-row"><span class="label">MACD</span><span class="val {macd_class}">{macd}</span></div>
        <div class="detail-row"><span class="label">Regime</span><span class="val">{regime}</span></div>
      </div>
      <div class="detail-group">
        <div class="group-title">Moving Averages</div>
        <div class="detail-row"><span class="label">EMA 20</span><span class="val {ema20_class}">{ema20}</span></div>
        <div class="detail-row"><span class="label">EMA 50</span><span class="val {ema50_class}">{ema50}</span></div>
        <div class="detail-row"><span class="label">EMA 200</span><span class="val {ema200_class}">{ema200}</span></div>
        <div class="detail-row"><span class="label">SMA 20</span><span class="val">{sma20}</span></div>
        <div class="detail-row"><span class="label">SMA 50</span><span class="val">{sma50}</span></div>
        <div class="detail-row"><span class="label">SMA 200</span><span class="val">{sma200}</span></div>
      </div>
      <div class="detail-group">
        <div class="group-title">VWAP & Volume Profile</div>
        <div class="detail-row"><span class="label">VWAP</span><span class="val">{vwap}</span></div>
        <div class="detail-row"><span class="label">Dist VWAP</span><span class="val {dist_vwap_class}">{dist_vwap}</span></div>
        <div class="detail-row"><span class="label">VP POC</span><span class="val">{vp_poc}</span></div>
        <div class="detail-row"><span class="label">Above POC</span><span class="val {above_poc_class}">{above_poc}</span></div>
      </div>
      <div class="detail-group">
        <div class="group-title">ML Forecast (10-day)</div>
        <div class="detail-row"><span class="label">Low</span><span class="val red">{ml_low}</span></div>
        <div class="detail-row"><span class="label">Medium</span><span class="val {ml_med_class}">{ml_medium}</span></div>
        <div class="detail-row"><span class="label">High</span><span class="val green">{ml_high}</span></div>
        <div class="detail-row"><span class="label">Conviction</span><span class="val {ml_conv_class}">{ml_conviction}</span></div>
        <div class="detail-row"><span class="label">Price Source</span><span class="val">{ml_price_source}</span></div>
      </div>
      <div class="detail-group">
        <div class="group-title">Trade Plan</div>
        <div class="detail-row"><span class="label">Entry</span><span class="val blue">{entry}</span></div>
        <div class="detail-row"><span class="label">Max Entry</span><span class="val blue">{entry_high}</span></div>
        <div class="detail-row"><span class="label">Entry Source</span><span class="val">{entry_source}</span></div>
        <div class="detail-row"><span class="label">Action</span><span class="val {entry_action_class}">{entry_action}</span></div>
        <div class="detail-row"><span class="label">Stop Loss</span><span class="val red">{stop}</span></div>
        <div class="detail-row"><span class="label">Stop Basis</span><span class="val">{stop_basis}</span></div>
        <div class="detail-row"><span class="label">TP1</span><span class="val green">{tp1} ({tp1_pct})</span></div>
        <div class="detail-row"><span class="label">TP1 R/R</span><span class="val">{tp1_rr}</span></div>
        <div class="detail-row"><span class="label">TP2</span><span class="val green">{tp2} ({tp2_pct})</span></div>
        <div class="detail-row"><span class="label">TP2 R/R</span><span class="val">{tp2_rr}</span></div>
        <div class="detail-row"><span class="label">TP3</span><span class="val green">{tp3} ({tp3_pct})</span></div>
        <div class="detail-row"><span class="label">TP3 R/R</span><span class="val">{tp3_rr}</span></div>
        <div class="detail-row"><span class="label">Duration</span><span class="val yellow">{hold_label}</span></div>
      </div>
      <div class="detail-group">
        <div class="group-title">Support & Resistance</div>
        {support_resistance_html}
      </div>
      <div class="detail-group">
        <div class="group-title">ChartScan AI</div>
        <div class="detail-row"><span class="label">Signal</span><span class="val {cs_class}">{cs_signal}</span></div>
        <div class="detail-row"><span class="label">Confidence</span><span class="val">{cs_confidence}</span></div>
        <div class="detail-row"><span class="label">Candle</span><span class="val">{candle_signal}</span></div>
      </div>
      <div class="detail-group">
        <div class="group-title">Fundamentals</div>
        <div class="detail-row"><span class="label">P/E (TTM)</span><span class="val {pe_class}">{pe}</span></div>
        <div class="detail-row"><span class="label">Sector Avg P/E</span><span class="val">{sector_pe}</span></div>
        <div class="detail-row"><span class="label">vs Sector</span><span class="val {pe_vs_sector_class}">{pe_vs_sector}</span></div>
        <div class="detail-row"><span class="label">P/B</span><span class="val">{pb}</span></div>
        <div class="detail-row"><span class="label">Market Cap</span><span class="val">{market_cap}</span></div>
        <div class="detail-row"><span class="label">Dividend</span><span class="val">{dividend}</span></div>
        <div class="detail-row"><span class="label">Fair Value</span><span class="val {fv_class}">{fv_price} ({fv_method})</span></div>
        <div class="detail-row"><span class="label">vs Fair Value</span><span class="val {fv_class}">{fv_vs}</span></div>
      </div>
    </div>
    <div class="explanation">{explanation}</div>
    {chart_html}
    {exit_strategy_html}
    <div class="basis">Score: {score_breakdown}</div>
  </div>
</div>"""


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _fmt(val, decimals=2, suffix=""):
    if val is None:
        return "N/A"
    if isinstance(val, bool):
        return "Yes" if val else "No"
    try:
        return f"{float(val):,.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return str(val)

def _color_class(val, threshold_high=None, threshold_low=None, invert=False):
    if val is None:
        return ""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    if invert:
        if threshold_high and v >= threshold_high:
            return "red"
        if threshold_low and v <= threshold_low:
            return "green"
    else:
        if threshold_high and v >= threshold_high:
            return "green"
        if threshold_low and v <= threshold_low:
            return "red"
    return ""

def _fmt_signed_pct(val):
    """Like _fmt but with an explicit +/- sign; safely handles None
    (the previous inline f-string crashed with TypeError when the value
    was explicitly None rather than a missing key)."""
    if val is None:
        return "N/A"
    try:
        return f"{float(val):+.1f}%"
    except (TypeError, ValueError):
        return "N/A"

def _price_vs_ma_class(close, ma_val):
    """Green if price is above the moving average, red if below, blank if
    either value is missing (previously a missing MA silently defaulted to
    0, which made 'close > 0' true and wrongly colored it green)."""
    if close is None or ma_val is None:
        return ""
    try:
        return "green" if float(close) > float(ma_val) else "red"
    except (TypeError, ValueError):
        return ""

def _rec_class(rec):
    rec = rec.lower() if rec in ("Strong Buy", "Buy", "Avoid", "Watch") else "watch"
    if rec == "strong buy": return "strong-buy"
    return rec

def _build_score_breakdown(row):
    parts = []
    key_map = {
        "trend": "Score - Trend", "macd": "Score - MACD", "rsi": "Score - RSI",
        "volume": "Score - Volume", "adi": "Score - ADI", "support": "Score - Support",
        "vwap": "Score - VWAP", "volume_profile": "Score - Volume Profile",
    }
    for cat, key in key_map.items():
        val = row.get(key)
        if val is not None:
            label = cat.replace("_", " ").title()
            parts.append(f"{label}: {val:.1f}")
    return " | ".join(parts) if parts else f"Total: {row.get('Score', row.get('score', 0)):.1f}"

def _build_exit_strategy_html(row):
    exit_strat = row.get("hold_exit_strategy", "")
    exit_triggers = row.get("hold_exit_triggers", [])
    if not exit_strat and not exit_triggers:
        return ""
    html = '<div class="exit-strategy">'
    html += '<div class="group-title">Exit Strategy</div>'
    if exit_strat:
        html += f'<div class="exit-text">{exit_strat}</div>'
    if exit_triggers:
        html += '<div class="exit-triggers"><b>Exit Triggers:</b><ul>'
        for t in exit_triggers[:5]:
            html += f'<li>{t}</li>'
        html += '</ul></div>'
    html += '</div>'
    return html

def _build_sr_html(row, close_price):
    support = row.get("support_levels", [])
    resistance = row.get("resistance_levels", [])
    if not support and not resistance:
        return '<div class="sr-levels"><div class="sr-row"><span class="sr-label">No S/R data available</span></div></div>'

    html = '<div class="sr-levels">'

    # Resistance levels (highest first)
    if resistance:
        resistance_sorted = sorted(resistance, key=lambda x: -x["price"])
        for r in resistance_sorted[:3]:
            dist = ((r["price"] - close_price) / close_price * 100) if close_price else 0
            s_cls = r["strength"].lower()
            html += f'<div class="sr-row"><span class="sr-label">R ({r["strength"]})</span><span class="sr-price">{r["price"]:.2f}</span><span class="sr-strength {s_cls}">{dist:+.1f}%</span></div>'

    # Current price marker
    html += f'<div class="sr-row" style="border-top:1px dashed rgba(255,255,255,0.1);padding-top:4px;"><span class="sr-label" style="color:var(--blue);font-weight:600;">Current</span><span class="sr-price" style="color:var(--blue);">{close_price:.2f}</span></div>'

    # Support levels (highest first — closest to price)
    if support:
        support_sorted = sorted(support, key=lambda x: -x["price"])
        for s in support_sorted[:3]:
            dist = ((s["price"] - close_price) / close_price * 100) if close_price else 0
            s_cls = s["strength"].lower()
            html += f'<div class="sr-row"><span class="sr-label">S ({s["strength"]})</span><span class="sr-price">{s["price"]:.2f}</span><span class="sr-strength {s_cls}">{dist:+.1f}%</span></div>'

    html += '</div>'
    return html

def _build_chart_html(row, fund_df=None):
    """Build an interactive Plotly chart for Strong Buy/Buy tickers.

    The trade plan (entry / stop / take-profits) is drawn as a stacked
    risk-reward box — like TradingView's "Long Position" tool — in a
    dedicated lane to the right of the price history, with price labels
    on the right edge instead of a legend full of crossing horizontal
    lines. The ML forecast gets its own shaded cone in a lane before it.
    Reference lines (EMAs, nearest support/resistance) are still plotted
    but hidden by default (toggle from the legend) so they don't compete
    with the trade plan for attention.
    """
    rec = row.get("Recommendation", row.get("recommendation", "Watch"))
    if rec not in ("Strong Buy", "Buy"):
        return ""

    ticker = row.get("Selected Stock", row.get("ticker", "?"))
    close = row.get("Current EGP Price", row.get("close"))
    dates = row.get("chart_dates")
    if not close or not dates:
        return ""

    import json

    chart_data = {
        "dates": dates,
        "high": row.get("chart_high", []),
        "low": row.get("chart_low", []),
        "close": row.get("chart_close", []),
    }

    entry = row.get("Optimal Entry Price", row.get("entry_price"))
    stop = row.get("Stop Loss", row.get("stop_loss"))
    tp1 = row.get("Take Profit 1", row.get("tp1"))
    tp2 = row.get("Take Profit 2", row.get("tp2"))
    tp3 = row.get("Take Profit 3", row.get("tp3"))
    tp1_rr = row.get("TP1 Risk/Reward", row.get("tp1_rr"))
    ml_low = row.get("ML Low Price", row.get("ml_low_price"))
    ml_med = row.get("ML Medium Price", row.get("ml_medium_price"))
    ml_high = row.get("ML High Price", row.get("ml_high_price"))
    ema20 = row.get("20 EMA", row.get("ema_20"))
    ema50 = row.get("50 EMA", row.get("ema_50"))

    support = row.get("support_levels", [])
    resistance = row.get("resistance_levels", [])
    nearest_support = None
    strongest_resistance = None
    if support:
        support_below = [s for s in support if s["price"] < close]
        if support_below:
            nearest_support = max(support_below, key=lambda x: x["price"])
    if resistance:
        resistance_above = [r for r in resistance if r["price"] > close]
        if resistance_above:
            strongest_resistance = max(resistance_above, key=lambda x: x.get("touches", 0))

    # Fallback: if no resistance found, use the recent high from chart data
    # Use the second-highest high (excluding the latest candle which is
    # the real-time TradingView session — that's the current price, not resistance)
    if not strongest_resistance and chart_data.get("high") and len(chart_data["high"]) > 1:
        all_highs = chart_data["high"][:-1]
        recent_high = max(all_highs)
        chart_close_last = chart_data["close"][-1] if chart_data.get("close") else 0
        if recent_high and recent_high < chart_close_last:
            strongest_resistance = {"price": round(recent_high, 3), "strength": "Recent High", "touches": 0}

    def _pct(base, val):
        if base and val and base > 0:
            return f"{(val - base) / base * 100:+.1f}%"
        return None

    # Pre-formatted labels for the right-edge price ladder — computed once
    # in Python so the JS stays dumb (no duplicated % / R:R math).
    labels = {
        "entry": f"Entry {_fmt(entry)}" if entry else None,
        "stop": f"Stop {_fmt(stop)} ({_pct(entry, stop) or 'N/A'})" if stop else None,
        "fcst_high": f"Fcst High {_fmt(ml_high)} ({_pct(close, ml_high) or 'N/A'})" if ml_high else None,
        "fcst_med": f"Fcst Med {_fmt(ml_med)} ({_pct(close, ml_med) or 'N/A'})" if ml_med else None,
        "fcst_low": f"Fcst Low {_fmt(ml_low)} ({_pct(close, ml_low) or 'N/A'})" if ml_low else None,
    }
    if entry and tp1_rr:
        labels["rr"] = f"R:R 1:{float(tp1_rr):.1f}"
    elif entry and stop and tp1 and entry > stop:
        risk = entry - stop
        reward = tp1 - entry
        labels["rr"] = f"R:R 1:{reward / risk:.1f}" if risk > 0 else None
    else:
        labels["rr"] = None

    # Reward zone split into up to 3 stacked bands (entry→TP1→TP2→TP3),
    # each a bit darker green than the last, each with its own label.
    tier_colors = [
        ("rgba(34,197,94,0.12)", "#4ade80"),
        ("rgba(34,197,94,0.20)", "#22c55e"),
        ("rgba(34,197,94,0.30)", "#16a34a"),
    ]
    reward_tiers = []
    prev = entry
    for i, tp in enumerate((tp1, tp2, tp3)):
        if tp and prev and tp > prev:
            fill, text_color = tier_colors[min(i, len(tier_colors) - 1)]
            reward_tiers.append({
                "price": tp,
                "fill": fill,
                "text_color": text_color,
                "label": f"TP{i+1} {_fmt(tp)} ({_pct(entry, tp) or 'N/A'})",
            })
            prev = tp

    levels = {
        "entry": entry, "stop": stop,
        "tp1": tp1, "tp2": tp2, "tp3": tp3,
        "ml_low": ml_low, "ml_med": ml_med, "ml_high": ml_high,
        "ema20": ema20, "ema50": ema50,
        "support": nearest_support["price"] if nearest_support else None,
        "resistance": strongest_resistance["price"] if strongest_resistance else None,
        "labels": labels,
        "reward_tiers": reward_tiers,
    }

    chart_json = json.dumps(chart_data)
    levels_json = json.dumps(levels)
    chart_id = f"chart-{ticker}"

    html = f"""
    <div class="chart-container">
      <div class="chart-title">Price Chart — {ticker}
        <button onclick="toggleFS('{chart_id}')" style="float:right;background:var(--border);color:var(--text);border:none;padding:2px 8px;border-radius:4px;cursor:pointer;font-size:0.7rem;">Fullscreen</button>
      </div>
      <div id="{chart_id}" style="width:100%;height:450px;"></div>
    </div>
    <script>
    function toggleFS(id) {{
      var el = document.getElementById(id);
      if (!document.fullscreenElement) {{
        el.requestFullscreen().catch(e=>{{}});
        setTimeout(function() {{ Plotly.Plots.resize(el); }}, 300);
      }} else {{
        document.exitFullscreen();
        setTimeout(function() {{ Plotly.Plots.resize(el); }}, 300);
      }}
    }}
    (function() {{
      var d = {chart_json};
      var lv = {levels_json};

      if (!d.dates || d.dates.length === 0) {{
        document.getElementById('{chart_id}').innerHTML =
          '<div style="color:var(--muted);font-size:0.8rem;padding:20px;text-align:center;">No chart data available</div>';
        return;
      }}

      var traces = [];
      var shapes = [];
      var annotations = [];
      var lastDate = d.dates[d.dates.length - 1];

      // Price line
      traces.push({{
        x: d.dates, y: d.close,
        type: 'scatter', mode: 'lines', name: 'Price',
        line: {{color: '#e1e4ea', width: 2}},
        showlegend: true
      }});

      // Horizontal line with text label (centered)
      var placedLabels = [];
      function hLine(val, name, color, dash, width, labelSide) {{
        if (!val) return;
        traces.push({{
          x: [d.dates[0], lastDate], y: [val, val],
          mode: 'lines', name: name,
          line: {{color: color, width: width || 1.5, dash: dash || 'solid'}},
          showlegend: true, hoverinfo: 'skip'
        }});
        // Text label position: 'left' = first date, 'mid' = middle, 'right' = last date
        var labelX = labelSide === 'left' ? d.dates[0] : (labelSide === 'right' ? lastDate : d.dates[Math.floor(d.dates.length / 2)]);
        var xAnchor = labelSide === 'left' ? 'left' : (labelSide === 'right' ? 'right' : 'center');
        var xShift = labelSide === 'left' ? 4 : (labelSide === 'right' ? -4 : 0);
        // Detect overlapping labels (within 0.5%) and stack them vertically
        var yShift = 4;
        for (var j = 0; j < placedLabels.length; j++) {{
          if (Math.abs(placedLabels[j] - val) / val < 0.005) {{
            yShift += 16;
          }}
        }}
        placedLabels.push(val);
        annotations.push({{
          x: labelX, y: val, xref: 'x', yref: 'y',
          text: '<b>' + name + '</b> ' + val.toFixed(2),
          showarrow: false, yanchor: 'bottom', yshift: yShift,
          xanchor: xAnchor, xshift: xShift,
          font: {{size: 10, color: color, family: 'monospace'}},
          bgcolor: 'rgba(26,29,39,0.85)', borderpad: 2
        }});
      }}

      // S/R and Entry — text in middle
      hLine(lv.support, 'Support', '#7dd3fc', 'dash', 1.5, 'mid');    // light blue
      hLine(lv.resistance, 'Resistance', '#a16207', 'dot', 1.5, 'mid'); // brown
      hLine(lv.entry, 'Entry', '#3b82f6', 'solid', 2, 'mid');          // blue

      // Take Profit zones (last 5 trading days = ~1 week) + stop loss red zone
      var zoneStart = d.dates.length > 5 ? d.dates[d.dates.length - 6] : d.dates[0];

      // Stop Loss zone (red, from stop to entry, last week)
      if (lv.stop && lv.entry) {{
        shapes.push({{
          type: 'rect', xref: 'x', x0: zoneStart, x1: lastDate, yref: 'y',
          y0: lv.stop, y1: lv.entry,
          fillcolor: 'rgba(239,68,68,0.15)', line: {{width: 0}}
        }});
        hLine(lv.stop, 'Stop Loss', '#ef4444', 'dashdot', 1.5, 'right');
      }}

      // TP zones (green, stacked from entry upward, last week)
      var prevPrice = lv.entry;
      var tpColors = ['rgba(34,197,94,0.10)', 'rgba(34,197,94,0.18)', 'rgba(34,197,94,0.26)'];
      var tpLineColors = ['#4ade80', '#22c55e', '#16a34a'];
      [lv.tp1, lv.tp2, lv.tp3].forEach(function(tp, i) {{
        if (tp && prevPrice && tp > prevPrice) {{
          shapes.push({{
            type: 'rect', xref: 'x', x0: zoneStart, x1: lastDate, yref: 'y',
            y0: prevPrice, y1: tp,
            fillcolor: tpColors[i], line: {{width: 0}}
          }});
          hLine(tp, 'TP' + (i + 1), tpLineColors[i], 'shortdot', 1.2, 'right');
          prevPrice = tp;
        }}
      }});

      // Forecast lines — text on left side
      // Offset forecast lines if they overlap with TP lines (within 0.5%)
      var tpPrices = [lv.tp1, lv.tp2, lv.tp3].filter(function(p) {{ return p; }});
      function offsetIfOverlap(val, offset) {{
        if (!val) return val;
        for (var i = 0; i < tpPrices.length; i++) {{
          if (Math.abs(tpPrices[i] - val) / val < 0.005) {{
            return val * (1 + offset);
          }}
        }}
        return val;
      }}
      hLine(offsetIfOverlap(lv.ml_high, 0.003), 'Fcst High', '#a855f7', 'dot', 1.5, 'left');  // purple, thicker when overlapping
      hLine(lv.ml_med, 'Fcst Med', '#f97316', 'dash', 1.5, 'left');     // orange
      hLine(offsetIfOverlap(lv.ml_low, -0.003), 'Fcst Low', '#9ca3af', 'dot', 1.5, 'left');   // grey

      // Y range
      var allP = d.close.slice();
      [lv.entry, lv.stop, lv.tp1, lv.tp2, lv.tp3, lv.support, lv.resistance,
       lv.ml_low, lv.ml_med, lv.ml_high].forEach(function(v) {{ if (v) allP.push(v); }});
      if (allP.length === 0) allP = [0, 1];
      var yMin = Math.min.apply(null, allP) * 0.95;
      var yMax = Math.max.apply(null, allP) * 1.05;

      var layout = {{
        paper_bgcolor: '#1a1d27',
        plot_bgcolor: '#1a1d27',
        font: {{color: '#e1e4ea', size: 11}},
        margin: {{l: 55, r: 55, t: 10, b: 50}},
        xaxis: {{
          gridcolor: 'rgba(255,255,255,0.05)',
          zeroline: false,
          type: 'category',
          tickangle: -45,
          tickfont: {{size: 9}},
          nticks: 15,
          automargin: true,
        }},
        yaxis: {{
          title: 'Price (EGP)',
          gridcolor: 'rgba(255,255,255,0.05)',
          zeroline: false,
          side: 'right',
          range: [yMin, yMax],
          automargin: true,
        }},
        shapes: shapes,
        legend: {{
          orientation: 'h', y: 1.15, x: 0.5, xanchor: 'center',
          bgcolor: 'rgba(0,0,0,0)', font: {{size: 9}},
          tracegroupgap: 5,
        }},
        annotations: annotations,
        dragmode: 'zoom',
        hovermode: 'x unified',
      }};

      Plotly.newPlot('{chart_id}', traces, layout, {{
        responsive: true,
        displayModeBar: true,
        displaylogo: false
      }});
    }})();
    </script>
    """
    return html


def _build_ticker_card(row, fund_df=None):
    # Determine ML recommendation
    ml_conv = row.get("ML Confidence", row.get("ml_confidence", 0)) or 0
    ml_signal = row.get("ML Signal", row.get("ml_signal", "N/A"))
    ml_rec_class = "green" if ml_signal == "Buy" else "red" if ml_signal == "Avoid" else "yellow"

    # ML forecast
    ml_med = row.get("ML Medium Price", row.get("ml_medium_price"))
    close = row.get("Current EGP Price", row.get("close"))
    if ml_med and close and close > 0:
        ml_med_pct = f"{(ml_med - close) / close * 100:+.1f}%"
    else:
        ml_med_pct = "N/A"

    # Entry
    entry = row.get("Optimal Entry Price", row.get("entry_price"))
    entry_source = row.get("Entry Source", row.get("entry_source", "N/A"))
    entry_action = row.get("Entry Action", row.get("entry_action", "N/A"))
    entry_high = row.get("Entry High", row.get("entry_high"))
    stop = row.get("Stop Loss", row.get("stop_loss"))
    stop_basis = row.get("Stop Loss Basis", row.get("stop_basis", "N/A"))
    tp1 = row.get("Take Profit 1", row.get("tp1"))
    tp2 = row.get("Take Profit 2", row.get("tp2"))
    tp3 = row.get("Take Profit 3", row.get("tp3"))
    tp1_rr = row.get("TP1 Risk/Reward", row.get("tp1_rr"))
    tp2_rr = row.get("TP2 Risk/Reward", row.get("tp2_rr"))
    tp3_rr = row.get("TP3 Risk/Reward", row.get("tp3_rr"))

    # TP percentages
    def _pct(tp):
        if tp and entry and entry > 0:
            return f"{(tp - entry) / entry * 100:+.1f}%"
        return "N/A"
    tp1_pct = _pct(tp1)
    tp2_pct = _pct(tp2)
    tp3_pct = _pct(tp3)

    # ChartScan
    cs_signal = row.get("ChartScanAI Signal", row.get("chartscan_signal", "N/A"))
    cs_conf = row.get("ChartScanAI Confidence", row.get("chartscan_confidence"))

    # Fundamentals
    ticker = row.get("Selected Stock", row.get("ticker", "?"))
    pe = pe_sector = pe_vs = pb = mcap = div = None
    pe_class = ""
    pe_vs_sector_class = ""
    if fund_df is not None and not fund_df.empty:
        frow = fund_df[fund_df["ticker"] == ticker]
        if not frow.empty:
            frow = frow.iloc[0]
            pe = frow.get("trailingPE")
            pe_sector = frow.get("sector_avg_trailingPE")
            pe_vs = frow.get("pe_vs_sector")
            pb = frow.get("priceToBook")
            mcap = frow.get("marketCap")
            div = frow.get("trailingAnnualDividendYield")
            pe_class = _color_class(pe, threshold_high=25, threshold_low=10)
            pe_vs_sector_class = "green" if pe_vs is not None and pe_vs < -10 else "red" if pe_vs is not None and pe_vs > 10 else ""

    # Index membership
    index_membership = row.get("index_membership", "UNINDEX")
    index_sentiment = row.get("index_sentiment", "Unknown")
    idx_cls = index_sentiment.lower() if index_sentiment else "neutral"
    index_badge = f"{index_membership} {index_sentiment}"

    def _fmt_mcap(val):
        if val is None: return "N/A"
        if val >= 1e12: return f"{val/1e12:.1f}T"
        if val >= 1e9: return f"{val/1e9:.1f}B"
        if val >= 1e6: return f"{val/1e6:.1f}M"
        return f"{val:,.0f}"

    def _fmt_pct(val):
        if val is None: return "N/A"
        return f"{float(val)*100:.1f}%"

    # Fair value from row
    fv_price = row.get("Implied Fair Value (EGP)", row.get("fair_value_egp"))
    fv_method = row.get("Fair Value Method", row.get("fair_value_method", ""))
    if fv_price and close and close > 0:
        fv_vs_pct = (fv_price - close) / close * 100
        fv_vs = f"{fv_vs_pct:+.1f}%"
        fv_cls = "green" if fv_vs_pct > 0 else "red"
    else:
        fv_vs = "N/A"
        fv_cls = ""

    # Chart for Buy/Strong Buy
    chart_html = _build_chart_html(row, fund_df)

    return TICKER_CARD_TEMPLATE.format(
        ticker=ticker,
        index_badge=index_badge,
        recommendation=row.get("Recommendation", row.get("recommendation", "?")),
        rec_class=_rec_class(row.get("Recommendation", row.get("recommendation", "Watch"))),
        score=f"{row.get('Score', row.get('score', 0)):.1f}",

        price=_fmt(close),
        rsi=_fmt(row.get("RSI (%)", row.get("rsi")), 1),
        rsi_class=_color_class(row.get("RSI (%)", row.get("rsi")), threshold_high=70, threshold_low=30),
        macd="Bullish" if row.get("MACD Bullish (Yes/No)", row.get("macd_bullish")) in ("Yes", True) else "Bearish",
        macd_class="green" if row.get("MACD Bullish (Yes/No)", row.get("macd_bullish")) in ("Yes", True) else "red",
        regime=row.get("regime", "N/A"),

        ema20=_fmt(row.get("20 EMA", row.get("ema_20"))),
        ema20_class=_price_vs_ma_class(close, row.get("20 EMA", row.get("ema_20"))),
        ema50=_fmt(row.get("50 EMA", row.get("ema_50"))),
        ema50_class=_price_vs_ma_class(close, row.get("50 EMA", row.get("ema_50"))),
        ema200=_fmt(row.get("200 EMA", row.get("ema_200"))),
        ema200_class=_price_vs_ma_class(close, row.get("200 EMA", row.get("ema_200"))),
        sma20=_fmt(row.get("20 SMA", row.get("SMA_20", row.get("sma_20")))),
        sma50=_fmt(row.get("50 SMA", row.get("sma_50"))),
        sma200=_fmt(row.get("200 SMA", row.get("sma_200"))),

        vwap=_fmt(row.get("VWAP", row.get("vwap"))),
        dist_vwap=_fmt_signed_pct(row.get("Dist VWAP %", row.get("dist_vwap_pct"))),
        dist_vwap_class=_color_class(row.get("Dist VWAP %", row.get("dist_vwap_pct")), threshold_high=5, threshold_low=-5, invert=True),
        vp_poc=_fmt(row.get("Volume Profile POC", row.get("vp_poc"))),
        above_poc=_fmt(row.get("Above POC", row.get("above_poc"))),
        above_poc_class="green" if row.get("Above POC", row.get("above_poc")) in (True, "True", "Yes") else "red",

        ml_low=f"{(row.get('ml_medium_price', 0) or 0) * 0.92:.2f}" if row.get("ml_medium_price") else "N/A",
        ml_medium=f"{row.get('ml_medium_price', 0) or 0:.2f}" if row.get("ml_medium_price") else "N/A",
        ml_med_class=ml_rec_class,
        ml_high=f"{(row.get('ml_medium_price', 0) or 0) * 1.05:.2f}" if row.get("ml_medium_price") else "N/A",
        ml_conviction=f"{ml_conv * 100:.0f}/100 {ml_signal}",
        ml_conv_class=ml_rec_class,
        ml_price_source=row.get("ml_last_price_source", "yfinance"),

        entry=_fmt(entry),
        entry_high=_fmt(entry_high),
        entry_source=entry_source,
        entry_action=entry_action,
        entry_action_class="green" if "BUY" in str(entry_action).upper() else ("red" if "AVOID" in str(entry_action).upper() else "yellow"),
        stop=_fmt(stop),
        stop_basis=stop_basis,
        tp1=_fmt(tp1),
        tp1_pct=tp1_pct,
        tp1_rr=_fmt(tp1_rr, 1, "x") if tp1_rr else "N/A",
        tp2=_fmt(tp2),
        tp2_pct=tp2_pct,
        tp2_rr=_fmt(tp2_rr, 1, "x") if tp2_rr else "N/A",
        tp3=_fmt(tp3),
        tp3_pct=tp3_pct,
        tp3_rr=_fmt(tp3_rr, 1, "x") if tp3_rr else "N/A",
        hold_label=row.get("hold_label", "N/A"),

        cs_signal=cs_signal if cs_signal and cs_signal != "N/A" else "No pattern",
        cs_class="green" if cs_signal == "Buy" else "red" if cs_signal == "Sell" else "",
        cs_confidence=f"{cs_conf * 100:.0f}%" if cs_conf else "N/A",
        candle_signal=row.get("candle_signal", "N/A"),

        pe=_fmt(pe, 1) if pe else "N/A",
        pe_class=pe_class,
        sector_pe=_fmt(pe_sector, 1) if pe_sector else "N/A",
        pe_vs_sector=f"{pe_vs:+.1f}%" if pe_vs is not None else "N/A",
        pe_vs_sector_class=pe_vs_sector_class,
        pb=_fmt(pb, 2) if pb else "N/A",
        market_cap=_fmt_mcap(mcap),
        dividend=_fmt_pct(div),
        fv_price=_fmt(fv_price) if fv_price else "N/A",
        fv_method=fv_method or "",
        fv_class=fv_cls,
        fv_vs=fv_vs,

        explanation=_generate_explanation(row),
        chart_html=chart_html,
        exit_strategy_html=_build_exit_strategy_html(row),
        support_resistance_html=_build_sr_html(row, close or 0),
        score_breakdown=_build_score_breakdown(row),
    )


# ─────────────────────────────────────────────────────────────
# Main Generator
# ─────────────────────────────────────────────────────────────
def _get_ta_timestamp(rows):
    """Extract the most recent TradingView data timestamp from rows."""
    timestamps = []
    for r in rows:
        ts = r.get("ta_fetch_time") or r.get("TA Data As Of")
        if ts:
            timestamps.append(ts)
    if timestamps:
        return max(timestamps)
    return "N/A"

# ─────────────────────────────────────────────────────────────
def generate_html_report(rows: List[Dict], output_path: str) -> str:
    """Generate styled HTML report. Returns the output path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load fundamentals cache
    from src.data.fundamentals import load_fundamentals
    fund_df = load_fundamentals()

    buy_rows = [r for r in rows if r.get("Recommendation", r.get("recommendation")) == "Buy"]
    strong_buy_rows = [r for r in rows if r.get("Recommendation", r.get("recommendation")) == "Strong Buy"]
    watch_rows = [r for r in rows if r.get("Recommendation", r.get("recommendation")) == "Watch"]
    avoid_rows = [r for r in rows if r.get("Recommendation", r.get("recommendation")) == "Avoid"]

    # Any row with an unrecognized/missing recommendation used to silently
    # vanish from the report (not counted in any section) while still being
    # counted in total_stocks. Fall back to Watch so nothing is dropped.
    _known = {id(r) for r in buy_rows + strong_buy_rows + watch_rows + avoid_rows}
    unclassified = [r for r in rows if id(r) not in _known]
    if unclassified:
        log.warning(
            "%d row(s) had an unrecognized Recommendation value and were filed under Watch: %s",
            len(unclassified),
            [r.get("Selected Stock", r.get("ticker", "?")) for r in unclassified],
        )
        watch_rows = watch_rows + unclassified

    # Build market overview index cards
    index_cards_html = ""
    seen_indices = set()
    for r in rows:
        mem = r.get("index_membership", "UNINDEX")
        if mem in seen_indices:
            continue
        seen_indices.add(mem)
        # Find a row with this index to get sentiment data
        idx_row = next((x for x in rows if x.get("index_membership") == mem), None)
        if idx_row:
            sentiment = idx_row.get("index_sentiment", "Unknown")
            close = idx_row.get("index_close")
            rsi = idx_row.get("index_rsi")
            adx = idx_row.get("index_adx")
            reasons = idx_row.get("index_reasons", [])
            s_cls = sentiment.lower() if sentiment else "neutral"
            idx_label = "EGX100 (Market)" if mem == "UNINDEX" else mem
            close_str = f"{close:,.1f}" if close else "N/A"
            rsi_str = f"{rsi:.0f}" if rsi else "N/A"
            adx_str = f"{adx:.0f}" if adx else "N/A"
            reasons_str = "; ".join(reasons[:3]) if reasons else "No data"
            index_cards_html += f"""
            <div class="index-card">
              <div class="idx-name">{idx_label}</div>
              <div class="idx-sentiment {s_cls}">{sentiment}</div>
              <div class="idx-detail">{close_str} | RSI {rsi_str} | ADX {adx_str}</div>
              <div class="idx-reason">{reasons_str}</div>
            </div>"""

    strong_buy_section = f'<div class="section-title strong-buy">STRONG BUY ({len(strong_buy_rows)})</div>\n<div class="section-body">'
    if strong_buy_rows:
        for r in sorted(strong_buy_rows, key=lambda x: -(x.get("Score", x.get("score", 0)) or 0)):
            strong_buy_section += _build_ticker_card(r, fund_df)
    else:
        strong_buy_section += '<div class="no-signals">No strong buy signals today</div>'
    strong_buy_section += '</div>'

    buy_section = f'<div class="section-title buy">BUY ({len(buy_rows)})</div>\n<div class="section-body">'
    if buy_rows:
        for r in sorted(buy_rows, key=lambda x: -(x.get("Score", x.get("score", 0)) or 0)):
            buy_section += _build_ticker_card(r, fund_df)
    else:
        buy_section += '<div class="no-signals">No buy signals today</div>'
    buy_section += '</div>'

    watch_section = f'<div class="section-title watch">WATCH ({len(watch_rows)})</div>\n<div class="section-body">'
    if watch_rows:
        for r in sorted(watch_rows, key=lambda x: -(x.get("Score", x.get("score", 0)) or 0)):
            watch_section += _build_ticker_card(r, fund_df)
    else:
        watch_section += '<div class="no-signals">No watch signals</div>'
    watch_section += '</div>'

    avoid_section = f'<div class="section-title avoid">AVOID ({len(avoid_rows)})</div>\n<div class="section-body">'
    if avoid_rows:
        for r in sorted(avoid_rows, key=lambda x: -(x.get("Score", x.get("score", 0)) or 0)):
            avoid_section += _build_ticker_card(r, fund_df)
    else:
        avoid_section += '<div class="no-signals">No avoid signals</div>'
    avoid_section += '</div>'

    html = HTML_TEMPLATE.format(
        report_date=date.today().strftime("%d %B %Y"),
        ta_timestamp=_get_ta_timestamp(rows),
        total_stocks=len(rows),
        strong_buy_count=len(strong_buy_rows),
        buy_count=len(buy_rows),
        watch_count=len(watch_rows),
        avoid_count=len(avoid_rows),
        index_cards=index_cards_html,
        strong_buy_section=strong_buy_section,
        buy_section=buy_section,
        watch_section=watch_section,
        avoid_section=avoid_section,
    )

    output_path.write_text(html, encoding="utf-8")
    log.info("HTML report generated: %s (%d stocks)", output_path, len(rows))
    return str(output_path)
