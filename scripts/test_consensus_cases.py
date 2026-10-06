#!/usr/bin/env python3
"""Replay consensus rules for the cases the user flagged."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def consensus(base_rec, ml_conv, cs_signal, cs_conf, candle, entry_action,
              entry_source="", strong=None, adx=30, golden=True, diamond=True,
              macd_bull=True, death=False):
    ml_rec = "Buy" if ml_conv >= 35 else ("Watch" if ml_conv >= 20 else "Avoid")

    # ChartScan sanitize (mirrors daily_run)
    cs_result = None if cs_signal is None else {"signal": cs_signal, "confidence": cs_conf,
                                                "buy_patterns": 0, "sell_patterns": 0}
    if cs_result and cs_result["signal"] == "Sell":
        if cs_conf < 0.50 or candle == "bullish" and cs_conf < 0.70:
            cs_result["signal"] = "Neutral"
            cs_result["sanitized"] = True

    if cs_result and cs_result["signal"] == "Buy" and cs_conf >= 0.40:
        cs_rec = "Buy"
    elif cs_result and cs_result["signal"] == "Sell" and cs_conf >= 0.50:
        cs_rec = "Avoid"
    elif cs_result and cs_conf > 0:
        cs_rec = "Watch"
    else:
        cs_rec = None

    named = [("Base", base_rec), ("ML", ml_rec), ("ChartScan", cs_rec)]
    active = [(n, r) for n, r in named if r is not None]
    buy_v = sum(1 for _, r in active if r == "Buy")
    avoid_v = sum(1 for _, r in active if r == "Avoid")
    watch_v = sum(1 for _, r in active if r == "Watch")
    agree = [n for n, r in active if r == "Buy"]

    if strong is None:
        strong = (
            base_rec == "Buy" and not death and golden and diamond
            and adx >= 30 and macd_bull
        )

    if strong and cs_rec != "Avoid":
        rec, basis = "Buy", f"Strong technicals ML={ml_rec} CS={cs_rec}"
    elif buy_v >= 2 and buy_v == len(active) and len(active) >= 2:
        rec, basis = "Strong Buy", f"all {len(active)} Buy"
    elif buy_v >= 2:
        rec, basis = "Buy", f"{buy_v}/{len(active)} Buy {agree}"
    elif buy_v == 1 and avoid_v == 0:
        rec, basis = "Buy", f"1 Buy no Avoid {agree}"
    elif buy_v == 1:
        rec, basis = "Watch", f"1 Buy others disagree"
    elif watch_v >= 1 and avoid_v == 0:
        rec, basis = "Watch", "no Buy no Avoid"
    elif avoid_v >= 1 and buy_v == 0:
        rec, basis = "Avoid", str(named)
    else:
        rec, basis = "Watch", "insufficient"

    # Entry override
    ea = entry_action.upper()
    src = entry_source.upper()
    chase = any(k in ea for k in ("CHASE", "WAIT", "NO CLEAN")) or "FALLBACK" in src
    poor = "POOR SETUP" in ea
    if rec in ("Buy", "Strong Buy"):
        if poor and not strong:
            rec = "Watch"
            basis = f"planner {entry_action} downgrades"
        elif chase:
            basis += f"; entry={entry_action} (chase kept)"

    return rec, basis, ml_rec, cs_rec, cs_result


cases = [
    # ETEL: base Buy 64.5, ML conf 0-15 Avoid/Watch, ChartScan Buy low conf,
    # candle neutral, planner fallback chase, strong technicals True
    dict(label="ETEL", base_rec="Buy", ml_conv=12, cs_signal="Buy", cs_conf=0.37,
         candle="neutral", entry_action="CHASE — NO CLEAN PULLBACK",
         entry_source="Fallback (no clean setup)", adx=38.8, strong=True),
    # EFIH: base Buy, ML conf 30 -> Watch (NOT Avoid), ChartScan Neutral
    dict(label="EFIH", base_rec="Buy", ml_conv=30, cs_signal="Neutral", cs_conf=0.33,
         candle="neutral", entry_action="WAIT FOR PULLBACK", adx=26.1, strong=True),
    # MBSC: ChartScan Sell 58% but candle bullish -> sanitized Neutral
    dict(label="MBSC", base_rec="Buy", ml_conv=67, cs_signal="Sell", cs_conf=0.58,
         candle="bullish", entry_action="BUY NOW", adx=31.4, strong=True),
    # ARCC: high score, ML Buy, CS Neutral, entry chase — momentum filter not simulated
    dict(label="ARCC", base_rec="Buy", ml_conv=85, cs_signal="Neutral", cs_conf=0.41,
         candle="neutral", entry_action="CHASE — NO CLEAN PULLBACK", adx=35.1, strong=True),
]

print(f"{'case':8} {'final':10} {'ml_rec':8} {'cs_rec':8} basis")
print("-" * 100)
ok = True
for c in cases:
    rec, basis, ml_rec, cs_rec, cs_raw = consensus(**{k: v for k, v in c.items() if k != "label"})
    print(f"{c['label']:8} {rec:10} {ml_rec:8} {str(cs_rec):8} {basis}")
    if c["label"] == "ETEL":
        ok = ok and rec == "Buy"
        print(f"         ETEL chartscan after sanitize: {cs_raw}")
    if c["label"] == "EFIH":
        ok = ok and rec == "Buy" and ml_rec == "Watch"
    if c["label"] == "MBSC":
        ok = ok and rec == "Buy" and cs_rec != "Avoid"
        print(f"         MBSC chartscan after sanitize: {cs_raw}")

print()
print("EXPECTATIONS:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
