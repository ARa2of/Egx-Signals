"""Unit tests for the limit-entry first-touch outcome simulation.

The old simulation took next-day open as the entry unconditionally, so a
gap-up open "filled" above the TP levels and produced impossible
negative-PnL TP wins; outcome_date was the last bar of the download window
(not the touch bar) and hold_days counted bars to the window end.

Cases:
  A) gap-up above TP: no fill at open, later revisit -> fill + TP1, real touch date
  B) stop loss touched -> exit at SL, r = -1
  C) gap straight through SL -> exit at the market open, not the stop price
  D) TP2 hit on a later bar -> touch date + bars-held recorded
  E) no touch within full horizon -> expired at horizon close
  F) limit never filled -> no_entry
  G) horizon not elapsed -> pending with mark-to-market fields
  H) knife day: fill bar closes below limit without touching SL -> knife_exit at close
"""
import sys
import types
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.store.signal_store import simulate_outcomes

today = date.today()
H = 5  # short horizon for tests

def frame(run_dt, rows):
    """rows: list of (open, high, low, close); first bar is run+1."""
    idx = [run_dt + timedelta(days=1 + i) for i in range(len(rows))]
    return pd.DataFrame(
        {"Open": [r[0] for r in rows], "High": [r[1] for r in rows],
         "Low": [r[2] for r in rows], "Close": [r[3] for r in rows]},
        index=pd.DatetimeIndex(idx))

A_dt, B_dt, C_dt, D_dt, E_dt, F_dt, G_dt, H_dt = (
    [today - timedelta(days=10)] * 6 + [today - timedelta(days=3), today - timedelta(days=10)]
)

FRAMES = {
    # A: day1 no fill (low 108 > 100); day2 low 99 fills at 100, high 101.5 -> TP1 (101)
    "TSTA.CA": frame(A_dt, [(110, 112, 108, 111), (100.5, 101.5, 99, 101)]),
    # B: fill at open 99.5, same bar low 93 <= SL 94 -> SL at 94
    "TSTB.CA": frame(B_dt, [(99.5, 100, 93, 95)]),
    # C: gap straight through SL: open 90 < SL 94 -> fill at 90, stop at market (90)
    "TSTC.CA": frame(C_dt, [(90, 92, 88, 91)]),
    # D: bar1 fill no TP; bar2 high 103.5 -> TP2 (103)
    "TSTD.CA": frame(D_dt, [(100, 100.5, 98, 100), (101, 103.5, 100.5, 103)]),
    # E: 5 bars, fill bar1 (closes above the limit — not a knife day), no touch ever
    "TSTE.CA": frame(E_dt, [(99, 100.5, 97, 100.2)] + [(99, 100.5, 97, 99)] * 4),
    # F: limit never filled (all lows above 100)
    "TSTF.CA": frame(F_dt, [(108, 110, 105, 109)] * 3),
    # G: 2 bars, fill, no touch, horizon (5d) not elapsed
    "TSTG.CA": frame(G_dt, [(99, 100.5, 98, 100), (99, 100, 98, 99.5)]),
    # H: knife day — fills at 99.5, never touches SL 94, closes 96 < limit 100
    "TSTH.CA": frame(H_dt, [(99.5, 100.2, 99, 96)]),
}

fake_yf = types.ModuleType("yfinance")
fake_yf.download = lambda ticker, **kw: FRAMES[ticker]
sys.modules["yfinance"] = fake_yf

trades = pd.DataFrame([
    dict(ticker="TSTA", run_date=A_dt, entry_price=100.0, stop_loss=94.0, tp1=101.0, tp2=103.0, tp3=106.0),
    dict(ticker="TSTB", run_date=B_dt, entry_price=100.0, stop_loss=94.0, tp1=101.0, tp2=103.0, tp3=106.0),
    dict(ticker="TSTC", run_date=C_dt, entry_price=100.0, stop_loss=94.0, tp1=101.0, tp2=103.0, tp3=106.0),
    dict(ticker="TSTD", run_date=D_dt, entry_price=100.0, stop_loss=94.0, tp1=101.0, tp2=103.0, tp3=106.0),
    dict(ticker="TSTE", run_date=E_dt, entry_price=100.0, stop_loss=94.0, tp1=101.0, tp2=103.0, tp3=106.0),
    dict(ticker="TSTF", run_date=F_dt, entry_price=100.0, stop_loss=94.0, tp1=101.0, tp2=103.0, tp3=106.0),
    dict(ticker="TSTG", run_date=G_dt, entry_price=100.0, stop_loss=94.0, tp1=101.0, tp2=103.0, tp3=106.0),
    dict(ticker="TSTH", run_date=H_dt, entry_price=100.0, stop_loss=94.0, tp1=101.0, tp2=103.0, tp3=106.0),
])

out = simulate_outcomes(trades, horizon_days=H).set_index("ticker")
checks = []

def chk(name, cond, detail=""):
    checks.append((name, bool(cond), detail))

# A: gap-up then revisit
a = out.loc["TSTA"]
chk("A outcome tp1_hit", a["outcome"] == "tp1_hit", a["outcome"])
chk("A actual entry = limit 100", abs(a["entry_price_actual"] - 100.0) < 1e-6, a["entry_price_actual"])
chk("A exit = tp1 101", abs(a["exit_price"] - 101.0) < 1e-6, a["exit_price"])
chk("A outcome_date = fill bar (day2)", a["outcome_date"] == (A_dt + timedelta(days=2)), a["outcome_date"])
chk("A hold = 0 bars", a["hold_days"] == 0, a["hold_days"])
chk("A pnl ~ +0.85%", abs(a["pnl_pct"] - 0.85) < 0.01, a["pnl_pct"])

# B: normal SL
b = out.loc["TSTB"]
chk("B outcome stop_loss", b["outcome"] == "stop_loss", b["outcome"])
chk("B exit = SL 94", abs(b["exit_price"] - 94.0) < 1e-6, b["exit_price"])
chk("B r ~ -1.0 (to open entry)", abs(b["r_multiple"] + 1.0) < 0.01, b["r_multiple"])

# C: gap through SL
c = out.loc["TSTC"]
chk("C outcome stop_loss", c["outcome"] == "stop_loss", c["outcome"])
chk("C exit = open 90 (market)", abs(c["exit_price"] - 90.0) < 1e-6, c["exit_price"])
chk("C actual entry = open 90", abs(c["entry_price_actual"] - 90.0) < 1e-6, c["entry_price_actual"])
chk("C pnl ~ -0.15% (costs only)", abs(c["pnl_pct"] + 0.15) < 0.01, c["pnl_pct"])

# D: TP2 on later bar
dd = out.loc["TSTD"]
chk("D outcome tp2_hit", dd["outcome"] == "tp2_hit", dd["outcome"])
chk("D outcome_date = bar2", dd["outcome_date"] == (D_dt + timedelta(days=2)), dd["outcome_date"])
chk("D hold = 1 bar", dd["hold_days"] == 1, dd["hold_days"])
chk("D pnl ~ +2.85%", abs(dd["pnl_pct"] - 2.85) < 0.01, dd["pnl_pct"])

# E: full horizon, no touch
e = out.loc["TSTE"]
chk("E outcome expired", e["outcome"] == "expired", e["outcome"])
chk("E hold = 4 (5 bars)", e["hold_days"] == 4, e["hold_days"])
chk("E outcome_date = last bar", e["outcome_date"] == (E_dt + timedelta(days=5)), e["outcome_date"])

# F: never filled
f = out.loc["TSTF"]
chk("F outcome no_entry", f["outcome"] == "no_entry", f["outcome"])
chk("F pnl = 0", f["pnl_pct"] == 0.0, f["pnl_pct"])

# G: pending, horizon not elapsed
g = out.loc["TSTG"]
chk("G outcome pending", g["outcome"] == "pending", g["outcome"])
chk("G entry fields recorded", g["entry_date"] is not None and abs(g["entry_price_actual"] - 99.0) < 1e-6,
    (g["entry_date"], g["entry_price_actual"]))
chk("G exit_price = mark close 99.5", abs(g["exit_price"] - 99.5) < 1e-6, g["exit_price"])

# H: knife day — closed below the limit, stop never touched
h = out.loc["TSTH"]
chk("H outcome knife_exit", h["outcome"] == "knife_exit", h["outcome"])
chk("H exit = close 96", abs(h["exit_price"] - 96.0) < 1e-6, h["exit_price"])
chk("H hold = 0 bars", h["hold_days"] == 0, h["hold_days"])
# fill at min(open 99.5, limit 100) = 99.5; pnl = (96-99.5)/99.5 - costs
chk("H actual entry = open 99.5", abs(h["entry_price_actual"] - 99.5) < 1e-6, h["entry_price_actual"])
chk("H pnl ~ -3.67%", abs(h["pnl_pct"] + 3.67) < 0.02, h["pnl_pct"])

ok = True
for name, passed, detail in checks:
    print(("PASS" if passed else "FAIL"), name, "|", detail)
    ok = ok and passed

print(f"\n{sum(1 for _, p, _ in checks if p)}/{len(checks)} checks passed")
sys.exit(0 if ok else 1)
