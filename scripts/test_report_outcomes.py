"""Focused test: the Trade Outcomes table must use the store's evaluated
(intraday first-touch) outcomes, not the legacy close-only walk.

Cases:
  A) tp2_hit in store + price fell after -> must show TP2, not OPEN
  B) stop_loss in store                     -> must show SL, not OPEN
  C) genuinely pending (horizon not elapsed) -> may show OPEN
"""
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pyarrow as pa

from src.store.signal_store import SIGNAL_SCHEMA
from src.output.html_report import generate_html_report

today = date.today()
d = lambda n: today - timedelta(days=n)

rows = [{
    "Selected Stock": "TEST",
    "Recommendation": "Watch",  # report's own rows don't matter for this test
    "Current EGP Price": 100.0,
}]

store = pd.DataFrame([
    # A) tp2_hit — intraday touched TP2 then price collapsed. Legacy walk
    #    (close-only) would call this OPEN.
    dict(run_date=d(10), ticker="AAA", close=100.0, recommendation="Buy",
         entry_price=100.0, stop_loss=95.0, tp1=102.0, tp2=104.0, tp3=107.0,
         score=70.0, outcome="tp2_hit", outcome_date=d(8),
         entry_date=d(9), entry_price_actual=100.5,
         pnl_pct=3.48, r_multiple=1.79, hold_days=1),
    # B) stop_loss
    dict(run_date=d(12), ticker="BBB", close=50.0, recommendation="Buy",
         entry_price=50.0, stop_loss=48.0, tp1=51.0, tp2=52.0, tp3=54.0,
         score=65.0, outcome="stop_loss", outcome_date=d(11),
         entry_date=d(11), entry_price_actual=50.2,
         pnl_pct=-4.38, r_multiple=-1.1, hold_days=1),
    # C) pending — legitimately still open
    dict(run_date=d(2), ticker="CCC", close=30.0, recommendation="Buy",
         entry_price=30.0, stop_loss=28.0, tp1=31.0, tp2=32.0, tp3=34.0,
         score=60.0, outcome="pending", outcome_date=None,
         entry_date=None, entry_price_actual=None,
         pnl_pct=-1.0, r_multiple=-0.3, hold_days=2),
    # D) expired (time exit)
    dict(run_date=d(30), ticker="DDD", close=20.0, recommendation="Buy",
         entry_price=20.0, stop_loss=19.0, tp1=20.5, tp2=21.0, tp3=22.0,
         score=55.0, outcome="expired", outcome_date=d(8),
         entry_date=d(29), entry_price_actual=20.1,
         pnl_pct=2.0, r_multiple=0.9, hold_days=21),
], columns=None)
store["regime"] = "uptrend"

for c in ["outcome_date", "entry_date"]:
    store[c] = pd.to_datetime(store[c])

out = Path("output/_test_report.html")
generate_html_report(rows, str(out), signal_store=store)
html = out.read_text(encoding="utf-8")

# Extract the trade-outcome rows region
import re
checks = []

def badge_for(ticker):
    m = re.search(rf"<b>{ticker}</b></td>.*?outcome-badge[^>]*>([A-Z0-9]+)<", html, re.S)
    return m.group(1) if m else None

def row_for(ticker):
    m = re.search(rf"<tr>\s*<td>[^<]*</td>\s*<td>[^<]*</td>\s*<td>[^<]*</td>\s*<td><b>{ticker}</b></td>(.*?)</tr>", html, re.S)
    return m.group(1) if m else None

a = badge_for("AAA")
checks.append(("A tp2_hit -> TP2 badge", a == "TP2", a))
b = badge_for("BBB")
checks.append(("B stop_loss -> SL badge", b == "SL", b))
c = badge_for("CCC")
checks.append(("C pending -> OPEN badge", c == "OPEN", c))
d_b = badge_for("DDD")
checks.append(("D expired -> TIME badge", d_b == "TIME", d_b))

# Row A must show exit price 104.0 (tp2) and entry 100.5 (actual)
rowA = row_for("AAA")
checks.append(("A exit price = 104.00", rowA is not None and "104.00" in rowA, rowA))
checks.append(("A entry = 100.50", rowA is not None and "100.50" in rowA, rowA))

# Row B exit price = 48.00 (SL)
rowB = row_for("BBB")
checks.append(("B exit price = 48.00", rowB is not None and "48.00" in rowB, rowB))

# Row C: exit date and price shown as em-dash (OPEN)
rowC = row_for("CCC")
checks.append(("C shows open dashes", rowC is not None and "—" in rowC, rowC))

ok = True
for name, passed, detail in checks:
    print(("PASS" if passed else "FAIL"), name, "|", (detail or "")[:120])
    ok = ok and passed

print(f"\n{sum(1 for _, p, _ in checks if p)}/{len(checks)} checks passed")
sys.exit(0 if ok else 1)
