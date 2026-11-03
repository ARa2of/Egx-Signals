"""In-session live refresh - bar-close loop off the EGX beta tape.

Why this exists: the daily EOD job only ever sees finished bars, so during a
session the 1h/15m panels sit on yesterday's data. This loop aggregates the
running session's bars from the LIVE tape (EGX beta market-watch for the names,
EGX beta index-data for EGX30), splices them over the TV history and runs the
SAVED bundles - no refit, ever.

Discipline (docs/EGX_FORECAST_MAINTENANCE.md section 7):
    - only COMPLETED bars are used; the in-progress bar is never predicted on
    - overlay bars live in data/intraday/live/ and never touch the TV cache the
      EOD job trains/predicts from
    - EGX beta stamps are converted to the FEED clock (Europe/London wall-clock,
      verified against every session on disk 2026-11-01)

Usage:
    python -m egxfc.live --selftest            # offline check of the bar grid
    python -m egxfc.live --once                # one poll + one cycle
    python -m egxfc.live --until 14:35         # run the loop to a Cairo time
    python -m egxfc.live --clear               # wipe the overlay
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import pandas as pd

from . import config as C
from . import features as F
from . import predict as P
from . import store as S
from . import overlay as O

CAIRO = ZoneInfo("Africa/Cairo")
LONDON = ZoneInfo("Europe/London")
TRADING_DAYS = {6, 0, 1, 2, 3}          # Sun..Thu
SESSION_OPEN = (10, 0)                  # Cairo
SESSION_CLOSE = (14, 30)
LEG_MIN = {"15m": 15, "1h": 60}
# RETIRED 2026-11-03: intraday point predicts disabled (skill ~0). The loop
# still aggregates bars + writes the overlay (future hourly-range overlay
# consumes it); predict_live/store/show are skipped when False.
ENABLE_INTRADAY_PREDICT = False
LIVE_JSON = C.DATA / "live_forecast.json"

HDRS = {
    "Accept": "application/json",
    "x-egx-bff-request": "1",
    "Referer": "https://beta.egx.com.eg/en/market/market-watch",
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
}
WATCH_URL = "https://beta.egx.com.eg/api/bff/egx/market-watch"
INDEX_URL = "https://beta.egx.com.eg/api/bff/egx/index-data"


# ------------------------------------------------------------------ clock maths
def now_cairo() -> datetime.datetime:
    return datetime.datetime.now(CAIRO)


def feed_stamp(dt_cairo: datetime.datetime) -> pd.Timestamp:
    """Cairo instant -> the stamp the TV feed uses (Europe/London, tz-naive)."""
    return pd.Timestamp(dt_cairo.astimezone(LONDON).replace(tzinfo=None))


def session_bars(day: datetime.date, leg: str) -> list[datetime.datetime]:
    """Aware Cairo datetimes for every bar START of that session."""
    t = datetime.datetime.combine(day, datetime.time(*SESSION_OPEN), CAIRO)
    end = datetime.datetime.combine(day, datetime.time(*SESSION_CLOSE), CAIRO)
    step = datetime.timedelta(minutes=LEG_MIN[leg])
    out = []
    while t < end:
        out.append(t)
        t += step
    return out


def current_bar_start(now: datetime.datetime, leg: str) -> datetime.datetime | None:
    if now.weekday() not in TRADING_DAYS:
        return None
    bars = session_bars(now.date(), leg)
    if not bars:
        return None
    close = datetime.datetime.combine(now.date(), datetime.time(*SESSION_CLOSE), CAIRO)
    if now < bars[0] or now > close:
        return None
    pick = None
    for b in bars:
        nxt = b + datetime.timedelta(minutes=LEG_MIN[leg])
        if b <= now < min(nxt, close + datetime.timedelta(minutes=1)):
            pick = b
            break
    return pick


# ---------------------------------------------------------------------- tape I/O
def _get(url: str, params: dict, timeout: int = 25):
    import requests

    try:
        r = requests.get(url, params=params, headers=HDRS, timeout=timeout)
        txt = r.text.strip()
        if not txt.startswith("{") and not txt.startswith("["):
            return None
        return r.json()
    except Exception:
        return None


def fetch_tape(tickers: set[str]) -> dict:
    """{TICKER: {price, open, high, low, vol}} plus EGX30 from index-data."""
    out: dict = {}
    j = _get(WATCH_URL, {"Page": 1, "PageSize": 300})
    rows = ((j or {}).get("data") or {}).get("data") or []
    for row in rows:
        ric = (row.get("reuters") or "").upper()
        tk = ric.split(".")[0] if ric else ""
        if tk not in tickers:
            continue
        px = row.get("lastPrice")
        if not px:
            continue
        out[tk] = {
            "price": float(px),
            "open": float(row.get("openPrice") or px),
            "high": float(row.get("high") or px),
            "low": float(row.get("low") or px),
            "vol": float(row.get("volume") or 0.0),
        }

    j = _get(INDEX_URL, {"interval": "1", "indexName": "CASE30"})
    live = ((j or {}).get("data") or {}).get("liveIndex") or {}
    lvl = live.get("indexClose")
    if lvl:
        out["EGX30"] = {
            "price": float(lvl),
            "open": float(live.get("indexOpen") or lvl),
            "high": float(live.get("high") or lvl),
            "low": float(live.get("low") or lvl),
            "vol": 0.0,
        }
    return out


# ------------------------------------------------------------------- aggregation
class Bars:
    """Tick -> completed-bar aggregator, one in-progress bar per (name, leg)."""

    def __init__(self) -> None:
        self.state: dict = {}
        self.cum: dict = {}

    def cycle(self, tape: dict, now: datetime.datetime) -> list[tuple]:
        done = []
        for leg in LEG_MIN:
            start = current_bar_start(now, leg)
            if start is None:
                continue
            for sym, row in tape.items():
                px = row.get("price")
                if not px:
                    continue
                vol = row.get("vol") or 0.0
                key = (sym, leg)
                st = self.state.get(key)
                if st is None or st["start"] != start:
                    if st is not None and not st.get("seed"):
                        done.append(self._close(sym, leg, st))
                    # joining mid-bar (loop started late) leaves an incomplete
                    # volume/open on the first bar seen -> mark it and drop it
                    seed = (now - start) > datetime.timedelta(seconds=90)
                    st = {"start": start, "open": px, "high": px, "low": px,
                          "close": px, "v0": self.cum.get(sym, 0.0), "vn": vol,
                          "seed": seed}
                    if (start.hour, start.minute) == SESSION_OPEN and row.get("open"):
                        st["open"] = row["open"]
                    self.state[key] = st
                else:
                    st["high"] = max(st["high"], px)
                    st["low"] = min(st["low"], px)
                    st["close"] = px
                    st["vn"] = vol
                self.cum[sym] = max(self.cum.get(sym, 0.0), vol)
        return done

    def flush(self) -> list[tuple]:
        """Close every in-progress bar (session close - no next start to trigger it)."""
        done = []
        for (sym, leg), st in list(self.state.items()):
            if not st.get("seed"):
                done.append(self._close(sym, leg, st))
        self.state.clear()
        return done

    @staticmethod
    def _close(sym: str, leg: str, st: dict) -> tuple:
        vol = max(0.0, st["vn"] - st["v0"])
        return (sym, leg, feed_stamp(st["start"]),
                float(st["open"]), float(st["high"]), float(st["low"]),
                float(st["close"]), float(vol))


def write_overlay(bars: list[tuple]) -> dict:
    by: dict = {}
    for sym, leg, ts, o, h, l, c, v in bars:
        by.setdefault((sym, leg), []).append(
            {"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})
    touched = {}
    for (sym, leg), rows in by.items():
        df = pd.DataFrame(rows).set_index("ts").sort_index()
        df.index.name = "datetime"
        p = F.LIVE_DIR / f"{sym}_{leg}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            try:
                old = pd.read_parquet(p)
                if old is not None and len(old):
                    df = pd.concat([old, df])
            except Exception:
                pass
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df.to_parquet(p)
        touched.setdefault(leg, []).append(sym)
    return touched


def clear_overlay() -> int:
    n = 0
    if F.LIVE_DIR.exists():
        for p in F.LIVE_DIR.glob("*.parquet"):
            p.unlink()
            n += 1
    if LIVE_JSON.exists():
        LIVE_JSON.unlink()
    return n


# -------------------------------------------------------------------- prediction
def predict_live(syms: list[str], leg: str) -> list[dict]:
    out = []
    for sym in sorted(syms):
        if sym == "EGX30":
            continue
        try:
            d = P.predict_leg(sym, leg, live=True)
        except Exception as e:  # noqa: BLE001
            out.append({"ticker": sym, "leg": leg, "error": f"{type(e).__name__}: {e}"})
            continue
        if d is None:
            continue
        out.append({
            "ticker": sym, "leg": leg, "asof": d.get("asof"),
            "close": round(d["close"], 4),
            "pred_close": round(d.get("pred_close", float("nan")), 4),
            "move_pct": round((d.get("pred_ret") or 0.0) * 100.0, 3),
            "fires": bool(d.get("fires")),
            "skill": d.get("skill"), "model": d.get("model"),
        })
    return out


def store_predictions(rows: list[dict]) -> None:
    cur = {}
    if LIVE_JSON.exists():
        try:
            cur = json.loads(LIVE_JSON.read_text(encoding="utf-8"))
        except Exception:
            cur = {}
    rows_all = cur.get("rows") or []
    idx = {(r.get("ticker"), r.get("leg")): r for r in rows_all}
    for r in rows:
        idx[(r.get("ticker"), r.get("leg"))] = r
    payload = {"generated": now_cairo().strftime("%Y-%m-%d %H:%M"),
               "rows": sorted(idx.values(), key=lambda r: (r.get("leg") or "", r.get("ticker") or ""))}
    LIVE_JSON.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")


def store_tape(tape: dict, now: datetime.datetime) -> None:
    """Keep live_forecast.json fresh while intraday point predicts stay retired.

    Writes per-name 15m tape closes only (no preds) so the forum poster passes
    keep running: the LIVE tape row + footer + range-overlay line. Never raises.
    """
    try:
        asof = feed_stamp(now).strftime("%Y-%m-%d %H:%M:%S")
        rows = [{"ticker": sym, "leg": "15m",
                 "asof": asof, "close": round(float(v.get("price")), 4)}
                for sym, v in sorted(tape.items())
                if sym != "EGX30" and (v.get("price") or 0) > 0]
        if rows:
            store_predictions(rows)
    except Exception:
        pass


def show(rows: list[dict], quiet: bool) -> None:
    if quiet or not rows:
        return
    print("%-6s %-4s %-6s %9s %9s %8s %5s" % ("TKR", "LEG", "MODEL", "CLOSE", "PRED", "MOVE%", "FIRE"))
    for r in rows:
        if r.get("error"):
            print("%-6s %-4s ERROR %s" % (r["ticker"], r["leg"], r["error"][:40]))
            continue
        print("%-6s %-4s %-6s %9.2f %9.2f %+8.3f %5s" % (
            r["ticker"], r["leg"], str(r.get("model"))[:6], r["close"],
            r["pred_close"], r["move_pct"], "yes" if r["fires"] else "no"))


# ------------------------------------------------------------------------ runner
def universe() -> list[str]:
    return S.tickers() or list(C.DEFAULT_UNIVERSE)


def run_loop(until: str, interval: int, gui_every: int, quiet: bool) -> int:
    hh, mm = [int(x) for x in until.split(":")]
    stop = datetime.datetime.combine(now_cairo().date(), datetime.time(hh, mm), CAIRO)
    warm = (datetime.datetime.combine(now_cairo().date(),
                                      datetime.time(*SESSION_OPEN), CAIRO)
            - datetime.timedelta(minutes=2))
    tickers = set(universe()) | {"EGX30"}
    agg = Bars()
    fired: set = set()          # overlay checkpoints already computed today
    last_gui = time.monotonic()
    cycles = 0
    while True:
        now = now_cairo()
        if now >= stop:
            break
        if now.weekday() not in TRADING_DAYS:
            break
        if now < warm:                       # pre-open: nothing to aggregate yet
            time.sleep(min(120, max(10, interval)))
            continue
        tape = fetch_tape(tickers)
        cycles += 1
        if tape:
            store_tape(tape, now)
            done = agg.cycle(tape, now)
            if done:
                touched = write_overlay(done)
                if ENABLE_INTRADAY_PREDICT:
                    for leg, syms in touched.items():
                        rows = predict_live(syms, leg)
                        store_predictions(rows)
                        show(rows, quiet)
                if gui_every and (time.monotonic() - last_gui) >= gui_every:
                    rebuild_gui(quiet)
                    last_gui = time.monotonic()
            ck = O.active_checkpoint(now)
            if ck is not None and ck not in fired:
                payload = O.compute_all(now)     # never raises; None pre-bars
                if payload and payload.get("rows"):
                    fired.add(ck)
                    rebuild_gui(quiet)
                    if not quiet:
                        print(f"[live] overlay Cairo {ck} "
                              f"(feed {payload.get('feed')}) "
                              f"rows={len(payload['rows'])}", flush=True)
        if not quiet:
            print(f"[live] {now.strftime('%H:%M:%S')} cycles={cycles} names={len(tape)}", flush=True)
        time.sleep(max(10, interval))
    close_dt = datetime.datetime.combine(now_cairo().date(),
                                        datetime.time(*SESSION_CLOSE), CAIRO)
    done = agg.flush() if now_cairo() >= close_dt else []
    if done:                    # session over: close the last bar(s) still open
        touched = write_overlay(done)
        if ENABLE_INTRADAY_PREDICT:
            for leg, syms in touched.items():
                rows = predict_live(syms, leg)
                store_predictions(rows)
                show(rows, quiet)
    if not quiet:
        print(f"[live] done {now_cairo().strftime('%H:%M')} cycles={cycles}", flush=True)
    return 0


def rebuild_gui(quiet: bool = True) -> None:
    cmd = [sys.executable, "-m", "egxfc.gui"]
    try:
        subprocess.run(cmd, cwd=str(C.SCRIPTS), capture_output=quiet, timeout=1800)
    except Exception:
        pass


# ----------------------------------------------------------------------- selftest
def selftest() -> int:
    """Grid must reproduce the feed's own bar stamps; aggregation must be exact."""
    fails = 0
    print("== bar grid vs feed stamps ==")
    for sym in ("EGAL", "CPCI"):
        for leg in LEG_MIN:
            p = C.DATA / f"{sym}_{leg}.parquet"
            if not p.exists():
                print(f"  {sym} {leg}: no cache, skipped")
                continue
            df = pd.read_parquet(p)
            day = df.index.max().date()
            want = [str(t)[:16] for t in df[df.index.date == day].index]
            grid = [str(feed_stamp(b))[:16] for b in session_bars(day, leg)]
            ok = want == grid
            fails += 0 if ok else 1
            print(f"  {sym:5} {leg:4} feed={len(want):2} grid={len(grid):2} "
                  f"{'MATCH' if ok else 'MISMATCH'}")
            if not ok:
                print(f"    feed {want[:3]}...{want[-1:]}")
                print(f"    grid {grid[:3]}...{grid[-1:]}")

    print("== tick -> bar aggregation ==")
    p = C.DATA / "EGAL_15m.parquet"
    df = pd.read_parquet(p)
    day = df.index.max().date()
    tod = df[df.index.date == day]
    agg = Bars()
    cum = 0.0
    emitted = []
    for ts, row in tod.iterrows():
        cairo = (ts.tz_localize(LONDON) if ts.tzinfo is None else ts).astimezone(CAIRO)
        tape = {"EGAL": {"price": float(row["open"]), "open": float(row["open"]),
                         "high": float(row["high"]), "low": float(row["low"]),
                         "vol": cum}}
        emitted += agg.cycle(tape, cairo + datetime.timedelta(seconds=30))
        cum += float(row.get("volume") or 0.0)
        tape["EGAL"]["price"] = float(row["close"])
        tape["EGAL"]["vol"] = cum
        emitted += agg.cycle(tape, cairo + datetime.timedelta(minutes=10))
    emitted += agg.flush()
    bad = 0
    bars15 = [e for e in emitted if e[1] == "15m"]
    for sym, leg, ts, o, h, l, c, v in bars15:
        ref = tod[tod.index == ts]
        if ref.empty:
            bad += 1
            print(f"  EXTRA bar {ts}")
            continue
        rc = float(ref["close"].iloc[0])
        if abs(c - rc) / rc > 1e-9:
            bad += 1
            print(f"  {ts} close {c} != feed {rc}")
        rv = float(ref["volume"].iloc[0])
        if abs(v - rv) > max(1.0, rv * 1e-9):
            bad += 1
            print(f"  {ts} volume {v:.0f} != feed {rv:.0f}")
    ok_agg = bad == 0 and len(bars15) == len(tod)
    print(f"  bars emitted={len(bars15)} of {len(tod)} expected, "
          f"{'PASS' if ok_agg else 'FAIL'}")
    fails += 0 if ok_agg else 1
    bars1h = [e for e in emitted if e[1] == "1h"]
    want1h = [b for b in session_bars(day, "1h")]
    ok1h = (len(bars1h) == len(want1h)
            and [str(b[2])[:16] for b in bars1h] == [str(feed_stamp(b))[:16] for b in want1h])
    print(f"  1h bars={len(bars1h)} of {len(want1h)} expected, "
          f"{'PASS' if ok1h else 'FAIL'}")
    fails += 0 if ok1h else 1
    emitted = bars15

    print("== live prediction path ==")
    clear_overlay()
    # splice a real session over the history: the name's bars AND the index bars
    # (idx_ret1 / idx_ret_sess are matrix features, so the index overlay matters)
    p30 = C.DATA / "EGX30_15m.parquet"
    d30 = pd.read_parquet(p30) if p30.exists() else None
    idx_day = None
    if d30 is not None and len(d30):
        cnt = d30.groupby(d30.index.date).size()
        full = sorted(d for d, n in cnt.items() if n >= 17)
        idx_day = full[-1] if full else None
    sig: list[tuple] = []
    for sym in ("EGAL", "EGX30"):
        q = C.DATA / f"{sym}_15m.parquet"
        if not q.exists() or idx_day is None:
            continue
        dq = pd.read_parquet(q)
        dq = dq[dq.index.date == idx_day]
        dq = dq[~dq.index.duplicated(keep="last")].sort_index()
        for ts2, row2 in dq.iterrows():
            sig.append((sym, "15m", pd.Timestamp(ts2),
                        float(row2["open"]), float(row2["high"]), float(row2["low"]),
                        float(row2["close"]), float(row2.get("volume") or 0.0)))
    if idx_day is None or not sig:
        print("  no cached bars to splice, SKIPPED")
    else:
        probe = 999.0                       # distinctive close proves the splice
        last_ts = [b[2] for b in sig if b[0] == "EGAL"][-1]
        sig = [b if not (b[0] == "EGAL" and b[2] == last_ts)
               else (b[0], b[1], b[2], b[3], max(b[4], probe),
                     min(b[5], probe), probe, b[7]) for b in sig]
        write_overlay(sig)
        try:
            bundle = S.load_bundle("EGAL", "15m")
            feats = (bundle or {}).get("features") or []
            df = F.build_matrix("EGAL", "15m", verbose=False, live=True)
            ok_px = False
            if df is not None and len(df):
                hit = df[df["ts"] == last_ts]
                ok_px = (len(hit) == 1 and abs(float(hit["close"].iloc[0]) - probe) < 1e-9
                         and not pd.isna(hit[[c for c in ("idx_ret1", "idx_ret_sess")
                                              if c in df.columns]].iloc[0]).any())
            print(f"  {idx_day} session spliced (close {probe:.0f} + index bars): "
                  f"{'PASS' if ok_px else 'FAIL'}")
            fails += 0 if ok_px else 1
            d = P.predict_leg("EGAL", "15m", live=True)
            if d is None:
                print("  predict FAIL (no model or no matrix)")
                fails += 1
            else:
                exp = df.dropna(subset=feats).iloc[-1] if (df is not None and feats) else None
                print(f"  asof={d['asof']} close={d['close']:.2f} "
                      f"pred={d.get('pred_close', float('nan')):.2f} "
                      f"fires={d.get('fires')}")
                ok_ts = exp is not None and str(d["asof"])[:16] == str(exp["ts"])[:16]
                ok_c = exp is not None and abs(float(d["close"]) - float(exp["close"])) < 1e-9
                print(f"  asof == last complete live row: {'PASS' if ok_ts else 'FAIL'}")
                print(f"  close == that row's close:      {'PASS' if ok_c else 'FAIL'}")
                fails += (0 if ok_ts else 1) + (0 if ok_c else 1)
        finally:
            clear_overlay()
    print("SELFTEST", "PASS" if fails == 0 else f"FAIL ({fails})")
    return 0 if fails == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--until", default="14:35")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--gui-every", type=int, default=1800, help="seconds, 0 = never")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--clear", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    if a.clear:
        print(f"[live] cleared {clear_overlay()} overlay file(s)")
        return 0
    if a.selftest:
        return selftest()

    tickers = set(universe()) | {"EGX30"}
    if a.once:
        agg = Bars()
        tape = fetch_tape(tickers)
        if not a.quiet:
            print(f"[live] tape names={len(tape)} at {now_cairo().strftime('%H:%M:%S')} Cairo")
        done = agg.cycle(tape, now_cairo())
        if done:
            touched = write_overlay(done)
            for leg, syms in touched.items():
                show(predict_live(syms, leg), a.quiet)
        elif not a.quiet:
            print("[live] no bar closed yet this cycle (in-progress bar only)")
        return 0
    return run_loop(a.until, a.interval, a.gui_every, a.quiet)


if __name__ == "__main__":
    raise SystemExit(main())
