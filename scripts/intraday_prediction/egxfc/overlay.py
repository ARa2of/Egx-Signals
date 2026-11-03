"""Hourly live range overlay - day-ahead high/low anchor, carved + shifted intraday.

Graduated 2026-11-03 from scripts/intraday_prediction/hourly_range_test2.py
(EGX30-only, 2-name prototype) to all 21 names via each stock's HOME index
(egxfc.enrich.home_key).

Rule: the overlay NARROWS or SHIFTS the day-ahead band as the session prints -
it never widens it. Updated band always contains the live price and everything
carved so far: rem_h >= max(carved_h, live), rem_l <= min(carved_l, live).

Checkpoints are keyed by CAIRO wall-clock: 11:00 / 12:00 / 13:00 (the
session's own clock, EGX 10:00-14:30 Cairo). TV feed stamps are
Europe/London tz-naive, so each ck's FEED label is derived per session date
with zoneinfo - the Cairo-London gap is 2h normally, but 1h during the spring
DST mismatch (London 03-29 -> Cairo 04-24) and 3h during the autumn mismatch
(London 10-25 -> Cairo 10-30). Hard-coded labels were wrong on those dates
(P1 audit, docs/HOURLY_RANGE_AUDIT_20261103.md s2.4):
    feed_cutoff(day, "11:00") -> "09:00" normal | "10:00" spring | "08:00" autumn
frac of the 18-bar session left: 14/18, 10/18, 6/18 (by Cairo ck).

Maths per checkpoint (unchanged from the prototype):
    beta      = cov(stock, home-index daily rets) / var(home) over daily closes
    idx_move  = home-index open -> checkpoint return
    pace      = vol_sofar / (avg20vol * frac_elapsed), clipped 0..5
    belief    = min(1, pace/1.5); pace < 0.8 -> x0.4
    w         = sqrt(frac_left); exhaustion (pace > 3 and carved covers >80%
                of predicted width) -> w x0.7
    adj       = anchor + beta * idx_move * live * belief   (each edge)
    upd       = live + (adj - live) * w
    rem       = carved-floored upd, live-containing
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

CKPTS = ["11:00", "12:00", "13:00"]  # canonical ck = Cairo wall clock
FRAC_LEFT = {"11:00": 14 / 18, "12:00": 10 / 18, "13:00": 6 / 18}
TZ_CAIRO = ZoneInfo("Africa/Cairo")
TZ_FEED = ZoneInfo("Europe/London")  # TV bar stamps (tz-naive wall clock)

_BETA_CACHE: dict = {}


def feed_cutoff(day, ck: str) -> str:
    """Europe/London wall-clock label for Cairo checkpoint `ck` on `day`.

    Zoneinfo-driven: correct on normal (2h), spring-mismatch (1h) and
    autumn-mismatch (3h) dates, including the transition days themselves
    (conversion happens at the checkpoint's actual instant).
    """
    hh, mm = int(ck[:2]), int(ck[3:5])
    t = _dt.datetime.combine(day, _dt.time(hh, mm), tzinfo=TZ_CAIRO)
    return t.astimezone(TZ_FEED).strftime("%H:%M")


def active_checkpoint(now_cairo) -> str | None:
    """Latest passed checkpoint (Cairo ck key) for a Cairo-aware datetime.

    None pre-11:00. Cairo checkpoints are fixed wall-clock, so this needs no
    date logic - the date only enters when converting to the feed label.
    """
    hm = now_cairo.strftime("%H:%M")
    pick = None
    for ck in CKPTS:
        if hm >= ck:
            pick = ck
    return pick


def clock_selftest() -> int:
    """Golden freeze of overlay maths + checkpoint semantics (no data files).

    Fixture values frozen 2026-11-03 BEFORE the P2 zoneinfo fix (P1 audit).
    Checkpoints are selected by FRAC_LEFT key order, so the fixture stays
    valid across the ck key-space change (feed labels -> Cairo labels).
    Covers: overlay_at pace-clip/exhaustion/slow-belief branches, day_overlay
    bar selection on winter/summer dates (normal 2h-gap: old code == new
    code), active_checkpoint semantics, the feed_cutoff DST table (winter,
    summer, spring 1h-mismatch Mar 30-Apr 24, autumn 3h-mismatch Oct 25-29,
    all four transition days), and day_overlay on both mismatch windows
    (autumn: old code looked 1h ahead; spring: old code returned None).
    """
    from datetime import date, datetime as dt
    fails = 0
    k0, k1, k2 = list(FRAC_LEFT)  # insertion order = 14/18, 10/18, 6/18
    print("== clock golden (frozen 2026-11-03) ==")

    def check(name, got, want):
        nonlocal fails
        if got == want:
            print(f"  PASS {name}")
        else:
            print(f"  FAIL {name}: got {got!r} want {want!r}")
            fails += 1

    # A. overlay_at branches: pace clip / exhaustion / slow belief
    a1 = overlay_at(k0, 100.0, 101.2, 99.1, 4000.0, 105.0, 95.0, 0.012, 0.6, 1000.0)
    check("A1 rem_h", a1["rem_h"], 105.0)
    check("A1 rem_l", a1["rem_l"], 96.22539479621452)
    check("A1 pace/w", (a1["pace"], a1["w"], a1["exhausted"]), (5.0, 0.882, False))
    a2 = overlay_at(k2, 104.0, 110.0, 99.0, 2500.0, 105.0, 95.0, -0.004, 0.6, 1000.0)
    check("A2 exhausted", (a2["exhausted"], a2["w"], a2["rem_h"], a2["rem_l"]),
          (True, 0.404, 110.0, 99.0))
    a3 = overlay_at(k1, 100.0, 100.5, 99.8, 50.0, 105.0, 95.0, 0.0, 0.6, 1000.0)
    check("A3 slow belief", (a3["pace"], a3["belief"], a3["rem_h"], a3["rem_l"]),
          (0.11, 0.03, 103.72677996249965, 96.27322003750035))

    def synth(day_s: str, open_hh: int = 8):
        """18 feed-stamped 15m bars, deterministic prices (default open 08:00)."""
        stamps = [f"{day_s} {open_hh + i // 4:02d}:{(i % 4) * 15:02d}" for i in range(18)]
        closes = [100.0 + 0.5 * i for i in range(18)]
        bars = pd.DataFrame({
            "datetime": stamps,
            "open": [c - 0.1 for c in closes],
            "high": [c + 0.4 for c in closes],
            "low": [c - 0.4 for c in closes],
            "close": closes,
            "volume": [1000.0] * 18,
        })
        idx = pd.DataFrame({
            "datetime": stamps,
            "open": [5000.0 + 2 * i for i in range(18)],
            "high": [5003.0 + 2 * i for i in range(18)],
            "low": [4998.0 + 2 * i for i in range(18)],
            "close": [5000.0 + 2 * i for i in range(18)],
            "volume": [100.0] * 18,
        })
        from datetime import timedelta
        ddates = [(dt.fromisoformat(day_s) - timedelta(days=d)).strftime("%Y-%m-%d")
                  for d in range(1, 11)]
        dly = pd.DataFrame({
            "datetime": [f"{d} 14:30" for d in ddates],
            "close": [95.0 + 0.3 * i for i in range(10)],
            "volume": [800.0] * 10,
        })
        return bars, idx, dly

    # B. day_overlay bar selection: winter + summer (normal 2h-gap dates)
    for label, day_s in (("winter", "2026-01-15"), ("summer", "2026-07-15")):
        bars, idx, dly = synth(day_s)
        r = day_overlay("GOLDEN", k0, date.fromisoformat(day_s),
                        bars, idx, dly, 105.0, 95.0)
        if r is None:
            check(f"B_{label}", "None", "overlay")
            continue
        check(f"B_{label}",
              (r["rem_h"], r["rem_l"], r["live"], r["carved"], r["pace"],
               r["belief"], r["w"], r["exhausted"], r["idx_move"], r["beta"]),
              (104.73210863385773, 95.91293759697577, 102.0, [102.4, 99.6],
               5.0, 1.0, 0.882, False, 0.0016, 0.6))

    # C. active_checkpoint: assert the FRAC_LEFT value (key-space agnostic)
    for hm, frac in (("10:59", None), ("11:00", 14 / 18), ("11:59", 14 / 18),
                     ("12:00", 10 / 18), ("13:00", 6 / 18), ("14:45", 6 / 18)):
        ck = active_checkpoint(dt(2026, 1, 15, int(hm[:2]), int(hm[3:])))
        check(f"C active {hm}", None if ck is None else FRAC_LEFT[ck], frac)

    # D. feed_cutoff DST table (literals frozen 2026-11-03 via zoneinfo)
    #    normal/both-DST = 2h gap; spring mismatch Mar 30-Apr 24 = 1h;
    #    autumn mismatch Oct 25-29 = 3h; transition days resolve at 11:00 Cairo.
    from datetime import date as _date
    table = {
        "2026-01-15": ("09:00", "10:00", "11:00"),  # winter (normal)
        "2026-07-15": ("09:00", "10:00", "11:00"),  # summer (both DST)
        "2026-03-28": ("09:00", "10:00", "11:00"),  # day before London flip
        "2026-03-29": ("10:00", "11:00", "12:00"),  # London flips 01:00 UTC
        "2026-04-05": ("10:00", "11:00", "12:00"),  # spring mismatch (1h)
        "2026-04-23": ("10:00", "11:00", "12:00"),  # last spring mismatch day
        "2026-04-24": ("09:00", "10:00", "11:00"),  # Cairo flips 00:00 local
        "2026-10-24": ("09:00", "10:00", "11:00"),  # day before London flip
        "2026-10-25": ("08:00", "09:00", "10:00"),  # London flips 01:00 UTC
        "2026-10-28": ("08:00", "09:00", "10:00"),  # autumn mismatch (3h)
        "2026-10-29": ("08:00", "09:00", "10:00"),  # last autumn mismatch day
        "2026-10-30": ("09:00", "10:00", "11:00"),  # Cairo flips 00:00 local
        "2026-11-03": ("09:00", "10:00", "11:00"),  # today (normal)
    }
    for day_s, wants in table.items():
        got = tuple(feed_cutoff(_date.fromisoformat(day_s), ck) for ck in CKPTS)
        check(f"D feed_cutoff {day_s}", got, wants)

    # E. day_overlay on both mismatch windows: exactly 5 past bars at ck1.
    #    autumn (open 07:00): old code filtered <=09:00 = 9 bars (look-ahead);
    #    spring (open 09:00): old code filtered <=09:00 = 1 bar -> None.
    #    Both must now select the same first-5-bars state as the B fixtures.
    want_b = (104.73210863385773, 95.91293759697577, 102.0, [102.4, 99.6],
              5.0, 1.0, 0.882, False, 0.0016, 0.6)
    for label, day_s, open_hh in (("autumn_3h", "2026-10-28", 7),
                                  ("spring_1h", "2026-04-05", 9)):
        bars, idx, dly = synth(day_s, open_hh)
        r = day_overlay("GOLDEN", k0, _date.fromisoformat(day_s),
                        bars, idx, dly, 105.0, 95.0)
        if r is None:
            check(f"E {label}", "None", "overlay")
            continue
        check(f"E {label}",
              (r["rem_h"], r["rem_l"], r["live"], r["carved"], r["pace"],
               r["belief"], r["w"], r["exhausted"], r["idx_move"], r["beta"]),
              want_b)
        check(f"E {label} feed label", r["feed"],
              feed_cutoff(_date.fromisoformat(day_s), k0))
    print(f"clock selftest: {'PASS' if fails == 0 else f'FAIL({fails})'}")
    return fails


def _bars_on_day(bars: pd.DataFrame, day) -> pd.DataFrame:
    dtc = "datetime" if "datetime" in bars.columns else bars.columns[0]
    b = bars.copy()
    b["_dt"] = pd.to_datetime(b[dtc])
    return b[b["_dt"].dt.date == day].sort_values("_dt")


def beta_vs_home(sym: str, dly: pd.DataFrame, idx_15m: pd.DataFrame,
                 cache: bool = True) -> float:
    """Beta of the stock vs its HOME index from daily closes. Cached per sym.

    cache=False (eval/replay) recomputes on the frame passed in, so callers
    can slice history <= session day for a walk-forward-honest beta.
    """
    if cache and sym in _BETA_CACHE:
        return _BETA_CACHE[sym]
    beta = 0.6
    try:
        ddtc = "datetime" if "datetime" in dly.columns else dly.columns[0]
        d = dly.copy()
        d["_dt"] = pd.to_datetime(d[ddtc])
        d["_d"] = d["_dt"].dt.date
        sret = d.groupby("_d")["close"].last().pct_change()
        ii = idx_15m.copy()
        idtc = "datetime" if "datetime" in ii.columns else ii.columns[0]
        ii["_dt"] = pd.to_datetime(ii[idtc])
        ii["_d"] = ii["_dt"].dt.date
        iret = ii.groupby("_d")["close"].last().pct_change()
        common = pd.DataFrame({"s": sret, "i": iret}).dropna()
        common = common[np.isfinite(common["s"]) & np.isfinite(common["i"])]
        common = common[(common["s"].abs() < 0.5) & (common["i"].abs() < 0.2)]
        if len(common) > 60:
            beta = float(common["s"].cov(common["i"]) / (common["i"].var() + 1e-12))
            beta = max(-0.5, min(2.0, beta))
    except Exception:
        pass
    if cache:
        _BETA_CACHE[sym] = beta
    return beta


def avg20_volume(dly: pd.DataFrame) -> float:
    try:
        return float(dly["volume"].tail(40).mean())
    except Exception:
        return 0.0


def overlay_at(ck: str, live: float, carved_h: float, carved_l: float,
               vol_sofar: float, pred_high: float, pred_low: float,
               idx_move: float, beta: float, avg20vol: float, *,
               shift: bool = True, belief_on: bool = True,
               exhaust_on: bool = True) -> dict:
    """Pure overlay maths for one checkpoint. Returns dict with rem_h/rem_l.

    Ablation flags (P4 eval only; defaults = production behaviour and are
    guarded by clock_selftest):
        shift=False      contain-only: widen the anchor only as far as needed
                         to contain the carve and the live price
        belief_on=False  beta shift with belief pinned to 1 (no pace scaling)
        exhaust_on=False no exhaustion damping of w
    """
    frac_el = 1.0 - FRAC_LEFT[ck]
    pace = vol_sofar / (avg20vol * frac_el + 1e-9) if avg20vol > 0 else 1.0
    pace = max(0.0, min(5.0, pace))
    belief = min(1.0, pace / 1.5)
    if pace < 0.8:
        belief *= 0.4
    if not belief_on:
        belief = 1.0
    w = np.sqrt(FRAC_LEFT[ck])
    pred_w = max(1e-9, pred_high - pred_low)
    carved_w = carved_h - carved_l
    exhausted = False
    if exhaust_on and pace > 3.0 and carved_w > 0.8 * pred_w:
        w *= 0.7
        exhausted = True
    if not shift:
        # A0 contain-only: anchor widened only to contain carve + live
        return {"rem_h": float(max(pred_high, carved_h, live)),
                "rem_l": float(min(pred_low, carved_l, live)),
                "pace": round(float(pace), 2), "belief": 1.0, "w": 1.0,
                "exhausted": False,
                "idx_move": round(float(idx_move), 5)}
    adj_h = pred_high + beta * idx_move * live * belief
    adj_l = pred_low + beta * idx_move * live * belief
    upd_h = live + (adj_h - live) * w
    upd_l = live + (adj_l - live) * w
    rem_h = max(carved_h, upd_h if upd_h >= live else max(upd_h, adj_h if adj_h < live else live))
    rem_l = min(carved_l, upd_l if upd_l <= live else min(upd_l, adj_l if adj_l > live else live))
    if rem_h < live:
        rem_h = max(carved_h, live)
    if rem_l > live:
        rem_l = min(carved_l, live)
    # never widen: clamp inside the predicted band when the carve allows it
    if rem_h > pred_high and carved_h <= pred_high:
        rem_h = max(carved_h, min(pred_high, max(live, upd_h)))
    if rem_l < pred_low and carved_l >= pred_low:
        rem_l = min(carved_l, max(pred_low, min(live, upd_l)))
    return {"rem_h": float(rem_h), "rem_l": float(rem_l),
            "pace": round(float(pace), 2), "belief": round(float(belief), 2),
            "w": round(float(w), 3), "exhausted": exhausted,
            "idx_move": round(float(idx_move), 5)}


def day_overlay(sym: str, ck: str, day, stock_15m: pd.DataFrame,
                idx_15m: pd.DataFrame, stock_daily: pd.DataFrame,
                pred_high: float, pred_low: float, *,
                beta_cache: bool = True, **kw) -> dict | None:
    """Full per-name checkpoint overlay. None when not enough bars yet.

    **kw are overlay_at ablation flags (shift / belief_on / exhaust_on);
    beta_cache=False recomputes beta on the frame passed in (eval/replay).
    """
    db = _bars_on_day(stock_15m, day)
    di = _bars_on_day(idx_15m, day)
    if len(db) < 4:
        return None
    feed_ck = feed_cutoff(day, ck)   # date-specific: fixes DST mismatch days
    past = db[db["_dt"].dt.strftime("%H:%M") <= feed_ck]
    if len(past) < 2:
        return None
    cki = di[di["_dt"].dt.strftime("%H:%M") <= feed_ck] if len(di) else di
    live = float(past.iloc[-1]["close"])
    carved_h, carved_l = float(past["high"].max()), float(past["low"].min())
    vol_sofar = float(past["volume"].sum())
    i_o = float(di.iloc[0]["close"]) if len(di) else float("nan")
    idx_move = (float(cki.iloc[-1]["close"]) / i_o - 1) if (len(cki) and np.isfinite(i_o) and i_o) else 0.0
    beta = beta_vs_home(sym, stock_daily, idx_15m, cache=beta_cache)
    vol = avg20_volume(stock_daily)
    r = overlay_at(ck, live, carved_h, carved_l, vol_sofar,
                   pred_high, pred_low, idx_move, beta, vol, **kw)
    r.update({"ticker": sym, "ck": ck, "cairo": ck, "feed": feed_ck,
              "live": live, "carved": [carved_h, carved_l],
              "anchor": [float(pred_high), float(pred_low)], "beta": round(beta, 3)})
    return r


def selftest() -> int:
    """Offline: replay Oct 2026 on 2 names, assert live+carve containment."""
    import pathlib
    import joblib
    fails = clock_selftest()
    fails += io_selftest()
    ROOT = pathlib.Path("D:/Stocks/EGX Project")
    DATA = ROOT / "data" / "intraday"
    from . import enrich as E
    for sym in ("MBEG", "MASR"):
        try:
            b = joblib.load(ROOT / "models" / sym / "daily.pkl")
            feats = pd.read_parquet(DATA / f"daily_features_{sym}.parquet")
            feats["_t"] = pd.to_datetime(feats["date"])
            feats = feats.sort_values("_t").reset_index(drop=True)
            octf = feats[(feats["_t"].dt.year == 2026) & (feats["_t"].dt.month == 10)].reset_index(drop=True)
            s15 = pd.read_parquet(DATA / f"{sym}_15m.parquet").reset_index()
            key = E.home_key(sym)
            i15 = pd.read_parquet(DATA / f"{key}_15m.parquet").reset_index()
            dly = pd.read_parquet(DATA / f"{sym}_daily.parquet").reset_index()
        except Exception as e:
            print(f"  {sym} load FAIL {e}")
            fails += 1
            continue
        n = 0
        for i, r in octf.iterrows():
            day = r["_t"].date()
            base = float(r["Close"])
            if not np.isfinite(base):
                continue
            try:
                feats_r = r.to_frame().T.reindex(columns=list(b["features"])).ffill().fillna(0)
                ph = float(b["models"]["t_high"].predict(feats_r.values)[0])
                pl = float(b["models"]["t_low"].predict(feats_r.values)[0])
            except Exception:
                continue
            for ck in CKPTS:
                o = day_overlay(sym, ck, day, s15, i15, dly,
                                base * (1 + ph), base * (1 + pl))
                if o is not None:
                    assert o["rem_h"] >= o["live"] and o["rem_l"] <= o["live"], "live containment"
                    assert o["rem_h"] >= o["carved"][0] and o["rem_l"] <= o["carved"][1], "carve floor"
                    n += 1
        ok = n > 0
        print(f"  {sym} home={key} overlays={n}: {'PASS' if ok else 'FAIL'}")
        fails += 0 if ok else 1
    print("overlay selftest:", "PASS" if fails == 0 else f"FAIL({fails})")
    return fails


# ------------------------------------------------------- live wiring (11/12/13)
def _paths():
    from . import config as C
    return C


def spliced_15m(sym: str) -> pd.DataFrame | None:
    """TV 15m cache + live session overlay, de-duplicated (live wins)."""
    C = _paths()
    p = C.DATA / f"{sym}_15m.parquet"
    if not p.exists():
        return None
    try:
        tv = pd.read_parquet(p).reset_index()
    except Exception:
        return None
    try:
        from . import features as F
        ov = F.overlay(sym, "15m")
        if ov is not None and len(ov):
            ov = ov.reset_index()
            key = "ts" if "ts" in ov.columns else ov.columns[0]
            tv = pd.concat([tv, ov], ignore_index=True)
            dtc = "datetime" if "datetime" in tv.columns else tv.columns[0]
            tv["_s"] = pd.to_datetime(tv[dtc].fillna(tv[key] if key in tv.columns else pd.NaT))
            tv = tv.sort_values("_s").drop_duplicates(subset=["_s"], keep="last").drop(columns=["_s"])
    except Exception:
        pass
    return tv.reset_index(drop=True)


def daily_anchors(sym: str) -> tuple | None:
    """(pred_high, pred_low, close) from the saved daily bundle. None if absent."""
    try:
        from . import predict as P
        d = P.predict_leg(sym, "daily", live=False)
        if not d:
            return None
        return (float(d["pred_high"]), float(d["pred_low"]), float(d["close"]))
    except Exception:
        return None


RANGE_FILE = "live_range.json"
RANGE_LOG = "range_outcomes.jsonl"   # append-only checkpoint log (P6)


def _atomic_write_text(path, text: str) -> None:
    """Temp file + os.replace: readers never observe a partial JSON file."""
    path = pathlib.Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def log_range_rows(payload: dict, path=None) -> int:
    """Append checkpoint rows to the append-only outcomes log.

    Idempotent on (date, ck, symbol): replaying a cached payload or re-firing
    a checkpoint never duplicates rows. Best-effort - returns rows appended
    and never raises. Single-writer assumption (the live loop).
    """
    try:
        p = pathlib.Path(path) if path is not None else (_paths().DATA / RANGE_LOG)
        have = set()
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    have.add((r.get("date"), r.get("ck"), r.get("sym")))
                except Exception:  # noqa: BLE001
                    continue
        date, ck = payload.get("date"), payload.get("ck")
        new = []
        for sym, row in (payload.get("rows") or {}).items():
            key = (date, ck, sym)
            if key in have:
                continue
            have.add(key)
            new.append(json.dumps(
                {"date": date, "ck": ck, "cairo": payload.get("cairo"),
                 "feed": payload.get("feed"), "sym": sym, **row},
                ensure_ascii=False))
        if new:
            with p.open("a", encoding="utf-8") as f:
                for line in new:
                    f.write(line + "\n")
        return len(new)
    except Exception:  # noqa: BLE001
        return 0


def load_range_log(path=None) -> pd.DataFrame:
    """Read the outcomes log into a DataFrame (empty when absent/empty)."""
    p = pathlib.Path(path) if path is not None else (_paths().DATA / RANGE_LOG)
    if not p.exists():
        return pd.DataFrame()
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:  # noqa: BLE001
                continue
    return pd.DataFrame(rows)


def join_forward(df: pd.DataFrame, bars_by_sym: dict) -> pd.DataFrame:
    """Join realised forward extremes onto logged ckpt rows (next-EOD join).

    h_fwd/l_fwd = max/min of 15m bars strictly AFTER the row's date-specific
    feed label; NaN while the session has no forward bars yet. Adds the
    three-way split flags (already-in vs new extreme) and coverage.
    """
    if not len(df):
        return df
    out = df.copy()
    h_fwd = np.full(len(out), np.nan)
    l_fwd = np.full(len(out), np.nan)
    for i, row in enumerate(out.itertuples(index=False)):
        bars = bars_by_sym.get(row.sym)
        if bars is None or not len(bars):
            continue
        try:
            day = pd.Timestamp(row.date).date()
            eff = feed_cutoff(day, row.ck)
        except Exception:  # noqa: BLE001
            continue
        b = _bars_on_day(bars, day)
        if not len(b):
            continue
        fut = b[b["_dt"].dt.strftime("%H:%M") > eff]
        if not len(fut):
            continue
        h_fwd[i] = float(fut["high"].max())
        l_fwd[i] = float(fut["low"].min())
    out["h_fwd"] = h_fwd
    out["l_fwd"] = l_fwd
    ch = out["carved"].apply(
        lambda c: float(c[0]) if isinstance(c, (list, tuple)) and len(c) else np.nan)
    cl = out["carved"].apply(
        lambda c: float(c[1]) if isinstance(c, (list, tuple)) and len(c) else np.nan)
    out["carved_h"], out["carved_l"] = ch, cl
    out["new_high"] = (out["h_fwd"] > out["carved_h"] + 1e-9).where(
        out["h_fwd"].notna())
    out["new_low"] = (out["l_fwd"] < out["carved_l"] - 1e-9).where(
        out["l_fwd"].notna())
    out["cov_high"] = (out["h_fwd"] <= out["rem_h"] + 1e-9).where(
        out["h_fwd"].notna())
    out["cov_low"] = (out["l_fwd"] >= out["rem_l"] - 1e-9).where(
        out["l_fwd"].notna())
    return out


def io_selftest() -> int:
    """P6 I/O tests on a temp dir: idempotent log, atomic write, join.

    Never touches the production data dir.
    """
    import tempfile
    fails = 0
    print("== io selftest (temp dir) ==")

    def check(name, got, want):
        nonlocal fails
        if got == want:
            print(f"  PASS {name}")
        else:
            print(f"  FAIL {name}: got {got!r} want {want!r}")
            fails += 1

    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        logp = td / RANGE_LOG
        row = {"rem_h": 110.0, "rem_l": 60.0, "carved": [102.0, 98.0],
               "live": 101.0}
        row_alt = {"rem_h": 200.0, "rem_l": 10.0, "carved": [102.0, 98.0],
                   "live": 101.0}
        p1 = {"date": "2026-01-15", "ck": "11:00", "cairo": "11:00",
              "feed": "09:00", "rows": {"GOLD": dict(row), "ALT": dict(row_alt)}}
        check("1 first append", log_range_rows(p1, path=logp), 2)
        check("1 replay appends 0", log_range_rows(p1, path=logp), 0)
        p2 = {"date": "2026-01-15", "ck": "11:00", "cairo": "11:00",
              "feed": "09:00", "rows": {"GOLD": dict(row), "NEW": dict(row)}}
        check("1 partial appends 1", log_range_rows(p2, path=logp), 1)
        p3 = {"date": "2026-01-15", "ck": "12:00", "cairo": "12:00",
              "feed": "10:00", "rows": {"GOLD": dict(row)}}
        check("1 new ck appends 1", log_range_rows(p3, path=logp), 1)
        df = load_range_log(path=logp)
        check("1 log rows", len(df), 4)
        check("1 log keys",
              sorted(set(zip(df["date"], df["ck"], df["sym"]))),
              [("2026-01-15", "11:00", "ALT"), ("2026-01-15", "11:00", "GOLD"),
               ("2026-01-15", "11:00", "NEW"), ("2026-01-15", "12:00", "GOLD")])

        # 2. atomic write: content lands, no .tmp left behind
        jp = td / "live_range.json"
        _atomic_write_text(jp, json.dumps(p1, ensure_ascii=False))
        check("2 atomic content",
              json.loads(jp.read_text(encoding="utf-8"))["date"], "2026-01-15")
        check("2 no tmp left", (td / "live_range.json.tmp").exists(), False)

        # 3. join_forward on synthetic bars (2026-01-15, normal 2h gap):
        #    ck 11:00 -> feed 09:00 -> forward = stamps > 09:00 (i=5..17)
        #    h_fwd = max(100+i) = 117; l_fwd = min(50+i) = 55
        #    ck 12:00 -> feed 10:00 -> i=9..17: h_fwd 117, l_fwd 59
        stamps = [f"2026-01-15 {8 + i // 4:02d}:{(i % 4) * 15:02d}"
                  for i in range(18)]
        bars = pd.DataFrame({"datetime": stamps,
                             "high": [100.0 + i for i in range(18)],
                             "low": [50.0 + i for i in range(18)]})
        j = join_forward(df, {"GOLD": bars, "ALT": bars, "NEW": bars})
        g11 = j[(j["sym"] == "GOLD") & (j["ck"] == "11:00")].iloc[0]
        check("3 h_fwd ck11", float(g11["h_fwd"]), 117.0)
        check("3 l_fwd ck11", float(g11["l_fwd"]), 55.0)
        check("3 split ck11", (bool(g11["new_high"]), bool(g11["new_low"])),
              (True, True))
        check("3 cov narrow ck11", (bool(g11["cov_high"]), bool(g11["cov_low"])),
              (False, False))
        a11 = j[(j["sym"] == "ALT") & (j["ck"] == "11:00")].iloc[0]
        check("3 cov wide ALT", (bool(a11["cov_high"]), bool(a11["cov_low"])),
              (True, True))
        g12 = j[(j["sym"] == "GOLD") & (j["ck"] == "12:00")].iloc[0]
        check("3 l_fwd ck12", float(g12["l_fwd"]), 59.0)
        miss = df[df["sym"] == "GOLD"].head(1).copy()
        miss["date"] = "2026-01-16"          # no bars that day
        jm = join_forward(miss, {"GOLD": bars})
        check("3 no bars -> NaN", bool(pd.isna(jm["h_fwd"].iloc[0])), True)

    print(f"io selftest: {'PASS' if fails == 0 else f'FAIL({fails})'}")
    return fails


def compute_all(now_cairo) -> dict | None:
    """Overlay for every universe name at the active checkpoint; writes RANGE file.

    Returns the payload, or None pre-11:00 / on failure. Never raises.
    """
    try:
        ck = active_checkpoint(now_cairo)
        if ck is None:
            return None
        C = _paths()
        from . import enrich as E
        syms = list(C.DEFAULT_UNIVERSE)
        day = now_cairo.date()
        rows: dict = {}
        for sym in syms:
            try:
                anch = daily_anchors(sym)
                if anch is None:
                    continue
                ph, pl, _close = anch
                s15 = spliced_15m(sym)
                if s15 is None or len(s15) < 4:
                    continue
                key = E.home_key(sym)
                i15 = spliced_15m(key)
                if i15 is None or len(i15) < 2:
                    continue
                dlyp = C.DATA / f"{sym}_daily.parquet"
                if not dlyp.exists():
                    continue
                dly = pd.read_parquet(dlyp).reset_index()
                o = day_overlay(sym, ck, day, s15, i15, dly, ph, pl)
                if o is not None:
                    rows[sym] = o
            except Exception:
                continue
        payload = {"generated": now_cairo.strftime("%Y-%m-%d %H:%M"),
                   "date": now_cairo.strftime("%Y-%m-%d"),
                   "ck": ck, "cairo": ck, "feed": feed_cutoff(day, ck),
                   "rows": rows}
        # P6: atomic replace (readers never see partial JSON) + append-only
        # outcomes log (idempotent on date/ck/sym). Both best-effort.
        try:
            _atomic_write_text(
                C.DATA / RANGE_FILE,
                json.dumps(payload, ensure_ascii=False, indent=1))
        except Exception:
            pass
        log_range_rows(payload)
        return payload
    except Exception:
        return None


if __name__ == "__main__":
    import argparse
    import datetime
    from zoneinfo import ZoneInfo
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--test-clock", action="store_true",
                    help="golden + DST clock tests only (no data files)")
    ap.add_argument("--test-io", action="store_true",
                    help="P6 atomic-write/log tests only (temp dir)")
    ap.add_argument("--run", default=None, help="Cairo HH:MM to compute at (offline, uses cache)")
    a = ap.parse_args()
    if a.test_clock:
        raise SystemExit(clock_selftest())
    if a.test_io:
        raise SystemExit(io_selftest())
    if a.run:
        hh, mm = [int(x) for x in a.run.split(":")]
        now = datetime.datetime.combine(datetime.datetime.now(ZoneInfo("Africa/Cairo")).date(),
                                        datetime.time(hh, mm), ZoneInfo("Africa/Cairo"))
        p = compute_all(now)
        print("rows:", 0 if not p else len(p["rows"]), "ck:", None if not p else p["ck"])
    else:
        raise SystemExit(selftest())
