"""Layer-1 walk-forward rescore (P3 of the range verify-and-harden pass).

READ-ONLY with respect to models/ and registry.json. Fits are offline, one
per (name, month, target): train = everything strictly before the month
(production ``_slice``), score = the month. The fit never sees the scored
month, and there is NO per-month candidate selection - the model family is
fixed per name/target from the registry - so the SELECT-on-test-month bias
found in the P1 audit does not apply to this table.

    python -m egxfc.rescore_l1 --selftest     # frozen definitions on fixtures
    python -m egxfc.rescore_l1                # full table (all complete months)
    python -m egxfc.rescore_l1 --months 2026-01 2026-10
    python -m egxfc.rescore_l1 --names MFPC EGAS

FROZEN DEFINITIONS (declared 2026-11-03 before any number was produced;
``selftest()`` checks each against hand-computable fixtures):

  units       all errors divided by ATR(14) of day t (true range through t,
              known at the EOD run of day t)
  model       pred level = close_t * (1 + yhat), yhat from the registry
              family for that name/target, fitted on rows < month start
  no_change   pred = close_t on both edges (the registry's rmse0 hurdle;
              a degenerate band, kept for continuity only)
  persist     pred_high = high_t, pred_low = low_t
  typ_exc     pred_high = close_t + median(high-close, last 20 incl. t)
              pred_low  = close_t - median(close-low,  last 20 incl. t)
  atr_box     pred_high = close_t + 0.5*ATR_t, pred_low = close_t - 0.5*ATR_t
              (k = 0.5 declared a priori: half the typical range ~ 0.5*ATR)
  metrics     MAE(ATR) per edge + pooled band, signed bias per edge,
              coverage per edge and both, Winkler interval score at
              alpha = 0.2 (80% nominal): score = (u-l) + (2/alpha)*miss,
              applied to each realised edge against the SAME band and
              averaged, miss-size tail (mean / p95 overshoot)
  support     one common eval set per name-month: rows where every feature,
              target, ATR, excursion median and raw OHLC are finite - all
              methods are scored on identical rows
  pooling     per-day mean across names first, then mean across days
              (equal day weight; no name-count weighting)
  bootstrap   day-block (block = 5 sessions, B = 1000, seed = 20261103) CI
              on delta = model - best baseline; best baseline = min pooled
              band-MAE over persist / typ_exc / atr_box, fixed per month
              before resampling; verdict BEATS if hi < 0, LOSES if lo > 0
  guards      month skipped when train < 300 or test < 15 rows
              (MIN_TRAIN_ROWS / MIN_TEST_ROWS); months run = every complete
              calendar month from 2016-10 through last complete month
  known caveat the last train row's LABEL may fall on the test month's first
              session (same boundary convention as production `_slice`,
              1 row) - declared, not changed here
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from . import train as T

ALPHA = 0.2              # 80% nominal
BLOCK = 5                # day-block bootstrap, sessions
B_BOOT = 1000
SEED = 20261103
ATR_N = 14
TYP_N = 20
ATR_K = 0.5              # a priori, not tuned
TARGETS = ("t_high", "t_low")
METHODS = ("model", "no_change", "persist", "typ_exc", "atr_box")
BASELINES = ("persist", "typ_exc", "atr_box")
FIRST_MONTH = "2016-10"  # matrix starts 2016-09 (indicator warmup)
MIN_TRAIN = C.MIN_TRAIN_ROWS["daily"]
MIN_TEST = C.MIN_TEST_ROWS["daily"]
# common eval support: every method is scored only on rows where all of
# these are finite (features, targets, ATR, excursion medians, raw OHLC)
SUPPORT = list(TARGETS) + ["atr", "up20", "dn20", "high", "low", "raw_close"]
OUT_DIR = C.ROOT / "output" / "intraday_prediction"


# --------------------------------------------------------------------- data
def load_raw(sym: str) -> pd.DataFrame | None:
    """Daily bars + causal ATR(14) + trailing-20 excursion medians, by date."""
    p = C.DATA / f"{sym}_daily.parquet"
    if not p.exists():
        return None
    raw = pd.read_parquet(p)
    raw.index = pd.to_datetime(raw.index)
    raw = raw.reset_index()
    dtc = "datetime" if "datetime" in raw.columns else raw.columns[0]
    raw = raw.rename(columns={dtc: "date"})
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    if not {"high", "low", "close"}.issubset(raw.columns):
        return None
    raw = (raw[["date", "high", "low", "close"]]
           .dropna().drop_duplicates("date").sort_values("date")
           .rename(columns={"close": "raw_close"}).reset_index(drop=True))
    pc = raw["raw_close"].shift(1)
    tr = pd.concat([(raw["high"] - raw["low"]),
                    (raw["high"] - pc).abs(),
                    (raw["low"] - pc).abs()], axis=1).max(axis=1)
    raw["atr"] = tr.rolling(ATR_N, min_periods=ATR_N).mean()
    raw["up20"] = (raw["high"] - raw["raw_close"]).rolling(
        TYP_N, min_periods=TYP_N).median()
    raw["dn20"] = (raw["raw_close"] - raw["low"]).rolling(
        TYP_N, min_periods=TYP_N).median()
    return raw


def prep(sym: str, families: dict) -> tuple[pd.DataFrame, list[str], dict] | None:
    """Merged matrix+raw frame, feature list, per-target family for `sym`."""
    try:
        mx = F.build_matrix(sym, "daily", verbose=False)
        raw = load_raw(sym)
        if mx is None or not len(mx) or raw is None or not len(raw):
            return None
        mx = mx.copy()
        mx["date"] = pd.to_datetime(mx["date"]).dt.normalize()
        mg = mx.merge(raw, on="date", how="inner")
        feats = list(F.feature_list("daily"))
        fam = families.get(sym) or {}
        fam = {t: fam.get(t) or "ridge" for t in TARGETS}
        return mg, feats, fam
    except Exception as e:  # noqa: BLE001
        print(f"  {sym}: prep FAIL {type(e).__name__}: {e}", flush=True)
        return None


# ------------------------------------------------------- frozen definitions
def interval_score(y, l, u, alpha: float = ALPHA) -> np.ndarray:
    """Winkler/interval score: width + (2/alpha) penalty outside [l, u]."""
    y, l, u = np.asarray(y, float), np.asarray(l, float), np.asarray(u, float)
    pen = np.where(y > u, (2.0 / alpha) * (y - u),
                   np.where(y < l, (2.0 / alpha) * (l - y), 0.0))
    return (u - l) + pen


def edge_metrics(ph, pl, rh, rl, atr) -> dict:
    """All frozen metrics for one (method x name x month) block, ATR units."""
    ph, pl, rh, rl, atr = (np.asarray(x, float) for x in (ph, pl, rh, rl, atr))
    eh, el = np.abs(ph - rh) / atr, np.abs(pl - rl) / atr
    sh = interval_score(rh, pl, ph) / atr
    sl = interval_score(rl, pl, ph) / atr
    mh = np.maximum(0.0, rh - ph) / atr       # realised above predicted high
    ml = np.maximum(0.0, pl - rl) / atr       # realised below predicted low
    return {
        "n": int(len(ph)),
        "mae_band_atr": float(((eh + el) / 2.0).mean()),
        "mae_high_atr": float(eh.mean()),
        "mae_low_atr": float(el.mean()),
        "cov_high": float((rh <= ph).mean()),
        "cov_low": float((rl >= pl).mean()),
        "cov_both": float(((rh <= ph) & (rl >= pl)).mean()),
        "winkler_atr": float(((sh + sl) / 2.0).mean()),
        "bias_high_atr": float(((ph - rh) / atr).mean()),
        "bias_low_atr": float(((pl - rl) / atr).mean()),
        "miss_high_mean_atr": float(mh.mean()),
        "miss_high_p95_atr": float(np.percentile(mh, 95)),
        "miss_low_mean_atr": float(ml.mean()),
        "miss_low_p95_atr": float(np.percentile(ml, 95)),
    }


def block_boot_ci(d: np.ndarray, block: int = BLOCK, b: int = B_BOOT,
                  seed: int = SEED) -> tuple[float, float, float]:
    """Moving-block bootstrap CI for the mean of per-day deltas."""
    d = np.asarray(d, float)
    n = len(d)
    if n == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    if n < block:
        draws = [rng.integers(0, n, n).mean() for _ in range(b)]
    else:
        nb = int(np.ceil(n / block))
        draws = []
        for _ in range(b):
            starts = rng.integers(0, n - block + 1, size=nb)
            samp = np.concatenate([d[s:s + block] for s in starts])[:n]
            draws.append(samp.mean())
    return (float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)), float(np.mean(draws)))


def complete_months() -> list[str]:
    today = pd.Timestamp.today().normalize()
    last = (today.replace(day=1) - pd.Timedelta(days=1)).strftime("%Y-%m")
    return [str(p) for p in pd.period_range(FIRST_MONTH, last, freq="M")]


def load_families() -> dict:
    try:
        reg = json.loads((C.ROOT / "models" / "registry.json").read_text(
            encoding="utf-8"))
        out = {}
        for sym, v in (reg.get("models") or {}).items():
            d = (v or {}).get("daily") or {}
            out[sym] = d.get("model_by_target") or {}
        return out
    except Exception:  # noqa: BLE001
        return {}


# ------------------------------------------------------------------- run
def run(months: list[str], names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    fams = load_families()
    recs, day_rows, skipped = [], [], []
    for si, sym in enumerate(names, 1):
        pp = prep(sym, fams)
        if pp is None:
            skipped.append({"month": "*", "sym": sym, "why": "no data"})
            continue
        mg, feats, fam = pp
        for mo in months:
            tr, te = T._slice(mg, "daily", mo)
            trd = tr.dropna(subset=feats + SUPPORT)
            ted = te.dropna(subset=feats + SUPPORT)
            if len(trd) < MIN_TRAIN or len(ted) < MIN_TEST:
                skipped.append({"month": mo, "sym": sym,
                                "why": f"train={len(trd)} test={len(ted)}"})
                continue
            Xtr = trd[feats].to_numpy(float)
            Xte = ted[feats].to_numpy(float)
            yh = {}
            try:
                for tgt in TARGETS:
                    est = T.candidates()[fam.get(tgt, "ridge")]
                    est.fit(Xtr, trd[tgt].to_numpy(float))
                    yh[tgt] = est.predict(Xte)
            except Exception as e:  # noqa: BLE001
                skipped.append({"month": mo, "sym": sym,
                                "why": f"fit {type(e).__name__}"})
                continue
            c = ted["raw_close"].to_numpy(float)
            atr = ted["atr"].to_numpy(float)
            rh = c * (1.0 + ted["t_high"].to_numpy(float))   # realised high
            rl = c * (1.0 + ted["t_low"].to_numpy(float))    # realised low
            methods = {
                "model": (c * (1.0 + yh["t_high"]), c * (1.0 + yh["t_low"])),
                "no_change": (c, c),
                "persist": (ted["high"].to_numpy(float),
                            ted["low"].to_numpy(float)),
                "typ_exc": (c + ted["up20"].to_numpy(float),
                            c - ted["dn20"].to_numpy(float)),
                "atr_box": (c + ATR_K * atr, c - ATR_K * atr),
            }
            dates = ted["date"].to_numpy()
            for mname, (mph, mpl) in methods.items():
                m = edge_metrics(mph, mpl, rh, rl, atr)
                recs.append({"month": mo, "sym": sym, "method": mname, **m})
                eh = np.abs(mph - rh) / atr
                el = np.abs(mpl - rl) / atr
                for d in pd.unique(dates):
                    sel = dates == d
                    day_rows.append({
                        "month": mo, "date": d, "sym": sym, "method": mname,
                        "band": float(((eh[sel] + el[sel]) / 2.0).mean())})
        print(f"  [{si}/{len(names)}] {sym}: {len([r for r in recs if r['sym'] == sym])} rows",
              flush=True)
    metrics = pd.DataFrame(recs)
    daydf = pd.DataFrame(day_rows)
    if skipped:
        sk = pd.DataFrame(skipped)
        print(f"skipped blocks: {len(sk)} "
              f"(sample: {sk.head(3).to_dict('records')})", flush=True)
    return metrics, daydf


def delta_table(daydf: pd.DataFrame) -> pd.DataFrame:
    """Per month: model vs best baseline, day-block bootstrap CI, verdict."""
    rows = []
    months = list(dict.fromkeys(daydf["month"])) + ["__ALL__"]
    for mo in months:
        d = daydf if mo == "__ALL__" else daydf[daydf["month"] == mo]
        piv = d.pivot_table(index="date", columns="method", values="band",
                            aggfunc="mean")
        if "model" not in piv or len(piv) < 3:
            continue
        pooled = piv.mean()
        base = min(BASELINES, key=lambda b: pooled.get(b, np.inf))
        if base not in piv:
            continue
        delta = (piv["model"] - piv[base]).to_numpy(float)
        lo, hi, bmean = block_boot_ci(delta)
        rows.append({
            "month": mo, "n_days": len(piv),
            "mae_model": float(pooled["model"]),
            "best_base": base, "mae_base": float(pooled[base]),
            "delta": float(delta.mean()), "ci_lo": lo, "ci_hi": hi,
            "boot_mean": bmean,
            "verdict": "BEATS" if hi < 0 else ("LOSES" if lo > 0 else "MIXED"),
        })
    return pd.DataFrame(rows)


def name_table(daydf: pd.DataFrame) -> pd.DataFrame:
    """Per name: months model beats its month's best baseline."""
    rows = []
    for sym, d in daydf.groupby("sym"):
        piv = d.pivot_table(index=["month", "date"], columns="method",
                            values="band", aggfunc="mean")
        if "model" not in piv:
            continue
        wins = tot = 0
        deltas = []
        for mo, g in piv.groupby(level="month"):
            pooled = g.mean()
            cands = [b for b in BASELINES if b in pooled]
            if not cands:
                continue
            base = min(cands, key=lambda b: pooled[b])
            delta = float((g["model"] - g[base]).mean())
            deltas.append(delta)
            tot += 1
            wins += delta < 0
        rows.append({"sym": sym, "months": tot, "beats": wins,
                     "mean_delta": float(np.mean(deltas)) if deltas else np.nan})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("mean_delta")
    return out


# --------------------------------------------------------------- selftest
def selftest() -> int:
    """Frozen definitions on hand-computable fixtures (no refits)."""
    fails = 0
    print("== rescore_l1 frozen-definition selftest ==")

    def check(name, got, want, tol=1e-9):
        nonlocal fails
        ok = (np.allclose(got, want, atol=tol, rtol=0) if np.all(np.isfinite(np.asarray(got, float)))
              and np.all(np.isfinite(np.asarray(want, float))) else got == want)
        if ok:
            print(f"  PASS {name}")
        else:
            print(f"  FAIL {name}: got {got!r} want {want!r}")
            fails += 1

    # 1. ATR(14): 13 flat bars (TR=2) + one wide bar (TR=11) -> 37/14
    n = 14
    bars = pd.DataFrame({"high": [101.0] * 13 + [110.0],
                         "low": [99.0] * 13 + [99.0],
                         "raw_close": [100.0] * 13 + [105.0]})
    pc = bars["raw_close"].shift(1)
    tr = pd.concat([(bars["high"] - bars["low"]),
                    (bars["high"] - pc).abs(),
                    (bars["low"] - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(ATR_N, min_periods=ATR_N).mean()
    check("1 atr14", float(atr.iloc[-1]), 37.0 / 14.0)

    # 2. baselines on one declared day: c=100 h=102 l=98 atr=2 up=1.5 dn=1.0
    c, h, l, atr_t, up, dn = 100.0, 102.0, 98.0, 2.0, 1.5, 1.0
    check("2 persist", (h, l), (102.0, 98.0))
    check("2 typ_exc", (c + up, c - dn), (101.5, 99.0))
    check("2 atr_box", (c + ATR_K * atr_t, c - ATR_K * atr_t), (101.0, 99.0))

    # 3. interval score: width 2, 2/alpha = 10
    check("3 inside", float(interval_score(5.0, 4.0, 6.0)), 2.0)
    check("3 above", float(interval_score(8.0, 4.0, 6.0)), 22.0)
    check("3 below", float(interval_score(2.0, 4.0, 6.0)), 22.0)

    # 4. edge_metrics on two hand-scored days
    ph = np.array([101.0, 100.0]); pl = np.array([99.0, 97.0])
    rh = np.array([102.0, 100.0]); rl = np.array([98.0, 97.0])
    atr = np.array([2.0, 3.0])
    m = edge_metrics(ph, pl, rh, rl, atr)
    check("4 mae_band", m["mae_band_atr"], 0.25)
    check("4 mae_high/low", (m["mae_high_atr"], m["mae_low_atr"]), (0.25, 0.25))
    check("4 coverage", (m["cov_high"], m["cov_low"], m["cov_both"]),
          (0.5, 0.5, 0.5))
    check("4 winkler", m["winkler_atr"], 3.5)
    check("4 bias", (m["bias_high_atr"], m["bias_low_atr"]), (-0.25, 0.25))
    check("4 miss mean", (m["miss_high_mean_atr"], m["miss_low_mean_atr"]),
          (0.25, 0.25))
    check("4 miss p95", m["miss_high_p95_atr"], 0.475)

    # 5. bootstrap: deterministic, ordered, brackets the mean
    d = np.linspace(-1.0, 1.0, 20)
    a = block_boot_ci(d); b = block_boot_ci(d)
    check("5 boot deterministic", a, b)
    check("5 boot ordered", bool(a[0] <= a[2] <= a[1]), True)

    # 6. walk-forward guard on real data: train strictly before month
    pp = prep("MFPC", load_families())
    if pp is None:
        check("6 walk-forward guard", "prep None", "frame")
    else:
        mg, feats, fam = pp
        tr, te = T._slice(mg, "daily", "2026-05")
        check("6 train < month",
              bool(tr["date"].max() < pd.Timestamp("2026-05-01")), True)
        check("6 test inside month",
              bool(te["date"].min() >= pd.Timestamp("2026-05-01")
                   and te["date"].max() < pd.Timestamp("2026-06-01")), True)
        check("6 family fixed", bool(fam.get("t_high") in
                                     ("ridge", "rf", "lgbm")), True)

    print(f"rescore_l1 selftest: {'PASS' if fails == 0 else f'FAIL({fails})'}")
    return fails


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--months", nargs=2, default=None,
                    metavar=("START", "END"), help="inclusive YYYY-MM range")
    ap.add_argument("--names", nargs="+", default=None)
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    months = complete_months()
    if a.months:
        months = [m for m in months if a.months[0] <= m <= a.months[1]]
    names = a.names or list(C.DEFAULT_UNIVERSE)
    print(f"rescore_l1: {len(months)} months ({months[0]}..{months[-1]}), "
          f"{len(names)} names", flush=True)
    metrics, daydf = run(months, names)
    if not len(metrics):
        print("no rows scored")
        return 1
    deltas = delta_table(daydf)
    names_tbl = name_table(daydf)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUT_DIR / "rescore_l1.csv", index=False)
    deltas.to_csv(OUT_DIR / "rescore_l1_delta.csv", index=False)
    names_tbl.to_csv(OUT_DIR / "rescore_l1_by_name.csv", index=False)

    # markdown table for the enhancements doc
    md = ["# Layer-1 walk-forward rescore (ATR units)",
          "",
          f"Frozen definitions per `egxfc/rescore_l1.py` · "
          f"{len(months)} months ({months[0]}..{months[-1]}) · "
          f"{len(names)} names · day-block bootstrap "
          f"(block={BLOCK}, B={B_BOOT}, seed={SEED}) · alpha={ALPHA}",
          "",
          "| month | days | model | best base | base | delta [95% CI] | verdict |",
          "|---|---|---|---|---|---|---|"]
    for _, r in deltas.iterrows():
        md.append(f"| {r['month']} | {r['n_days']} | {r['mae_model']:.3f} | "
                  f"{r['best_base']} | {r['mae_base']:.3f} | "
                  f"{r['delta']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] | "
                  f"{r['verdict']} |")
    md += ["", "## Per-name (months model beats that month's best baseline)",
           "", "| name | months | beats | mean delta |", "|---|---|---|---|"]
    for _, r in names_tbl.iterrows():
        md.append(f"| {r['sym']} | {r['months']} | {r['beats']} | "
                  f"{r['mean_delta']:.3f} |")
    (OUT_DIR / "rescore_l1.md").write_text("\n".join(md) + "\n",
                                           encoding="utf-8")

    print("\n".join(md[5:]))
    v = deltas["verdict"].value_counts().to_dict() if len(deltas) else {}
    print(f"\nverdicts: {v}")
    print(f"wrote {OUT_DIR / 'rescore_l1.csv'}")
    print(f"wrote {OUT_DIR / 'rescore_l1_delta.csv'}")
    print(f"wrote {OUT_DIR / 'rescore_l1.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
