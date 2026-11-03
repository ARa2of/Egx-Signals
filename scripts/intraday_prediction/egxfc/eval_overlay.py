"""Layer-2 tautology-free overlay evaluation + ablation (P4).

Scores the hourly overlay against what it actually has to predict: the
FORWARD extremes (bars strictly after the checkpoint), not the whole-session
extremes that containment makes trivial once already printed (P1 audit s2.6).

    python -m egxfc.eval_overlay --selftest
    python -m egxfc.eval_overlay                    # walk anchors (honest)
    python -m egxfc.eval_overlay --anchors bundle   # persisted (Oct in-sample)
    python -m egxfc.eval_overlay --months 2026-08 2026-10

FROZEN PROTOCOL (declared 2026-11-03, before any number was produced):

  months        2025-09 .. last complete month (full 15m history on disk);
                selection = 2025-09..2026-07, LOCKED confirmation =
                2026-08..last - terms are picked on selection, confirmed
                once on locked
  anchors       walk (default): per name-month fit on rows < month with the
                registry family (same machinery as rescore_l1) - honest
                day-ahead band; --anchors bundle replays the persisted
                artifact per row (backtest-compatible, October in-sample)
  causality     beta and avg20-volume computed on daily history STRICTLY
                BEFORE the session day (mirrors live, whose daily parquet
                ends at the previous session until the EOD refresh);
                checkpoint cutoffs are zoneinfo-driven feed labels
                (egxfc.overlay.feed_cutoff) - DST-correct on every date
  forward       h_fwd / l_fwd = max/min of 15m bars with stamp > feed label
  three-way     new_high = h_fwd > carved_h (a high printed AFTER ck);
                already-in rows (final extreme already printed) have
                coverage guaranteed by containment - their FRACTION is
                reported, they never enter the honest coverage numbers
  primary       pooled Winkler interval score at alpha = 0.2 (80% nominal)
                on the two forward extremes vs band [rem_l, rem_h],
                ATR-normalised (same units as rescore_l1), day-level
                pooling: mean over names x checkpoints within a day, then
                mean over days; deltas compared with the same day-block
                bootstrap (block=5, B=1000, seed=20261103)
  ablation      constants are FROZEN (pace 0.8/1.5, exhaustion 3/80%/0.7) -
                only terms are switched, never retuned:
                  B0_anchor  frozen day-ahead band, no update (reference B)
                  A0_contain widen anchor only to contain carve+live (A)
                  A1_beta    + beta shift, belief pinned to 1, no exhaustion
                  A2_pace    + pace/belief (production belief), no exhaustion
                  A3_full    + exhaustion = production overlay
  ship bar      on LOCKED months A3 must beat BOTH A0 (reference A) and
                B0 (reference B) on pooled Winkler (ci_hi < 0) to ship
                as-is; otherwise P5 evaluates the conformal replacement
  guards        name-month skipped when train < 300 rows (MIN_TRAIN) or
                scored days < MIN_DAYS, or the name has no 15m/index bars
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from . import config as C
from . import enrich as E
from . import overlay as O
from . import train as T
from .rescore_l1 import (ALPHA, BLOCK, B_BOOT, MIN_TRAIN, SEED, SUPPORT,
                         TARGETS, block_boot_ci, complete_months,
                         interval_score, load_families, prep)

EVAL_FIRST = "2025-09"       # 15m history starts 2025-09-01
LOCKED_FIRST = "2026-08"     # locked confirmation months
MIN_DAYS = 10                # name-month needs >= 10 scored days
VARIANTS = ("B0_anchor", "A0_contain", "A1_beta", "A2_pace", "A3_full")
FLAGSETS = {
    "A0_contain": dict(shift=False),
    "A1_beta": dict(shift=True, belief_on=False, exhaust_on=False),
    "A2_pace": dict(shift=True, belief_on=True, exhaust_on=False),
    "A3_full": dict(shift=True, belief_on=True, exhaust_on=True),
}
OUT_DIR = C.ROOT / "output" / "intraday_prediction"


# ------------------------------------------------------------------ data
def load_bframe(sym: str) -> pd.DataFrame | None:
    """date/close/volume daily frame for beta + avg20_volume (caller slices)."""
    p = C.DATA / f"{sym}_daily.parquet"
    if not p.exists():
        return None
    raw = pd.read_parquet(p)
    raw.index = pd.to_datetime(raw.index)
    raw = raw.reset_index()
    dtc = "datetime" if "datetime" in raw.columns else raw.columns[0]
    out = raw[[dtc, "close", "volume"]].rename(columns={dtc: "date"})
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    return out.dropna().drop_duplicates("date").sort_values("date").reset_index(drop=True)


def load_15m(sym: str) -> pd.DataFrame | None:
    p = C.DATA / f"{sym}_15m.parquet"
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p).reset_index()
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------- frozen split + score
def forward_split(carved_h: float, carved_l: float, h_fwd: float,
                  l_fwd: float, eps: float = 1e-9) -> tuple[bool, bool]:
    """Was a NEW extreme printed after the checkpoint? (else already-in)."""
    return (bool(h_fwd > carved_h + eps), bool(l_fwd < carved_l - eps))


def forward_winkler(h_fwd: float, l_fwd: float, rem_l: float, rem_h: float,
                    atr: float) -> float:
    """(score(h_fwd) + score(l_fwd)) / 2 against the band, ATR units."""
    s = (interval_score(h_fwd, rem_l, rem_h)
         + interval_score(l_fwd, rem_l, rem_h)) / 2.0
    return float(s / atr)


def _anchors_walk(trd, ted, feats, fam, Xtr, Xte):
    yh = {}
    for tgt in TARGETS:
        est = T.candidates()[fam.get(tgt, "ridge")]
        est.fit(Xtr, trd[tgt].to_numpy(float))
        yh[tgt] = est.predict(Xte)
    return yh


def _anchors_bundle(ted, bundle):
    yh = {}
    feats = list(bundle["features"])
    for tgt in TARGETS:
        preds = []
        for _, row in ted.iterrows():
            fr = row.to_frame().T.reindex(columns=feats).ffill().fillna(0)
            preds.append(float(bundle["models"][tgt].predict(fr.values)[0]))
        yh[tgt] = np.asarray(preds)
    return yh


# ------------------------------------------------------------------- run
def run(months: list[str], names: list[str], anchors: str) -> pd.DataFrame:
    fams = load_families()
    rows: list[dict] = []
    skipped: list[dict] = []
    for si, sym in enumerate(names, 1):
        pp = prep(sym, fams)
        if pp is None:
            skipped.append({"month": "*", "sym": sym, "why": "prep"})
            continue
        mg, feats, fam = pp
        bframe = load_bframe(sym)
        s15 = load_15m(sym)
        try:
            key = E.home_key(sym)
        except Exception:  # noqa: BLE001
            key = None
        i15 = load_15m(key) if key else None
        if bframe is None or s15 is None or i15 is None:
            skipped.append({"month": "*", "sym": sym, "why": "15m/index/daily"})
            continue
        bundle = None
        if anchors == "bundle":
            import joblib
            try:
                bundle = joblib.load(C.ROOT / "models" / sym / "daily.pkl")
            except Exception:  # noqa: BLE001
                skipped.append({"month": "*", "sym": sym, "why": "bundle"})
                continue
        n_month = 0
        for mo in months:
            tr, te = T._slice(mg, "daily", mo)
            trd = tr.dropna(subset=feats + SUPPORT)
            ted = te.dropna(subset=feats + SUPPORT)
            if len(trd) < MIN_TRAIN or len(ted) < MIN_DAYS:
                skipped.append({"month": mo, "sym": sym,
                                "why": f"train={len(trd)} days={len(ted)}"})
                continue
            try:
                if anchors == "walk":
                    Xtr = trd[feats].to_numpy(float)
                    Xte = ted[feats].to_numpy(float)
                    yh = _anchors_walk(trd, ted, feats, fam, Xtr, Xte)
                else:
                    yh = _anchors_bundle(ted, bundle)
            except Exception as e:  # noqa: BLE001
                skipped.append({"month": mo, "sym": sym,
                                "why": f"anchor {type(e).__name__}"})
                continue
            c = ted["raw_close"].to_numpy(float)
            ph_all = c * (1.0 + yh["t_high"])
            pl_all = c * (1.0 + yh["t_low"])
            atr_all = ted["atr"].to_numpy(float)
            dates = pd.to_datetime(ted["date"]).to_numpy()
            n_month += 1
            for i in range(len(ted)):
                day = pd.Timestamp(dates[i]).date()
                ph, pl, atr = float(ph_all[i]), float(pl_all[i]), float(atr_all[i])
                if not np.isfinite(atr) or atr <= 0:
                    continue
                db = O._bars_on_day(s15, day)
                if len(db) < 4:
                    continue
                dly_lt = bframe[bframe["date"] < pd.Timestamp(day)]
                if len(dly_lt) < 30:
                    continue
                # beta is per-day: prime the cache once per (sym, day) so all
                # checkpoints/variants share the same causal beta
                O._BETA_CACHE.pop(sym, None)
                for ck in O.CKPTS:
                    eff = O.feed_cutoff(day, ck)
                    fut = db[db["_dt"].dt.strftime("%H:%M") > eff]
                    if not len(fut):
                        continue
                    o3 = O.day_overlay(sym, ck, day, s15, i15, dly_lt, ph, pl)
                    if o3 is None:
                        continue
                    o0 = O.day_overlay(sym, ck, day, s15, i15, dly_lt, ph, pl,
                                       **FLAGSETS["A0_contain"])
                    o1 = O.day_overlay(sym, ck, day, s15, i15, dly_lt, ph, pl,
                                       **FLAGSETS["A1_beta"])
                    o2 = O.day_overlay(sym, ck, day, s15, i15, dly_lt, ph, pl,
                                       **FLAGSETS["A2_pace"])
                    if o0 is None or o1 is None or o2 is None:
                        continue
                    h_fwd = float(fut["high"].max())
                    l_fwd = float(fut["low"].min())
                    ch, cl = o3["carved"]
                    new_high, new_low = forward_split(ch, cl, h_fwd, l_fwd)
                    bands = {"B0_anchor": (pl, ph),
                             "A0_contain": (o0["rem_l"], o0["rem_h"]),
                             "A1_beta": (o1["rem_l"], o1["rem_h"]),
                             "A2_pace": (o2["rem_l"], o2["rem_h"]),
                             "A3_full": (o3["rem_l"], o3["rem_h"])}
                    exh = {"B0_anchor": 0,
                           "A0_contain": int(o0["exhausted"]),
                           "A1_beta": int(o1["exhausted"]),
                           "A2_pace": int(o2["exhausted"]),
                           "A3_full": int(o3["exhausted"])}
                    aw = max(1e-9, ph - pl)
                    for v, (rl_, rh_) in bands.items():
                        rows.append({
                            "month": mo, "date": day, "sym": sym, "ck": ck,
                            "variant": v,
                            "wink": forward_winkler(h_fwd, l_fwd, rl_, rh_, atr),
                            "new_high": int(new_high),
                            "new_low": int(new_low),
                            "cov_nh": (h_fwd <= rh_) if new_high else np.nan,
                            "cov_nl": (l_fwd >= rl_) if new_low else np.nan,
                            "width": (rh_ - rl_) / aw,
                            "exh": exh[v],
                        })
        print(f"  [{si}/{len(names)}] {sym}: {n_month} months scored",
              flush=True)
    if skipped:
        sk = pd.DataFrame(skipped)
        print(f"skipped blocks: {len(sk)} "
              f"(sample: {sk.head(3).to_dict('records')})", flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------------- summaries
def _day_pivot(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(index="date", columns="variant", values="wink",
                          aggfunc="mean")


def _split_stats(df: pd.DataFrame) -> dict:
    out = {}
    for v, g in df.groupby("variant"):
        out[v] = {
            "winkler": float(g.groupby("date")["wink"].mean().mean()),
            "width": float(g["width"].mean()),
            "already_high": float(1.0 - g["new_high"].mean()),
            "already_low": float(1.0 - g["new_low"].mean()),
            "cov_new_high": (float(g["cov_nh"].mean())
                             if g["cov_nh"].notna().any() else np.nan),
            "cov_new_low": (float(g["cov_nl"].mean())
                            if g["cov_nl"].notna().any() else np.nan),
            "n_rows": int(len(g)),
            "n_new_high": int(g["new_high"].sum()),
            "n_new_low": int(g["new_low"].sum()),
            "n_exh": int(g["exh"].sum()),
        }
    return out


COMPARISONS = (("A3_full", "A0_contain"), ("A3_full", "B0_anchor"),
               ("A1_beta", "A0_contain"), ("A2_pace", "A1_beta"),
               ("A3_full", "A2_pace"))


def delta_rows(piv: pd.DataFrame, scope: str) -> list[dict]:
    out = []
    for a, b in COMPARISONS:
        if a not in piv or b not in piv:
            continue
        d = (piv[a] - piv[b]).dropna().to_numpy(float)
        if len(d) < 3:
            continue
        lo, hi, _ = block_boot_ci(d)
        out.append({"scope": scope, "cmp": f"{a} - {b}",
                    "delta": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
                    "verdict": "BEATS" if hi < 0 else
                               ("LOSES" if lo > 0 else "MIXED")})
    return out


def summarize(df: pd.DataFrame, months: list[str]) -> tuple:
    locked = [m for m in months if m >= LOCKED_FIRST]
    selection = [m for m in months if m < LOCKED_FIRST]
    scopes = [("pooled", df),
              ("selection", df[df["month"].isin(selection)]),
              ("locked", df[df["month"].isin(locked)])]
    split_rows, delta_all = [], []
    for sname, sub in scopes:
        if not len(sub):
            continue
        for ck, g in ([("all", sub)] + [(ck, sub[sub["ck"] == ck])
                                        for ck in O.CKPTS]):
            for v, st in _split_stats(g).items():
                split_rows.append({"scope": sname, "ck": ck, "variant": v, **st})
        piv = _day_pivot(sub)
        delta_all += delta_rows(piv, sname)
    # per-month winkler for the trend table
    mrows = []
    for (mo, v), g in df.groupby(["month", "variant"]):
        mrows.append({"month": mo, "variant": v,
                      "winkler": float(g.groupby("date")["wink"].mean().mean()),
                      "n_days": int(g["date"].nunique())})
    return (pd.DataFrame(split_rows), pd.DataFrame(delta_all),
            pd.DataFrame(mrows))


# --------------------------------------------------------------- selftest
def selftest() -> int:
    """Frozen P4 definitions on hand-computable fixtures (no refits)."""
    fails = 0
    print("== eval_overlay frozen-protocol selftest ==")

    def check(name, got, want, tol=1e-9):
        nonlocal fails
        try:
            ok = bool(np.allclose(np.asarray(got, float),
                                  np.asarray(want, float), atol=tol, rtol=0))
        except Exception:  # noqa: BLE001
            ok = got == want
        if ok:
            print(f"  PASS {name}")
        else:
            print(f"  FAIL {name}: got {got!r} want {want!r}")
            fails += 1

    k0 = list(O.FRAC_LEFT)[0]
    # 1. flag defaults == explicit True (refactor guard for overlay_at)
    a = O.overlay_at(k0, 100.0, 101.2, 99.1, 4000.0, 105.0, 95.0, 0.012, 0.6, 1000.0)
    b = O.overlay_at(k0, 100.0, 101.2, 99.1, 4000.0, 105.0, 95.0, 0.012, 0.6,
                     1000.0, shift=True, belief_on=True, exhaust_on=True)
    check("1 defaults == explicit", (a["rem_h"], a["rem_l"], a["w"]),
          (b["rem_h"], b["rem_l"], b["w"]))

    # 2. A0 contain-only: anchor widened only to contain carve + live
    r = O.overlay_at(k0, 100.0, 108.0, 94.0, 500.0, 105.0, 95.0, 0.01, 0.6,
                     1000.0, shift=False)
    check("2 contain carve outside", (r["rem_h"], r["rem_l"]), (108.0, 94.0))
    r = O.overlay_at(k0, 100.0, 102.0, 97.0, 500.0, 105.0, 95.0, 0.01, 0.6,
                     1000.0, shift=False)
    check("2 contain carve inside", (r["rem_h"], r["rem_l"]), (105.0, 95.0))

    # 3. three-way split flags
    check("3 new high", forward_split(101.0, 99.0, 103.0, 98.5), (True, True))
    check("3 already in", forward_split(101.0, 99.0, 100.5, 99.5), (False, False))

    # 4. forward Winkler: band [95,102], h=103 l=96, atr=2
    #    score(h)=7+10*1=17, score(l)=7 -> (17+7)/2=12 -> /2 atr = 6
    check("4 winkler", forward_winkler(103.0, 96.0, 95.0, 102.0, 2.0), 6.0)

    # 5. day-level pooling shape: 3 days x 2 variants (>=3 for bootstrap)
    df = pd.DataFrame([
        {"date": "2026-01-05", "variant": "A3_full", "wink": 4.0},
        {"date": "2026-01-05", "variant": "A0_contain", "wink": 5.0},
        {"date": "2026-01-06", "variant": "A3_full", "wink": 6.0},
        {"date": "2026-01-06", "variant": "A0_contain", "wink": 6.0},
        {"date": "2026-01-07", "variant": "A3_full", "wink": 7.0},
        {"date": "2026-01-07", "variant": "A0_contain", "wink": 7.0},
    ])
    piv = _day_pivot(df)
    check("5 pooled wink A3", float(piv["A3_full"].mean()), 17.0 / 3.0)
    d = delta_rows(piv, "t")
    cmps = {r["cmp"]: r for r in d}
    check("5 delta A3-A0", cmps["A3_full - A0_contain"]["delta"], -1.0 / 3.0)

    print(f"eval_overlay selftest: {'PASS' if fails == 0 else f'FAIL({fails})'}")
    return fails


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--months", nargs=2, default=None,
                    metavar=("START", "END"))
    ap.add_argument("--names", nargs="+", default=None)
    ap.add_argument("--anchors", choices=("walk", "bundle"), default="walk")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    months = [m for m in complete_months() if m >= EVAL_FIRST]
    if a.months:
        months = [m for m in months if a.months[0] <= m <= a.months[1]]
    names = a.names or list(C.DEFAULT_UNIVERSE)
    print(f"eval_overlay: {len(months)} months ({months[0]}..{months[-1]}), "
          f"{len(names)} names, anchors={a.anchors}", flush=True)
    df = run(months, names, a.anchors)
    if not len(df):
        print("no rows scored")
        return 1
    split_tbl, deltas, mrows = summarize(df, months)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = a.anchors
    df.to_parquet(OUT_DIR / f"eval_overlay_rows_{tag}.parquet", index=False)
    split_tbl.to_csv(OUT_DIR / f"eval_overlay_split_{tag}.csv", index=False)
    deltas.to_csv(OUT_DIR / f"eval_overlay_delta_{tag}.csv", index=False)
    mrows.to_csv(OUT_DIR / f"eval_overlay_month_{tag}.csv", index=False)

    md = ["# Layer-2 overlay eval (tautology-free, P4)",
          "",
          f"Frozen protocol per `egxfc/eval_overlay.py` · anchors={a.anchors} · "
          f"{len(months)} months ({months[0]}..{months[-1]}) · alpha={ALPHA} · "
          f"day-block bootstrap (block={BLOCK}, B={B_BOOT}, seed={SEED})",
          "",
          "## Primary: pooled Winkler @80 (day-level) + split diagnostics",
          "",
          "| scope | ck | variant | winkler | width | already-h | already-l "
          "| cov_new-h | cov_new-l | n_new-h | n_new-l | n_exh |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in split_tbl.iterrows():
        f = (lambda x: "nan" if pd.isna(x) else f"{x:.3f}")
        md.append(f"| {r['scope']} | {r['ck']} | {r['variant']} | "
                  f"{r['winkler']:.3f} | {r['width']:.3f} | "
                  f"{r['already_high']:.0%} | {r['already_low']:.0%} | "
                  f"{f(r['cov_new_high'])} | {f(r['cov_new_low'])} | "
                  f"{r['n_new_high']} | {r['n_new_low']} | {r['n_exh']} |")
    md += ["", "## Deltas (day-block bootstrap 95% CI)", "",
           "| scope | comparison | delta | 95% CI | verdict |",
           "|---|---|---|---|---|"]
    for _, r in deltas.iterrows():
        md.append(f"| {r['scope']} | {r['cmp']} | {r['delta']:.3f} | "
                  f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] | {r['verdict']} |")
    # ship bar on locked months
    lk = deltas[deltas["scope"] == "locked"] if len(deltas) else deltas
    bar = {}
    if len(lk):
        for _, r in lk.iterrows():
            bar[r["cmp"]] = r["verdict"]
    ship = (bar.get("A3_full - A0_contain") == "BEATS"
            and bar.get("A3_full - B0_anchor") == "BEATS")
    verdict_txt = ("SHIP as-is" if ship else
                   "DOES NOT SHIP - P5 evaluates conformal replacement")
    md += ["", "## Ship bar (LOCKED months, A3 vs A0 and vs B0)", "",
           f"- A3 vs A0: **{bar.get('A3_full - A0_contain', 'n/a')}**",
           f"- A3 vs B0: **{bar.get('A3_full - B0_anchor', 'n/a')}**",
           f"- verdict: **{verdict_txt}**", ""]
    (OUT_DIR / f"eval_overlay_{tag}.md").write_text("\n".join(md),
                                                    encoding="utf-8")
    print("\n".join(md[6:]))
    print(f"\nwrote {OUT_DIR / f'eval_overlay_{tag}.md'} (+ split/delta/month csv, rows parquet)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
