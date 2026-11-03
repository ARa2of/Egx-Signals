"""One-off: October 2026 backtest of the live range overlay, all 21 names x 3 cks.

Coverage = realised rest-of-day high/low inside the overlay band.
Narrowing = overlay width vs day-ahead anchor width.
"""
import pathlib
import joblib
import pandas as pd
import numpy as np

import egxfc.overlay as O
from egxfc import config as C
from egxfc import enrich as E

ROOT = pathlib.Path("D:/Stocks/EGX Project")
DATA = ROOT / "data" / "intraday"

stats = {ck: {"n": 0, "cov_h": 0, "cov_l": 0, "cov_both": 0,
              "w_ratio": []} for ck in O.CKPTS}
for sym in C.DEFAULT_UNIVERSE:
    try:
        b = joblib.load(ROOT / "models" / sym / "daily.pkl")
        feats = pd.read_parquet(DATA / f"daily_features_{sym}.parquet")
    except Exception:
        continue
    feats["_t"] = pd.to_datetime(feats["date"])
    feats = feats.sort_values("_t").reset_index(drop=True)
    octf = feats[(feats["_t"].dt.year == 2026) & (feats["_t"].dt.month == 10)].reset_index(drop=True)
    try:
        s15 = pd.read_parquet(DATA / f"{sym}_15m.parquet").reset_index()
        i15 = pd.read_parquet(DATA / f"{E.home_key(sym)}_15m.parquet").reset_index()
        dly = pd.read_parquet(DATA / f"{sym}_daily.parquet").reset_index()
    except Exception:
        continue
    for i, r in octf.iterrows():
        day = r["_t"].date()
        base = float(r["Close"])
        if not np.isfinite(base):
            continue
        try:
            fr = r.to_frame().T.reindex(columns=list(b["features"])).ffill().fillna(0)
            ph = float(b["models"]["t_high"].predict(fr.values)[0])
            pl = float(b["models"]["t_low"].predict(fr.values)[0])
        except Exception:
            continue
        dtc = "datetime" if "datetime" in s15.columns else s15.columns[0]
        db = s15[pd.to_datetime(s15[dtc]).dt.date == day]
        if len(db) < 4:
            continue
        for ck in O.CKPTS:
            o = O.day_overlay(sym, ck, day, s15, i15, dly, base * (1 + ph), base * (1 + pl))
            if o is None:
                continue
            # rest-of-day bars are feed-stamped: filter by the date-specific
            # feed label (raw ck is Cairo and would drop all future bars)
            eff = O.feed_cutoff(day, ck)
            fut = db[pd.to_datetime(db[dtc]).dt.strftime("%H:%M") > eff]
            if not len(fut):
                continue
            rest_h, rest_l = float(fut["high"].max()), float(fut["low"].min())
            s = stats[ck]
            s["n"] += 1
            s["cov_h"] += rest_h <= o["rem_h"] + 1e-9
            s["cov_l"] += rest_l >= o["rem_l"] - 1e-9
            s["cov_both"] += (rest_h <= o["rem_h"] + 1e-9) and (rest_l >= o["rem_l"] - 1e-9)
            aw = (base * (1 + ph)) - (base * (1 + pl))
            s["w_ratio"].append((o["rem_h"] - o["rem_l"]) / max(1e-9, aw))

for ck in O.CKPTS:
    s = stats[ck]
    n = s["n"] or 1
    print(f"Cairo {ck}: n={s['n']} "
          f"covH={s['cov_h']/n:.0%} covL={s['cov_l']/n:.0%} "
          f"both={s['cov_both']/n:.0%} width={np.mean(s['w_ratio']):.0%} of anchor")
