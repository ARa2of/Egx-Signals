#!/usr/bin/env python3
"""Unit checks for scoring/regime/intraday fixes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.signals.scoring import (
    resolve_rsi_regime, score_vwap, score_volume_profile, score_rsi, compute_base_score,
)
from src.signals.technical import detect_trend_regime
from src.data.intraday_analysis import analyze_volume_profile, analyze_momentum
from src.output.agent_json import _safe, _ml_confidence
from src.config import load_params

p = load_params()
print("version", p["version"])
print("weights sum", sum(p["weights"].values()))
assert abs(sum(p["weights"].values()) - 100.0) < 1e-6

assert resolve_rsi_regime("trending_up") == "trending_up"
assert resolve_rsi_regime("trending") == "trending_up"
assert resolve_rsi_regime("trending_down") == "trending_down"
assert resolve_rsi_regime("ranging") == "ranging"
assert resolve_rsi_regime("transitioning") == "ranging"
assert resolve_rsi_regime(None) == "ranging"
assert resolve_rsi_regime("unknown") == "ranging"
print("resolve_rsi_regime OK")

s, r = score_vwap(0.068, 130.0, 138.9, adx=38.0, ema_bullish=True)
s2, r2 = score_vwap(0.068, 130.0, 138.9, adx=10.0, ema_bullish=False)
print(f"VWAP 6.8% strong={s} weak={s2}  ({r[0] if r else ''} | {r2[0] if r2 else ''})")
assert s > s2

s3, r3 = score_volume_profile(True, 131.5, 138.7, 124.4, 150.0, strong_trend=True)
s4, r4 = score_volume_profile(True, 131.5, 138.7, 124.4, 150.0, strong_trend=False)
print(f"VP breakout trend={s3} no-trend={s4}")
assert s3 > s4

s5, r5 = score_rsi(77.0, 38.0, "trending")
s6, r6 = score_rsi(77.0, 38.0, "ranging")
print(f"RSI77 ADX38 trending={s5} ranging={s6}")
assert s5 >= s6

# detect_trend_regime: bullish EMA structure + price>SMA50 + ADX strong
n = 260
idx = pd.date_range("2025-09-01", periods=n, freq="B")
close = pd.Series(np.linspace(80, 150, n) + np.random.default_rng(1).normal(0, 0.4, n), index=idx)
df = pd.DataFrame({
    "Open": close.shift(1).fillna(close.iloc[0]),
    "High": close + 1.5,
    "Low": close - 1.5,
    "Close": close,
    "Volume": np.full(n, 1e6),
})
from src.signals.technical import add_moving_averages, add_adx
df2 = add_adx(add_moving_averages(df))
regime = detect_trend_regime(df2)
print("synthetic regime:", regime["regime"], "adx", regime["adx"], "dir", regime["direction"])
assert regime["regime"] == "trending_up", regime

# Intraday bull-trend dampening: heavy volume at highs during rally
np.random.seed(0)
n = 200
idx = pd.date_range("2026-09-01", periods=n, freq="B")
close = np.linspace(100, 150, n) + np.random.normal(0, 0.5, n)
high = close + np.random.uniform(0.5, 2, n)
low = close - np.random.uniform(0.5, 2, n)
vol = np.random.uniform(1e5, 2e5, n)
vol[-40:] = 8e5
idf = pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": vol}, index=idx)
vp_n = analyze_volume_profile(idf, bull_trend=False)
vp_b = analyze_volume_profile(idf, bull_trend=True)
print(f"intraday VP skew no-trend={vp_n['volume_skew']} bull={vp_b['volume_skew']} adj={vp_b['trend_adjusted']}")
assert vp_b["trend_adjusted"] is True
assert abs(vp_b["volume_skew"]) <= abs(vp_n["volume_skew"]) + 1e-9

mom_b = analyze_momentum(idf, bull_trend=True)
mom_n = analyze_momentum(idf, bull_trend=False)
print(f"intraday momentum no-trend={mom_n['momentum_score']} bull={mom_b['momentum_score']}")
assert mom_b["momentum_score"] >= mom_n["momentum_score"]

# agent_json helpers
assert _safe(None) is None
assert _ml_confidence({"ml_confidence": 0.35}) == 35.0
assert _ml_confidence({"ML Conviction": 72}) == 72.0
print("agent_json helpers OK")

# compute_base_score end-to-end: ETEL-like inputs should hit Buy band
base = compute_base_score(
    current_price=150.0, ema20=132.6, ema50=122.3, ema200=100.0,
    macd=6.2, macd_signal=5.6, rsi=77.8, adx=35.7, regime="trending_up",
    vol_multiplier=2.5, buy_vol_multiplier=2.0, adl_trend=1e6, mfi=61.0,
    is_near_support=False, volume_confirmed=False, support=None,
    dist_vwap=0.068, vwap=141.2,
    above_poc=True, poc=131.5, va_high=138.7, va_low=124.4,
    intraday_score=None, intraday_details=None,
)
print(f"ETEL-like base score={base['raw_score']} rec={base['recommendation']}")
assert base["raw_score"] >= 60, base
assert base["recommendation"] == "Buy", base

print()
print("ALL UNIT CHECKS PASSED")
