#!/usr/bin/env python3
"""Local checks for ML v3 enhancements."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.signals.ml import (
    compute_ml_conviction,
    _build_direction_models,
    _direction_accuracy,
    _direction_proba,
    build_features,
)

fc = {
    "Medium": {"price": 146.07, "xgb_return": -0.8, "lgbm_return": -0.5},
    "High": {"price": 152.0},
    "Low": {"price": 142.0},
}
old_style = compute_ml_conviction(
    fc, 147.5, mae_pct=2.0, direction_accuracy={"xgb": 0.51, "lgbm": 0.50}
)
new_style = compute_ml_conviction(
    fc,
    147.5,
    mae_pct=2.0,
    direction_accuracy={"xgb": 0.51, "lgbm": 0.50},
    trend_context={"bullish": True, "adx": 36},
    direction_proba={"ensemble_p_up": 0.48, "xgb_p_up": 0.47, "lgbm_p_up": 0.49},
)
print("no-trend conviction:", old_style["conviction_score"], old_style["conviction_label"])
print("bull-trend conviction:", new_style["conviction_score"], new_style["conviction_label"])
print("trend_note:", new_style.get("trend_note"))
assert new_style["conviction_score"] > old_style["conviction_score"]
assert new_style["conviction_score"] >= 20

fc2 = {
    "Medium": {"price": 158.0, "xgb_return": 4.0, "lgbm_return": 5.0},
    "High": {"price": 165.0},
    "Low": {"price": 150.0},
}
strong = compute_ml_conviction(
    fc2,
    147.5,
    direction_accuracy={"xgb": 0.62, "lgbm": 0.60},
    trend_context={"bullish": True, "adx": 40},
    direction_proba={"ensemble_p_up": 0.65},
)
print("strong bull conviction:", strong["conviction_score"], strong["conviction_label"])
assert strong["conviction_score"] >= 70

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(200, 8)), columns=[f"f{i}" for i in range(8)])
y = (X["f0"] + 0.3 * X["f1"] > 0).astype(int)
models = _build_direction_models(X.iloc[:140], y.iloc[:140])
acc = _direction_accuracy(models, X.iloc[140:], y.iloc[140:])
print("direction acc:", acc)
assert all(0.5 <= a <= 1.0 for a in acc.values())
proba = _direction_proba(models, X.iloc[[-1]])
assert "ensemble_p_up" in proba

from src.signals.chartscan import init_chartscan, is_enabled
ok = init_chartscan("weights/custom_yolov8.pt", enabled=True)
print("chartscan init:", ok, "enabled:", is_enabled())
assert ok

print("ALL ML CHECKS PASSED")
