#!/usr/bin/env python3
"""
Monthly Parameter Tuning Runner
Called by the monthly-tune.yml workflow
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from src.store.signal_store import load_store
from src.enhancement.param_tuner import run_grid_search
from src.config import load_enhancement_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("monthly_tune")

store = load_store()
if store.empty:
    print("No signals in store yet")
    sys.exit(0)

log.info("Loaded %d signals from store", len(store))

enhance_cfg = load_enhancement_config()
tuning_cfg = enhance_cfg["param_tuning"]

result = run_grid_search(
    signal_store=store,
    test_window_days=tuning_cfg.get("test_window_days", 60),
    min_trades=tuning_cfg.get("min_trades", 30),
    metric=tuning_cfg.get("metric", "win_rate"),
)

if result:
    print("Grid search result:", result)

    # Update params.yaml with best params
    params_path = Path("config/params.yaml")
    with open(params_path) as f:
        params = yaml.safe_load(f)

    # Update relevant params
    for key, value in result["params"].items():
        if key in params.get("rsi", {}).get("ranging", {}):
            params["rsi"]["ranging"][key] = value
        elif key in params.get("volume", {}):
            params["volume"][key] = value
        elif key in params.get("trade", {}):
            params["trade"][key] = value

    params["version"] = f"v{int(params.get('version', 'v1')[1:]) + 1}"
    params["updated"] = str(pd.Timestamp.now().date())

    with open(params_path, "w") as f:
        yaml.dump(params, f)

    print(f"Updated params to version {params['version']}")
else:
    print("Grid search did not find improvement")
