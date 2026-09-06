#!/usr/bin/env python3
"""
Standalone Optimization Runner
Runs grid search parameter tuning on historical signals
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_params, load_enhancement_config
from src.store.signal_store import load_store
from src.enhancement.param_tuner import run_grid_search, generate_quality_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("optimization")

def main():
    parser = argparse.ArgumentParser(description="EGX Parameter Optimization")
    parser.add_argument("--test-days", type=int, default=60,
                        help="Test window in days (default: 60)")
    parser.add_argument("--min-trades", type=int, default=30,
                        help="Minimum trades required (default: 30)")
    parser.add_argument("--metric", choices=["win_rate", "expectancy", "sharpe"],
                        default="win_rate", help="Optimization metric")
    parser.add_argument("--apply", action="store_true",
                        help="Apply best params to config/params.yaml")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("EGX PARAMETER OPTIMIZATION")
    log.info("=" * 60)

    # Load signal store
    store = load_store()
    if store.empty:
        log.error("No signals in store. Run daily analysis first.")
        sys.exit(1)

    log.info("Loaded %d signals from %s to %s",
             len(store), store["run_date"].min(), store["run_date"].max())

    # Run grid search
    result = run_grid_search(
        signal_store=store,
        test_window_days=args.test_days,
        min_trades=args.min_trades,
        metric=args.metric,
    )

    if result:
        log.info("=" * 60)
        log.info("BEST PARAMETERS FOUND:")
        for k, v in result["params"].items():
            log.info("  %s = %s", k, v)
        log.info("Metric (%s): %.4f", args.metric, result["score"])
        log.info("Trades in test window: %d", result["n_trades"])

        if args.apply:
            # Update params.yaml
            import yaml
            params_path = Path("config/params.yaml")
            with open(params_path) as f:
                params = yaml.safe_load(f)

            for key, value in result["params"].items():
                if key in params.get("rsi", {}).get("ranging", {}):
                    params["rsi"]["ranging"][key] = value
                elif key in params.get("volume", {}):
                    params["volume"][key] = value
                elif key in params.get("trade", {}):
                    params["trade"][key] = value

            # Increment version
            current_version = params.get("version", "v1")
            version_num = int(current_version[1:]) + 1 if current_version.startswith("v") else 2
            params["version"] = f"v{version_num}"
            params["updated"] = str(pd.Timestamp.now().date())

            with open(params_path, "w") as f:
                yaml.dump(params, f, default_flow_style=False)

            log.info("Applied params and updated version to %s", params["version"])
    else:
        log.warning("No improvement found or insufficient data")

    # Quality report
    log.info("=" * 60)
    log.info("QUALITY REPORT:")
    report = generate_quality_report(store)
    for k, v in report.items():
        if k != "alerts":
            log.info("  %s: %s", k, v)
    if report.get("alerts"):
        for alert in report["alerts"]:
            log.warning("  ALERT: %s", alert)

    log.info("=" * 60)
    log.info("OPTIMIZATION COMPLETE")

if __name__ == "__main__":
    main()