"""
Enhancement Engine for EGX Signals
Fast grid search parameter tuning and quality monitoring
"""
import itertools
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, timedelta
from typing import Dict, List, Optional, Any
from src.config import load_params, load_enhancement_config
from src.store.signal_store import load_store, load_latest_signals

log = logging.getLogger(__name__)

params = load_params()
enhance_cfg = load_enhancement_config()
tuning_cfg = enhance_cfg["param_tuning"]
quality_cfg = enhance_cfg["quality_monitoring"]

# ─────────────────────────────────────────────────────────────
# Parameter Grid Search
# ─────────────────────────────────────────────────────────────
def run_grid_search(signal_store: pd.DataFrame, test_window_days: int = 60,
                    min_trades: int = 30, metric: str = "win_rate") -> Optional[Dict]:
    """Run fast grid search over parameter space."""
    if signal_store.empty:
        return None

    # Get recent signals
    cutoff = signal_store["run_date"].max() - timedelta(days=test_window_days)
    recent = signal_store[signal_store["run_date"] >= cutoff].copy()

    if len(recent) < min_trades:
        log.warning("Insufficient recent signals (%d < %d)", len(recent), min_trades)
        return None

    param_grid = tuning_cfg["param_grid"]
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    best_score = -1
    best_params = None
    best_n_trades = 0

    total_combos = np.prod([len(v) for v in param_values])
    log.info("Grid search: %d combinations over %d days, %d signals",
             total_combos, test_window_days, len(recent))

    for i, combo in enumerate(itertools.product(*param_values)):
        test_params = dict(zip(param_names, combo))

        # Re-score recent signals with test parameters
        scored = rescore_signals(recent, test_params)

        # Filter to Buy signals
        buys = scored[scored["recommendation"] == "Buy"]
        if len(buys) < min_trades:
            continue

        # Compute metric
        if metric == "win_rate":
            score = (buys["outcome"] == "tp1_hit").mean() if "outcome" in buys.columns else 0
        elif metric == "expectancy":
            score = buys["r_multiple"].mean() if "r_multiple" in buys.columns else 0
        elif metric == "sharpe":
            returns = buys["r_multiple"] if "r_multiple" in buys.columns else pd.Series([0])
            score = returns.mean() / returns.std() if returns.std() > 0 else 0
        else:
            score = 0

        if score > best_score:
            best_score = score
            best_params = test_params
            best_n_trades = len(buys)

        if i % 100 == 0 and i > 0:
            log.debug("  Completed %d/%d combos, best %s: %.4f", i, total_combos, metric, best_score)

    if best_params:
        log.info("Grid search complete: best %s = %.4f with %d trades, params: %s",
                 metric, best_score, best_n_trades, best_params)
        return {
            "params": best_params,
            "score": float(best_score),
            "n_trades": best_n_trades,
            "metric": metric,
            "test_window_days": test_window_days,
        }

    return None

def rescore_signals(signals: pd.DataFrame, test_params: Dict) -> pd.DataFrame:
    """Re-score signals with test parameters (simplified - only RSI/volume thresholds)."""
    # This is a simplified re-scoring for speed
    # In practice, you'd re-run the full scoring with test params
    # For now, we approximate by adjusting the score based on parameter changes

    scored = signals.copy()

    # For each signal, check if it would still be Buy with new thresholds
    # This is a placeholder - real implementation would need the raw indicator values
    # For now, we just return the signals as-is
    # TODO: Implement proper re-scoring when raw indicators are stored

    return scored

# ─────────────────────────────────────────────────────────────
# Quality Monitoring
# ─────────────────────────────────────────────────────────────
def generate_quality_report(signal_store: pd.DataFrame) -> Dict[str, Any]:
    """Generate quality metrics report."""
    if signal_store.empty:
        return {"error": "No signals in store"}

    report = {
        "generated_at": str(date.today()),
        "total_signals": len(signal_store),
        "date_range": {
            "start": str(signal_store["run_date"].min()),
            "end": str(signal_store["run_date"].max()),
        },
    }

    # Recommendation distribution
    report["recommendations"] = signal_store["recommendation"].value_counts().to_dict()

    # Score calibration
    if "score" in signal_store.columns:
        score_bins = pd.cut(signal_store["score"], bins=[0, 40, 50, 60, 70, 80, 90, 100])
        calibration = signal_store.groupby(score_bins)["recommendation"].apply(
            lambda x: (x == "Buy").mean() if len(x) > 0 else 0
        )
        report["score_calibration"] = {str(k): float(v) for k, v in calibration.items()}

    # Win rate by regime
    if "regime" in signal_store.columns:
        regime_perf = signal_store.groupby("regime")["recommendation"].apply(
            lambda x: (x == "Buy").mean() if len(x) > 0 else 0
        )
        report["regime_performance"] = regime_perf.to_dict()

    # ML performance
    if "ml_confidence" in signal_store.columns:
        ml_buys = signal_store[signal_store["ml_signal"] == "Buy"]
        if len(ml_buys) > 0:
            report["ml_buy_rate"] = float((ml_buys["recommendation"] == "Buy").mean())
            report["ml_avg_confidence"] = float(ml_buys["ml_confidence"].mean())

    # TV fetch success rate
    if "ta_source" in signal_store.columns:
        tv_rate = (signal_store["ta_source"] == "TradingView").mean()
        report["tv_fetch_rate"] = float(tv_rate)

    # Alerts
    alerts = []
    if quality_cfg.get("enabled", True):
        if "score_calibration" in report:
            # Check if high-score stocks actually win more
            high_score_buys = calibration.get("(70, 80]", 0) if "calibration" in locals() else 0
            low_score_buys = calibration.get("(40, 50]", 0) if "calibration" in locals() else 0
            if high_score_buys - low_score_buys < quality_cfg.get("score_calibration_slope_below", 0.1):
                alerts.append(f"Score calibration weak: high-score win rate only {high_score_buys:.1%} vs low-score {low_score_buys:.1%}")

        if report.get("tv_fetch_rate", 1) < 1 - quality_cfg.get("tv_failure_rate_above", 0.2):
            alerts.append(f"TV fetch rate low: {report['tv_fetch_rate']:.1%}")

    report["alerts"] = alerts

    return report

def check_param_stability(signal_store: pd.DataFrame, lookback_days: int = 90) -> Dict[str, Any]:
    """Check if optimal parameters are stable over time."""
    if signal_store.empty:
        return {}

    cutoff = signal_store["run_date"].max() - timedelta(days=lookback_days)
    recent = signal_store[signal_store["run_date"] >= cutoff].copy()

    if recent.empty:
        return {}

    # Group by params_version and check performance
    version_perf = recent.groupby("params_version").agg(
        n_signals=("ticker", "count"),
        avg_score=("score", "mean"),
        buy_rate=("recommendation", lambda x: (x == "Buy").mean()),
    ).reset_index()

    return {
        "versions_tested": len(version_perf),
        "version_performance": version_perf.to_dict("records"),
        "current_version": params.get("version", "unknown"),
    }

# ─────────────────────────────────────────────────────────────
# ML Retraining
# ─────────────────────────────────────────────────────────────
def should_retrain_ml(signal_store: pd.DataFrame) -> bool:
    """Check if ML model should be retrained."""
    ml_cfg = enhance_cfg["ml_retrain"]
    if not ml_cfg.get("enabled", True):
        return False

    min_rows = ml_cfg.get("min_new_rows", 50)
    # Check if we have enough new signals since last retrain
    # This would need a persistent state - simplified for now
    return len(signal_store) >= min_rows

# ─────────────────────────────────────────────────────────────
# Main Enhancement Function
# ─────────────────────────────────────────────────────────────
def run_weekly_enhancement() -> Dict[str, Any]:
    """Run the weekly enhancement routine."""
    log.info("Starting weekly enhancement...")

    signal_store = load_store()
    results = {"timestamp": str(date.today()), "actions": []}

    # 1. Quality report
    if quality_cfg.get("enabled", True):
        report = generate_quality_report(signal_store)
        results["quality_report"] = report
        if report.get("alerts"):
            for alert in report["alerts"]:
                log.warning("QUALITY ALERT: %s", alert)
                results["actions"].append(f"ALERT: {alert}")

    # 2. Parameter stability check
    stability = check_param_stability(signal_store)
    results["param_stability"] = stability

    # 3. Grid search (monthly, but we check here)
    # In practice, this runs monthly via separate workflow
    if tuning_cfg.get("enabled", True):
        # Check if it's time for monthly tuning
        # For now, skip - runs in monthly workflow
        pass

    log.info("Weekly enhancement complete")
    return results