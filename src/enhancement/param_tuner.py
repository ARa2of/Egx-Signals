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
    """Run fast grid search over parameter space.

    Only evaluates signals that have actual outcome data (not pending).
    """
    if signal_store.empty:
        return None

    # Get recent signals with actual outcomes
    cutoff = signal_store["run_date"].max() - timedelta(days=test_window_days)
    recent = signal_store[signal_store["run_date"] >= cutoff].copy()

    # Filter to signals with real outcomes (not pending/error/no_data)
    has_outcome = recent["outcome"].notna() & ~recent["outcome"].isin(["pending", "error", "no_data", "invalid"])
    recent = recent[has_outcome].copy()

    if len(recent) < min_trades:
        log.warning("Insufficient recent signals with outcomes (%d < %d)", len(recent), min_trades)
        return None

    param_grid = tuning_cfg["param_grid"]
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    best_score = -1
    best_params = None
    best_n_trades = 0

    total_combos = np.prod([len(v) for v in param_values])
    log.info("Grid search: %d combinations over %d days, %d signals with outcomes",
             total_combos, test_window_days, len(recent))

    for i, combo in enumerate(itertools.product(*param_values)):
        test_params = dict(zip(param_names, combo))

        # Re-score recent signals with test parameters
        scored = rescore_signals(recent, test_params)

        # Filter to Buy signals
        buys = scored[scored["recommendation"] == "Buy"]
        if len(buys) < min_trades:
            continue

        # Compute metric from real outcomes
        if metric == "win_rate":
            score = (buys["outcome"] == "tp1_hit").mean()
        elif metric == "expectancy":
            score = buys["r_multiple"].mean()
        elif metric == "sharpe":
            returns = buys["r_multiple"]
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
    """Re-score signals with test parameters using stored indicator values.

    The grid search only tests: rsi_oversold, rsi_healthy_high, volume_spike,
    near_support_pct, stop_loss_atr. We re-run the RSI and volume scoring
    functions with the new thresholds, then recompute total score.
    """
    from src.signals.scoring import score_rsi, score_volume, compute_base_score
    from src.config import load_params
    scored = signals.copy()
    base_params = load_params()
    weights = base_params["weights"]

    # Extract test parameters with fallbacks
    new_oversold = test_params.get("rsi_oversold",
                                   base_params["rsi"]["ranging"]["oversold"])
    new_healthy_high = test_params.get("rsi_healthy_high",
                                       base_params["rsi"]["ranging"]["healthy_high"])
    new_spike = test_params.get("volume_spike",
                                base_params["volume"]["spike_multiplier"])

    # Get current parameter values for comparison
    old_oversold = base_params["rsi"]["ranging"]["oversold"]
    old_healthy_high = base_params["rsi"]["ranging"]["healthy_high"]
    old_spike = base_params["volume"]["spike_multiplier"]

    for idx, row in scored.iterrows():
        rsi_val = row.get("rsi")
        regime = row.get("regime", "ranging")
        adx_val = row.get("adx")

        # Re-score RSI with new thresholds
        new_rsi_score, _ = score_rsi_custom(rsi_val, adx_val, regime,
                                            new_oversold, new_healthy_high)
        # score_rsi is stored as weighted value (0-12), divide by weight to get raw
        old_rsi_raw = row.get("score_rsi", 0) / weights["rsi"] if weights["rsi"] else 0

        # Re-score volume with new spike multiplier
        old_vol_score = row.get("score_volume", 0)
        # Approximate vol_multiplier from score (inverse of score_volume logic)
        approx_vol_mult = 1.0 + (old_vol_score / weights["volume"]) * 3.0
        new_vol_score, _ = score_volume_custom(approx_vol_mult, new_spike)
        # Scale to weight
        new_vol_pts = new_vol_score * weights["volume"]

        # Compute new total score using raw values * weight
        delta = (new_rsi_score - old_rsi_raw) * weights["rsi"] + \
                (new_vol_pts - old_vol_score)

        new_score = row.get("score", 0) + delta
        new_score = max(0.0, min(100.0, new_score))

        scored.at[idx, "score"] = new_score
        scored.at[idx, "score_rsi"] = new_rsi_score * weights["rsi"]
        scored.at[idx, "score_volume"] = new_vol_pts

        # Update recommendation based on new score
        buy_threshold = base_params["thresholds"]["buy"]
        watch_threshold = base_params["thresholds"]["watch"]
        death_cross = row.get("death_cross", False)
        if death_cross:
            scored.at[idx, "recommendation"] = "Avoid"
        elif new_score >= buy_threshold:
            scored.at[idx, "recommendation"] = "Buy"
        elif new_score >= watch_threshold:
            scored.at[idx, "recommendation"] = "Watch"
        else:
            scored.at[idx, "recommendation"] = "Avoid"

    return scored


def score_rsi_custom(rsi, adx, regime, oversold, healthy_high):
    """RSI scoring with custom thresholds."""
    if rsi is None:
        return 0.0, []

    adx_threshold = params.get("data", {}).get("adx_trend_threshold", 25)

    if rsi <= oversold:
        score = 0.8
    elif rsi <= 35:
        score = 0.6
    elif rsi <= healthy_high:
        score = 1.0
    elif rsi <= 75:
        strong = adx is not None and adx >= adx_threshold
        score = 0.5 if strong else 0.25
    else:
        strong = adx is not None and adx >= adx_threshold
        score = 0.35 if strong else 0.1

    return score, []


def score_volume_custom(vol_mult, spike_mult):
    """Volume scoring with custom spike multiplier."""
    if vol_mult is None:
        return 0.0, []

    if vol_mult >= 3.0:
        score = 1.0
    elif vol_mult >= 2.0:
        score = 0.6 + 0.4 * (vol_mult - 2.0)
    elif vol_mult >= spike_mult:
        score = 0.3 + 0.3 * (vol_mult - spike_mult) / (2.0 - spike_mult)
    elif vol_mult >= 1.0:
        score = 0.1 * (vol_mult - 1.0) / (spike_mult - 1.0)
    else:
        score = 0.0

    return score, []

# ─────────────────────────────────────────────────────────────
# Quality Monitoring
# ─────────────────────────────────────────────────────────────
def generate_quality_report(signal_store: pd.DataFrame) -> Dict[str, Any]:
    """Generate quality metrics report with real outcome data."""
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

    # Outcome statistics (only for signals with real outcomes)
    has_outcome = signal_store["outcome"].notna() & ~signal_store["outcome"].isin(
        ["pending", "error", "no_data", "invalid"])
    evaluated = signal_store[has_outcome]

    if len(evaluated) > 0:
        report["outcomes"] = {
            "total_evaluated": len(evaluated),
            "outcome_distribution": evaluated["outcome"].value_counts().to_dict(),
            "win_rate_tp1": float((evaluated["outcome"] == "tp1_hit").mean()),
            "win_rate_tp2": float((evaluated["outcome"] == "tp2_hit").mean()),
            "win_rate_tp3": float((evaluated["outcome"] == "tp3_hit").mean()),
            "stop_loss_rate": float((evaluated["outcome"] == "stop_loss").mean()),
            "avg_r_multiple": float(evaluated["r_multiple"].mean()),
            "avg_hold_days": float(evaluated["hold_days"].mean()),
            "avg_mfe_pct": float(evaluated["mfe_pct"].mean()),
            "avg_mae_pct": float(evaluated["mae_pct"].mean()),
            "avg_pnl_pct": float(evaluated["pnl_pct"].mean()),
        }

        # Win rate by recommendation
        buy_outcomes = evaluated[evaluated["recommendation"] == "Buy"]
        if len(buy_outcomes) > 0:
            report["outcomes"]["buy_win_rate"] = float((buy_outcomes["outcome"] == "tp1_hit").mean())
            report["outcomes"]["buy_avg_r"] = float(buy_outcomes["r_multiple"].mean())

        # Win rate by regime
        regime_outcomes = evaluated.groupby("regime").agg(
            n=("ticker", "count"),
            win_rate=("outcome", lambda x: (x == "tp1_hit").mean()),
            avg_r=("r_multiple", "mean"),
        ).to_dict("index")
        report["outcomes"]["by_regime"] = regime_outcomes

        # Win rate by params_version
        version_outcomes = evaluated.groupby("params_version").agg(
            n=("ticker", "count"),
            win_rate=("outcome", lambda x: (x == "tp1_hit").mean()),
            avg_r=("r_multiple", "mean"),
        ).to_dict("index")
        report["outcomes"]["by_params_version"] = version_outcomes
    else:
        report["outcomes"] = {"total_evaluated": 0, "note": "No signals with outcomes yet"}

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
        thresholds = quality_cfg.get("alert_thresholds", {})

        if "outcomes" in report and report["outcomes"].get("total_evaluated", 0) > 0:
            wr = report["outcomes"].get("win_rate_tp1", 0)
            if wr < thresholds.get("win_rate_below", 0.45):
                alerts.append(f"Win rate low: {wr:.1%} (threshold: {thresholds['win_rate_below']:.0%})")

            avg_r = report["outcomes"].get("avg_r_multiple", 0)
            if avg_r < 0:
                alerts.append(f"Negative expectancy: avg R-multiple = {avg_r:.2f}")

        if "score_calibration" in report:
            high_score_buys = calibration.get("(70, 80]", 0) if "calibration" in locals() else 0
            low_score_buys = calibration.get("(40, 50]", 0) if "calibration" in locals() else 0
            if high_score_buys - low_score_buys < thresholds.get("score_calibration_slope_below", 0.1):
                alerts.append(f"Score calibration weak: high-score win rate only {high_score_buys:.1%} vs low-score {low_score_buys:.1%}")

        if report.get("tv_fetch_rate", 1) < 1 - thresholds.get("tv_failure_rate_above", 0.2):
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