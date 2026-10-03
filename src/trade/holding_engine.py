"""
Personalized Holding Period Engine
Uses historical outcomes, ATR, volatility, and regime to set
individual holding period recommendations per ticker.
"""
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def compute_personalized_hold(signal_store: pd.DataFrame, ticker: str,
                               atr_pct: float = None, regime: str = "unknown",
                               rsi: float = None, adx: float = None) -> Dict:
    """Compute personalized holding period for a ticker.

    Factors:
    1. Historical hold days (if trades exist for this ticker)
    2. ATR volatility (higher volatility = shorter holds)
    3. Regime (trending = longer, ranging = shorter)
    4. RSI level (overbought = shorter, oversold = longer)

    Returns:
        - hold_days_min: Minimum holding period
        - hold_days_max: Maximum holding period
        - hold_days_target: Target holding period
        - exit_strategy: Recommended exit approach
        - confidence: How confident we are (0-100)
    """
    # Default values
    base_min = 5
    base_max = 14
    base_target = 10

    # Factor 1: Historical hold days for this ticker
    historical_factor = 1.0
    hist_confidence = 0
    if signal_store is not None and not signal_store.empty:
        ticker_trades = signal_store[
            (signal_store["ticker"] == ticker) &
            (signal_store["outcome"].notna()) &
            (~signal_store["outcome"].isin(["pending", "error", "no_data", "invalid"]))
        ]

        if len(ticker_trades) >= 2:
            # Use TP hold days as target (when trade succeeds)
            tp_trades = ticker_trades[ticker_trades["outcome"].isin(["tp1_hit", "tp2_hit", "tp3_hit"])]
            sl_trades = ticker_trades[ticker_trades["outcome"] == "stop_loss"]

            if len(tp_trades) > 0:
                tp_avg_hold = tp_trades["hold_days"].mean()
                # Scale based on historical TP hold
                if tp_avg_hold > 0:
                    historical_factor = tp_avg_hold / base_target
                    hist_confidence = min(80, len(tp_trades) * 20)

            if len(sl_trades) > 0:
                sl_avg_hold = sl_trades["hold_days"].mean()
                # If SL tends to happen quickly, tighten holds
                if sl_avg_hold < 7:
                    base_min = max(3, base_min - 2)
                    base_max = max(7, base_max - 3)

    # Factor 2: ATR volatility
    vol_factor = 1.0
    if atr_pct is not None:
        if atr_pct > 4:
            vol_factor = 0.7  # High volatility = shorter holds
        elif atr_pct > 2.5:
            vol_factor = 0.85
        elif atr_pct < 1.5:
            vol_factor = 1.3  # Low volatility = longer holds
        elif atr_pct < 1.0:
            vol_factor = 1.5

    # Factor 3: Regime
    regime_factor = 1.0
    if regime == "trending_up" or regime == "trending_down":
        regime_factor = 1.3  # Trending = ride longer
    elif regime == "ranging":
        regime_factor = 0.7  # Ranging = take profits quicker
    elif regime == "transitioning":
        regime_factor = 0.9

    # Factor 4: RSI
    rsi_factor = 1.0
    if rsi is not None:
        if rsi > 75:
            rsi_factor = 0.7  # Overbought = exit sooner
        elif rsi < 30:
            rsi_factor = 1.3  # Oversold = hold longer (reversal expected)

    # Compute personalized values
    combined_factor = historical_factor * vol_factor * regime_factor * rsi_factor
    combined_factor = max(0.5, min(2.0, combined_factor))  # Cap the adjustment

    hold_min = max(2, int(base_min * combined_factor))
    hold_max = max(hold_min + 2, int(base_max * combined_factor))
    hold_target = int(base_target * combined_factor)

    # Exit strategy based on factors
    if regime in ("trending_up", "trending_down"):
        exit_strategy = "trail_stop"  # Use trailing stop in trends
    elif rsi is not None and rsi > 75:
        exit_strategy = "take_profit"  # Take profits when overbought
    elif vol_factor < 0.8:
        exit_strategy = "quick_exit"  # High volatility = exit fast
    else:
        exit_strategy = "time_exit"  # Default: time-based exit

    # Confidence score
    confidence = hist_confidence  # Base from historical data
    if atr_pct is not None:
        confidence += 20  # We have volatility data
    if regime != "unknown":
        confidence += 15
    confidence = min(100, confidence)

    return {
        "hold_days_min": hold_min,
        "hold_days_max": hold_max,
        "hold_days_target": hold_target,
        "exit_strategy": exit_strategy,
        "confidence": confidence,
        "factors": {
            "historical": round(historical_factor, 2),
            "volatility": round(vol_factor, 2),
            "regime": round(regime_factor, 2),
            "rsi": round(rsi_factor, 2),
        },
    }


def format_hold_label(hold_min: int, hold_max: int, exit_strategy: str) -> str:
    """Format holding period label for display."""
    strategy_labels = {
        "trail_stop": "Trail Stop",
        "take_profit": "Take Profit",
        "quick_exit": "Quick Exit",
        "time_exit": "Time Exit",
    }
    strategy = strategy_labels.get(exit_strategy, "Time Exit")
    return f"{hold_min}-{hold_max} days ({strategy})"
