"""
JSON Export for AI Agent Access
Generates structured JSON with all signal data, trade plans, and performance metrics.
Designed for easy consumption by AI agents and language models.
"""
import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
AGENT_JSON_PATH = _PROJECT_ROOT / "output" / "agent_data.json"


def _safe(val):
    """Convert numpy/pandas types to native Python for JSON."""
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        if hasattr(val, "item"):
            val = val.item()
        if isinstance(val, float) and pd.isna(val):
            return None
    except (TypeError, ValueError):
        return None
    if isinstance(val, pd.Timestamp):
        return val.isoformat()
    if isinstance(val, (date, datetime)):
        return val.isoformat()
    return val


def _ml_confidence(row: Dict):
    """Normalize ML confidence to 0-100."""
    conf = row.get("ML Conviction", row.get("ml_confidence"))
    if conf is None:
        return None
    conf = _safe(conf)
    if conf is None:
        return None
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        return None
    # ml_confidence is stored 0-1; ML Conviction is 0-100
    if conf <= 1.0:
        conf *= 100.0
    return round(conf, 1)


def _build_ticker_block(row: Dict) -> Dict:
    """Build a comprehensive ticker data block for AI consumption."""
    return {
        "ticker": row.get("Selected Stock", row.get("ticker")),
        "run_date": str(row.get("Analysis Run Date", row.get("run_date", ""))),
        "recommendation": row.get("Recommendation", row.get("recommendation")),
        "recommendation_basis": row.get("Recommendation Basis", row.get("recommendation_basis", "")),
        "score": _safe(row.get("Score", row.get("score"))),
        "score_breakdown": {
            "trend": _safe(row.get("Score - Trend", row.get("score_trend"))),
            "macd": _safe(row.get("Score - MACD", row.get("score_macd"))),
            "rsi": _safe(row.get("Score - RSI", row.get("score_rsi"))),
            "volume": _safe(row.get("Score - Volume", row.get("score_volume"))),
            "adi": _safe(row.get("Score - ADI", row.get("score_adi"))),
            "support": _safe(row.get("Score - Support", row.get("score_support"))),
            "vwap": _safe(row.get("Score - VWAP", row.get("score_vwap"))),
            "volume_profile": _safe(row.get("Score - Volume Profile", row.get("score_volume_profile"))),
            "intraday": _safe(row.get("Score - Intraday", row.get("intraday_score"))),
        },
        "price": {
            "current_egp": _safe(row.get("Current EGP Price", row.get("close"))),
            "current_usd": _safe(row.get("Current USD Price", row.get("close_usd"))),
            "fair_value_egp": _safe(row.get("Implied Fair Value (EGP)", row.get("fair_value_egp"))),
            "fair_value_method": row.get("Fair Value Method", row.get("fair_value_method")),
            "undervalued": row.get("Undervalued (Yes/No)", row.get("undervalued")) in ("Yes", True),
        },
        "technicals": {
            "rsi": _safe(row.get("RSI (%)", row.get("rsi"))),
            "adx": _safe(row.get("ADX", row.get("adx"))),
            "mfi": _safe(row.get("MFI", row.get("mfi"))),
            "macd": _safe(row.get("MACD")),
            "macd_signal": _safe(row.get("MACD Signal")),
            "macd_bullish": row.get("MACD Bullish (Yes/No)", row.get("macd_bullish")) in ("Yes", True),
            "sma50": _safe(row.get("50 SMA")),
            "sma200": _safe(row.get("200 SMA")),
            "golden_cross": row.get("Golden Cross (Yes/No)", row.get("golden_cross")) in ("Yes", True),
            "death_cross": row.get("Death Cross (Yes/No)", row.get("death_cross")) in ("Yes", True),
            "bb_squeeze": row.get("BB Squeeze", row.get("bb_squeeze")) in ("Yes", True),
            "regime": row.get("regime"),
        },
        "volume": {
            "avg_3m": _safe(row.get("3-Month Avg Volume")),
            "last_day": _safe(row.get("Last Day Volume")),
            "multiplier": _safe(row.get("Volume Multiplier (vs 3M)")),
        },
        "support_resistance": {
            "support": _safe(row.get("Support")),
            "resistance": _safe(row.get("Resistance")),
            "support_levels": row.get("support_levels", []),
            "resistance_levels": row.get("resistance_levels", []),
        },
        "trade_plan": {
            "entry_price": _safe(row.get("Optimal Entry Price", row.get("entry_price"))),
            "entry_action": row.get("Entry Action", row.get("entry_action", "")),
            "entry_source": row.get("Entry Source", row.get("entry_source", "")),
            "stop_loss": _safe(row.get("Stop Loss", row.get("stop_loss"))),
            "tp1": _safe(row.get("Take Profit 1", row.get("tp1"))),
            "tp2": _safe(row.get("Take Profit 2", row.get("tp2"))),
            "tp3": _safe(row.get("Take Profit 3", row.get("tp3"))),
            "tp1_rr": _safe(row.get("TP1 Risk/Reward", row.get("tp1_rr"))),
            "tp2_rr": _safe(row.get("TP2 Risk/Reward", row.get("tp2_rr"))),
            "tp3_rr": _safe(row.get("TP3 Risk/Reward", row.get("tp3_rr"))),
        },
        "ml": {
            "signal": row.get("ML Signal", row.get("ml_signal")),
            "confidence": _ml_confidence(row),
            "medium_price": _safe(row.get("ML Medium Price", row.get("ml_medium_price"))),
            "low_price": _safe(row.get("ml_low_price")),
            "high_price": _safe(row.get("ml_high_price")),
        },
        "intraday": {
            "score": _safe(row.get("intraday_score")),
            "volume_skew": _safe(row.get("intraday_volume_skew")),
            "momentum": _safe(row.get("intraday_momentum")),
            "vwap_score": _safe(row.get("intraday_vwap_score")),
            "ad_signal": row.get("intraday_ad_signal"),
            "ad_score": _safe(row.get("intraday_ad_score")),
        },
        "holding": {
            "label": row.get("hold_label", row.get("hold_label", "")),
            "exit_strategy": row.get("hold_exit_strategy", ""),
            "days_min": _safe(row.get("hold_days_min")),
            "days_max": _safe(row.get("hold_days_max")),
            "days_target": _safe(row.get("hold_days_target")),
            "personalized": row.get("hold_personalized", False),
        },
        "patterns": {
            "candle_signal": row.get("candle_signal"),
            "chartscan_signal": row.get("chartscan_signal"),
            "chartscan_confidence": _safe(row.get("chartscan_confidence")),
        },
        "index": {
            "membership": row.get("Index Membership", row.get("index_membership", "UNINDEX")),
            "sentiment": row.get("index_sentiment"),
        },
    }


def generate_agent_json(rows: List[Dict], output_path: Optional[str] = None) -> str:
    """Generate structured JSON for AI agent consumption.

    Args:
        rows: Analysis rows from daily_run.py
        output_path: Optional override path

    Returns:
        Path to written JSON file
    """
    out = Path(output_path) if output_path else AGENT_JSON_PATH
    out.parent.mkdir(parents=True, exist_ok=True)

    tickers = [_build_ticker_block(r) for r in rows]

    # Summary statistics
    rec_counts = {}
    for t in tickers:
        rec = t["recommendation"]
        rec_counts[rec] = rec_counts.get(rec, 0) + 1

    scores = [t["score"] for t in tickers if t["score"] is not None]

    # Categorize tickers
    buy_tickers = [t["ticker"] for t in tickers if t["recommendation"] in ("Buy", "Strong Buy")]
    watch_tickers = [t["ticker"] for t in tickers if t["recommendation"] == "Watch"]
    avoid_tickers = [t["ticker"] for t in tickers if t["recommendation"] == "Avoid"]

    report = {
        "meta": {
            "generated": date.today().isoformat(),
            "source": "EGX Signal Generator",
            "version": "2.2",
            "description": "Structured signal data for AI agent consumption",
        },
        "summary": {
            "total_stocks": len(tickers),
            "recommendations": rec_counts,
            "avg_score": round(sum(scores) / len(scores), 1) if scores else 0,
            "min_score": round(min(scores), 1) if scores else 0,
            "max_score": round(max(scores), 1) if scores else 0,
            "buy_tickers": buy_tickers,
            "watch_tickers": watch_tickers,
            "avoid_tickers": avoid_tickers,
        },
        "stocks": tickers,
    }

    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log.info("Agent JSON generated: %s (%d stocks)", out, len(tickers))
    return str(out)
