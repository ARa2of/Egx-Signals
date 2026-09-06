"""
ChartScan AI - YOLOv8 Candlestick Pattern Detection
Migrated from Tool 1
"""
import logging
import os
from typing import Dict, Optional
import pandas as pd

log = logging.getLogger(__name__)

_chartscan_model = None
_chartscan_enabled = False

def init_chartscan(model_path: str = "weights/custom_yolov8.pt", enabled: bool = True) -> bool:
    """Initialize ChartScan AI. Returns True if model loaded successfully."""
    global _chartscan_model, _chartscan_enabled
    _chartscan_enabled = enabled
    
    if not enabled:
        log.info("ChartScan AI disabled")
        return False
    
    if _chartscan_model is not None:
        return True
    
    try:
        from ultralytics import YOLO
        from pathlib import Path
        # Resolve path relative to project root, not CWD
        project_root = Path(__file__).parent.parent.parent
        abs_path = project_root / model_path
        if not abs_path.exists():
            log.warning("ChartScanAI model not found at %s", abs_path)
            _chartscan_enabled = False
            return False
        _chartscan_model = YOLO(str(abs_path))
        log.info("ChartScanAI model loaded from %s", abs_path)
        return True
    except Exception as e:
        log.warning("Failed to load ChartScanAI model: %s", e)
        _chartscan_enabled = False
        return False

def chartscan_analyze(history: pd.DataFrame, ticker: str) -> Optional[Dict]:
    """Run YOLOv8 candlestick pattern detection on a stock chart.
    
    Generates a candlestick chart from yfinance history, runs YOLO inference,
    and returns Buy/Sell signal with confidence.
    
    Returns dict with keys: signal, confidence, buy_patterns, sell_patterns
    or None on failure.
    """
    if not _chartscan_enabled or _chartscan_model is None:
        return None
    
    if history is None or len(history) < 30:
        return None
    
    try:
        import mplfinance as mpf
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
        from io import BytesIO

        # Use latest 180 candles (or all if fewer)
        chart_data = history.iloc[-180:].copy()

        # Ensure proper datetime index for mplfinance
        if not isinstance(chart_data.index, pd.DatetimeIndex):
            chart_data.index = pd.to_datetime(chart_data.index)

        # Generate candlestick chart
        fig, ax = mpf.plot(
            chart_data, type="candle", style="yahoo",
            volume=False, returnfig=True, figsize=(18, 6.5)
        )
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)

        # Run YOLO inference
        img = Image.open(buf)
        results = _chartscan_model.predict(img, conf=0.3, verbose=False)

        if not results or len(results) == 0:
            return None

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return {"signal": "Neutral", "confidence": 0.0,
                    "buy_patterns": 0, "sell_patterns": 0}

        # Parse class labels: class 0 = Buy, class 1 = Sell
        buy_count = 0
        sell_count = 0
        confs = []
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            confs.append(conf)
            if cls == 0:
                buy_count += 1
            elif cls == 1:
                sell_count += 1

        avg_conf = sum(confs) / len(confs) if confs else 0.0

        if buy_count > sell_count:
            signal = "Buy"
        elif sell_count > buy_count:
            signal = "Sell"
        else:
            signal = "Neutral"

        log.info("%s: ChartScanAI — signal=%s, conf=%.2f, buy=%d, sell=%d",
                 ticker, signal, avg_conf, buy_count, sell_count)

        return {
            "signal": signal,
            "confidence": round(avg_conf, 4),
            "buy_patterns": buy_count,
            "sell_patterns": sell_count,
        }
    except Exception as e:
        log.warning("ChartScanAI failed for %s: %s", ticker, e)
        return None

def is_enabled() -> bool:
    return _chartscan_enabled