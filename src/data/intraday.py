"""
Intraday Data Loader using tvDatafeed
Provides hourly and half-hourly data for EGX stocks.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd

log = logging.getLogger(__name__)

# Lazy import to avoid startup cost
_tv = None


def _get_tv():
    """Get tvDatafeed instance (lazy init)."""
    global _tv
    if _tv is None:
        try:
            from tvDatafeed import TvDatafeed
            _tv = TvDatafeed()
            log.info("tvDatafeed initialized (nologin mode)")
        except ImportError:
            log.warning("tvDatafeed not installed")
            return None
    return _tv


def fetch_hourly(ticker: str, n_bars: int = 500, exchange: str = "EGX") -> Optional[pd.DataFrame]:
    """Fetch hourly OHLCV data for a ticker.

    Args:
        ticker: EGX ticker symbol (e.g., 'COMI', 'ETEL')
        n_bars: Number of hourly bars to fetch (max ~1000 for nologin)
        exchange: Exchange name (default: EGX)

    Returns:
        DataFrame with columns: open, high, low, close, volume
        Index is datetime (hourly timestamps)
    """
    from tvDatafeed import Interval
    tv = _get_tv()
    if tv is None:
        return None

    try:
        data = tv.get_hist(
            symbol=ticker,
            exchange=exchange,
            interval=Interval.in_1_hour,
            n_bars=n_bars,
        )
        if data is not None and not data.empty:
            # Standardize columns
            data = data[["open", "high", "low", "close", "volume"]].copy()
            data.index.name = "datetime"
            log.info("Fetched %d hourly bars for %s", len(data), ticker)
            return data
    except Exception as e:
        log.warning("Failed to fetch hourly data for %s: %s", ticker, e)

    return None


def fetch_30min(ticker: str, n_bars: int = 500, exchange: str = "EGX") -> Optional[pd.DataFrame]:
    """Fetch 30-minute OHLCV data for a ticker."""
    from tvDatafeed import Interval
    tv = _get_tv()
    if tv is None:
        return None

    try:
        data = tv.get_hist(
            symbol=ticker,
            exchange=exchange,
            interval=Interval.in_30_minute,
            n_bars=n_bars,
        )
        if data is not None and not data.empty:
            data = data[["open", "high", "low", "close", "volume"]].copy()
            data.index.name = "datetime"
            log.info("Fetched %d 30-min bars for %s", len(data), ticker)
            return data
    except Exception as e:
        log.warning("Failed to fetch 30-min data for %s: %s", ticker, e)

    return None


def fetch_15min(ticker: str, n_bars: int = 500, exchange: str = "EGX") -> Optional[pd.DataFrame]:
    """Fetch 15-minute OHLCV data for a ticker."""
    from tvDatafeed import Interval
    tv = _get_tv()
    if tv is None:
        return None

    try:
        data = tv.get_hist(
            symbol=ticker,
            exchange=exchange,
            interval=Interval.in_15_minute,
            n_bars=n_bars,
        )
        if data is not None and not data.empty:
            data = data[["open", "high", "low", "close", "volume"]].copy()
            data.index.name = "datetime"
            log.info("Fetched %d 15-min bars for %s", len(data), ticker)
            return data
    except Exception as e:
        log.warning("Failed to fetch 15-min data for %s: %s", ticker, e)

    return None


def compute_intraday_rsi(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """Compute RSI from intraday data."""
    if df is None or df.empty or len(df) < period + 1:
        return None

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    last_rsi = rsi.dropna().iloc[-1] if not rsi.dropna().empty else None
    return float(last_rsi) if last_rsi is not None else None


def compute_intraday_vwap(df: pd.DataFrame) -> Optional[float]:
    """Compute VWAP from intraday data."""
    if df is None or df.empty:
        return None

    typical = (df["high"] + df["low"] + df["close"]) / 3
    cumulative_tp_vol = (typical * df["volume"]).cumsum()
    cumulative_vol = df["volume"].cumsum()
    vwap = cumulative_tp_vol / cumulative_vol.replace(0, pd.NA)
    last_vwap = vwap.dropna().iloc[-1] if not vwap.dropna().empty else None
    return float(last_vwap) if last_vwap is not None else None


def compute_intraday_volatility(df: pd.DataFrame) -> Optional[float]:
    """Compute intraday volatility (ATR as % of price)."""
    if df is None or df.empty or len(df) < 2:
        return None

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(14).mean().iloc[-1]
    last_close = df["close"].iloc[-1]
    if last_close > 0:
        return float(atr / last_close * 100)
    return None


def get_intraday_summary(ticker: str, exchange: str = "EGX") -> Dict:
    """Get a summary of intraday data for a ticker.

    Returns dict with:
        - hourly_data: DataFrame or None
        - hourly_rsi: float or None
        - hourly_vwap: float or None
        - volatility_pct: float or None
        - data_quality: str ('good', 'limited', 'none')
    """
    result = {
        "hourly_data": None,
        "hourly_rsi": None,
        "hourly_vwap": None,
        "volatility_pct": None,
        "data_quality": "none",
    }

    hourly = fetch_hourly(ticker, n_bars=200, exchange=exchange)
    if hourly is not None and len(hourly) >= 20:
        result["hourly_data"] = hourly
        result["hourly_rsi"] = compute_intraday_rsi(hourly)
        result["hourly_vwap"] = compute_intraday_vwap(hourly)
        result["volatility_pct"] = compute_intraday_volatility(hourly)
        result["data_quality"] = "good"
    elif hourly is not None:
        result["data_quality"] = "limited"

    return result
