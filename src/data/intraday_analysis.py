"""
Intraday Data Analysis for EGX Signals
5-min resolution, last 5 sessions, Buy/Watch tickers only
Provides: Volume Profile, Momentum, VWAP Interaction, Accumulation/Distribution
"""
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Cache ──
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "intraday_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Lazy import tvDatafeed
_tv = None


def _get_tv():
    global _tv
    if _tv is None:
        try:
            from tvDatafeed import TvDatafeed
            _tv = TvDatafeed()
            log.info("tvDatafeed initialized (nologin)")
        except ImportError:
            log.warning("tvDatafeed not installed")
            return None
    return _tv


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker}_intraday.parquet"


def fetch_intraday_5min(ticker: str, n_bars: int = 400,
                        exchange: str = "EGX",
                        use_cache: bool = True,
                        max_cache_age_hours: int = 4) -> Optional[pd.DataFrame]:
    """Fetch 5-min OHLCV for last 5 sessions (~390 bars/session).

    Only fetches for Buy/Watch tickers to save API calls.
    """
    from tvDatafeed import Interval
    tv = _get_tv()
    if tv is None:
        return None

    # Try cache first
    if use_cache:
        path = _cache_path(ticker)
        if path.exists():
            try:
                mtime = datetime.fromtimestamp(path.stat().st_mtime)
                age_hours = (datetime.now() - mtime).total_seconds() / 3600
                if age_hours < max_cache_age_hours:
                    df = pd.read_parquet(path)
                    if not df.empty:
                        return df
            except Exception:
                pass

    try:
        data = tv.get_hist(
            symbol=ticker,
            exchange=exchange,
            interval=Interval.in_5_minute,
            n_bars=n_bars,
        )
        if data is not None and not data.empty:
            data = data[["open", "high", "low", "close", "volume"]].copy()
            data.index.name = "datetime"
            # Save to cache
            try:
                data.to_parquet(_cache_path(ticker))
            except Exception:
                pass
            return data
    except Exception as e:
        log.warning("Failed to fetch intraday for %s: %s", ticker, e)

    return None


# ─────────────────────────────────────────────────────────────
# 1. VOLUME PROFILE ANALYSIS (Volume-at-Price distribution)
# ─────────────────────────────────────────────────────────────
def analyze_volume_profile(df: pd.DataFrame, bull_trend: bool = False) -> Dict:
    """Analyze volume distribution across price levels.

    Returns:
        - volume_skew: -1 (bearish, heavy volume at highs) to +1 (bullish, heavy at lows)
        - poc_price: Point of Control (price with highest volume)
        - va_high: Value Area High (70% of volume above this)
        - va_low: Value Area Low (70% of volume below this)
        - price_position_in_va: Where price sits in value area (0-1)

    When bull_trend is True, high volume at highs is treated as possible
    accumulation/breakout participation rather than distribution.
    """
    if df is None or df.empty or len(df) < 50:
        return {"volume_skew": 0, "poc_price": None, "va_high": None,
                "va_low": None, "price_position_in_va": 0.5,
                "trend_adjusted": False}

    typical = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"].values
    prices = typical.values

    # Create price bins
    n_bins = 50
    p_min, p_max = prices.min(), prices.max()
    if p_max <= p_min:
        return {"volume_skew": 0, "poc_price": float(p_max),
                "va_high": float(p_max), "va_low": float(p_min),
                "price_position_in_va": 0.5, "trend_adjusted": False}

    bin_edges = np.linspace(p_min, p_max, n_bins + 1)
    bin_vol = np.zeros(n_bins)
    bin_mid = (bin_edges[:-1] + bin_edges[1:]) / 2

    for price, volume in zip(prices, vol):
        idx = min(int((price - p_min) / (p_max - p_min) * (n_bins - 1)), n_bins - 1)
        bin_vol[idx] += volume

    # Point of Control (highest volume bin)
    poc_idx = np.argmax(bin_vol)
    poc_price = float(bin_mid[poc_idx])

    # Value Area (70% of volume)
    total_vol = bin_vol.sum()
    if total_vol > 0:
        sorted_indices = np.argsort(bin_vol)[::-1]
        cum_vol = 0
        va_indices = []
        for idx in sorted_indices:
            cum_vol += bin_vol[idx]
            va_indices.append(idx)
            if cum_vol >= total_vol * 0.7:
                break
        va_high = float(bin_mid[max(va_indices)])
        va_low = float(bin_mid[min(va_indices)])
    else:
        va_high = float(p_max)
        va_low = float(p_min)

    # Volume skew: compare volume above vs below POC
    vol_above_poc = bin_vol[poc_idx + 1:].sum()
    vol_below_poc = bin_vol[:poc_idx].sum()
    total_vol2 = vol_above_poc + vol_below_poc
    if total_vol2 > 0:
        skew = (vol_below_poc - vol_above_poc) / total_vol2  # +1 = all below (bullish)
    else:
        skew = 0

    trend_adjusted = False
    if bull_trend and skew < 0:
        # In a confirmed uptrend, volume at highs is often breakout/accumulation
        # participation — dampen the distribution signal instead of treating
        # every push higher as selling.
        skew = skew * 0.3
        trend_adjusted = True

    # Price position in value area
    last_price = float(df["close"].iloc[-1])
    if va_high > va_low:
        price_position = (last_price - va_low) / (va_high - va_low)
        price_position = max(0, min(1, price_position))
    else:
        price_position = 0.5

    return {
        "volume_skew": round(float(skew), 3),
        "poc_price": round(poc_price, 3),
        "va_high": round(va_high, 3),
        "va_low": round(va_low, 3),
        "price_position_in_va": round(float(price_position), 3),
        "trend_adjusted": trend_adjusted,
    }


# ─────────────────────────────────────────────────────────────
# 2. MOMENTUM SCORE
# ─────────────────────────────────────────────────────────────
def analyze_momentum(df: pd.DataFrame, bull_trend: bool = False) -> Dict:
    """Analyze intraday momentum across sessions.

    Returns:
        - momentum_score: 0-100 (higher = stronger bullish momentum)
        - rsi_acceleration: RSI is accelerating (+) or decelerating (-)
        - consecutive_up_sessions: Count of sessions with positive returns
        - session_trend: 'accelerating', 'stable', 'decelerating'
    """
    if df is None or df.empty or len(df) < 50:
        return {"momentum_score": 50, "rsi_acceleration": 0,
                "consecutive_up_sessions": 0, "session_trend": "stable"}

    # Resample to daily sessions
    daily = df["close"].resample("D").last().dropna()
    if len(daily) < 3:
        return {"momentum_score": 50, "rsi_acceleration": 0,
                "consecutive_up_sessions": 0, "session_trend": "stable"}

    # Session returns
    returns = daily.pct_change().dropna()
    consecutive_up = 0
    for r in returns.iloc[::-1]:
        if r > 0:
            consecutive_up += 1
        else:
            break

    # RSI on 5-min data
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).dropna()

    if len(rsi) >= 20:
        rsi_recent = rsi.iloc[-20:].mean()
        rsi_prev = rsi.iloc[-40:-20].mean() if len(rsi) >= 40 else rsi.iloc[0]
        rsi_accel = rsi_recent - rsi_prev
    else:
        rsi_recent = 50
        rsi_accel = 0

    # Momentum score (0-100)
    # Components: RSI position, RSI acceleration, consecutive up sessions
    rsi_component = rsi_recent  # 0-100
    accel_component = 50 + rsi_accel * 5  # centered at 50
    accel_component = max(0, min(100, accel_component))
    consecutive_component = min(100, consecutive_up * 20)

    momentum = (rsi_component * 0.4 + accel_component * 0.35 + consecutive_component * 0.25)
    momentum = max(0, min(100, float(momentum)))

    # Session trend classification
    if rsi_accel > 2:
        trend = "accelerating"
    elif rsi_accel < -2:
        trend = "decelerating"
    else:
        trend = "stable"

    # In a confirmed bull trend, RSI cooling from extremes is normal —
    # keep score anchored to actual RSI level when momentum is still firm.
    if bull_trend:
        if rsi_recent >= 55:
            momentum = max(momentum, min(100.0, float(rsi_recent)))
        if trend == "decelerating" and rsi_recent >= 60:
            trend = "stable"

    return {
        "momentum_score": round(momentum, 1),
        "rsi_acceleration": round(float(rsi_accel), 2),
        "consecutive_up_sessions": consecutive_up,
        "session_trend": trend,
    }


# ─────────────────────────────────────────────────────────────
# 3. VWAP INTERACTION
# ─────────────────────────────────────────────────────────────
def analyze_vwap_interaction(df: pd.DataFrame, bull_trend: bool = False) -> Dict:
    """Analyze how price interacts with VWAP across sessions.

    Returns:
        - vwap_score: 0-100 (higher = more bullish VWAP interaction)
        - sessions_above_vwap: Count of sessions closing above VWAP
        - avg_distance_from_vwap: Average % distance from VWAP
        - vwap_trend: 'above', 'below', 'crossing'
    """
    if df is None or df.empty or len(df) < 50:
        return {"vwap_score": 50, "sessions_above_vwap": 0,
                "avg_distance_from_vwap": 0, "vwap_trend": "neutral"}

    # Calculate cumulative VWAP
    typical = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].cumsum()
    cum_tp_vol = (typical * df["volume"]).cumsum()
    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)

    # Resample to daily
    daily_close = df["close"].resample("D").last().dropna()
    daily_vwap = vwap.resample("D").last().dropna()

    common_dates = daily_close.index.intersection(daily_vwap.index)
    if len(common_dates) < 3:
        return {"vwap_score": 50, "sessions_above_vwap": 0,
                "avg_distance_from_vwap": 0, "vwap_trend": "neutral"}

    sessions_above = 0
    distances = []
    for date in common_dates:
        close = daily_close[date]
        v = daily_vwap[date]
        if v > 0:
            dist = (close - v) / v * 100
            distances.append(dist)
            if close > v:
                sessions_above += 1

    avg_dist = np.mean(distances) if distances else 0

    # VWAP score (0-100)
    above_ratio = sessions_above / len(common_dates) if len(common_dates) > 0 else 0.5
    dist_component = 50 + avg_dist * 5  # +1% above VWAP = +5 pts
    dist_component = max(0, min(100, dist_component))

    vwap_score = (above_ratio * 60 + dist_component * 0.4)
    vwap_score = max(0, min(100, float(vwap_score)))

    # Sustained VWAP control in an uptrend is bullish even if distance is modest
    if bull_trend and above_ratio >= 0.6:
        vwap_score = max(vwap_score, 60.0)

    # VWAP trend
    if avg_dist > 1:
        trend = "above"
    elif avg_dist < -1:
        trend = "below"
    else:
        trend = "crossing"

    return {
        "vwap_score": round(vwap_score, 1),
        "sessions_above_vwap": sessions_above,
        "avg_distance_from_vwap": round(float(avg_dist), 2),
        "vwap_trend": trend,
    }


# ─────────────────────────────────────────────────────────────
# 4. ACCUMULATION / DISTRIBUTION
# ─────────────────────────────────────────────────────────────
def analyze_accumulation_distribution(df: pd.DataFrame, bull_trend: bool = False) -> Dict:
    """Detect smart money accumulation vs distribution.

    Logic:
    - High volume at low prices = accumulation (bullish)
    - High volume at high prices = distribution (bearish)
    - Uses CLV (Close Location Value) weighted by volume

    When bull_trend is True, volume at highs carries a weaker distribution
    penalty (breakout participation vs. distribution).

    Returns:
        - ad_signal: 'bullish', 'bearish', 'neutral'
        - ad_score: -1 (distribution) to +1 (accumulation)
        - money_flow_index: 0-100
        - volume_at_highs_ratio: % of volume in top 20% of price range
        - volume_at_lows_ratio: % of volume in bottom 20% of price range
    """
    if df is None or df.empty or len(df) < 50:
        return {"ad_signal": "neutral", "ad_score": 0,
                "money_flow_index": 50,
                "volume_at_highs_ratio": 0.5,
                "volume_at_lows_ratio": 0.5}

    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    vol = df["volume"].values

    # CLV (Close Location Value)
    clv = ((close - low) - (high - close)) / (high - low + 1e-10)
    ad_line = np.cumsum(clv * vol)

    # AD trend (last 50 bars)
    if len(ad_line) >= 50:
        ad_recent = ad_line[-50:]
        ad_trend = np.polyfit(range(len(ad_recent)), ad_recent, 1)[0]
    else:
        ad_trend = 0

    # Volume at highs vs lows
    price_range = high.max() - low.min()
    if price_range > 0:
        high_threshold = low.min() + price_range * 0.8  # Top 20%
        low_threshold = low.min() + price_range * 0.2   # Bottom 20%

        vol_at_highs = vol[high >= high_threshold].sum()
        vol_at_lows = vol[low <= low_threshold].sum()
        total_vol = vol.sum()

        vol_highs_ratio = vol_at_highs / total_vol if total_vol > 0 else 0.5
        vol_lows_ratio = vol_at_lows / total_vol if total_vol > 0 else 0.5
    else:
        vol_highs_ratio = 0.5
        vol_lows_ratio = 0.5

    # Money Flow Index (MFI)
    typical = (high + low + close) / 3
    money_flow = typical * vol
    positive_flow = money_flow[typical > np.roll(typical, 1)].sum()
    negative_flow = money_flow[typical < np.roll(typical, 1)].sum()
    mfi = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-10)))
    mfi = max(0, min(100, float(mfi)))

    # AD Score (-1 to +1)
    # Combine: AD trend direction + volume distribution
    ad_score = 0
    if ad_trend > 0:
        ad_score += 0.4
    if vol_lows_ratio > vol_highs_ratio + 0.05:
        ad_score += 0.4  # More volume at lows = accumulation
    elif vol_highs_ratio > vol_lows_ratio + 0.05:
        ad_score -= 0.2 if bull_trend else 0.4  # Softer distribution penalty in uptrend
    if mfi > 60:
        ad_score += 0.2
    elif mfi < 40:
        ad_score -= 0.2

    ad_score = max(-1, min(1, ad_score))

    if ad_score > 0.2:
        signal = "bullish"
    elif ad_score < -0.2:
        signal = "bearish"
    else:
        signal = "neutral"

    return {
        "ad_signal": signal,
        "ad_score": round(ad_score, 3),
        "money_flow_index": round(mfi, 1),
        "volume_at_highs_ratio": round(vol_highs_ratio, 3),
        "volume_at_lows_ratio": round(vol_lows_ratio, 3),
    }


# ─────────────────────────────────────────────────────────────
# MASTER: Full Intraday Analysis
# ─────────────────────────────────────────────────────────────
def run_intraday_analysis(ticker: str, recommendation: str = None,
                          trend_context: Dict = None) -> Dict:
    """Run full intraday analysis for a ticker.

    Only processes Buy/Watch tickers to save API calls.
    trend_context: optional dict with bullish/adx/regime flags so that
    volume-at-highs is not misread as distribution during confirmed uptrends.

    Returns combined analysis results.
    """
    # Skip Avoid tickers
    if recommendation and recommendation not in ("Buy", "Strong Buy", "Watch"):
        return {"skipped": True, "reason": f"recommendation={recommendation}"}

    # Fetch 5-min data (last 5 sessions)
    df = fetch_intraday_5min(ticker, n_bars=400)
    if df is None or df.empty:
        return {"skipped": True, "reason": "no_data"}

    trend_context = trend_context or {}
    bull_trend = bool(trend_context.get("bullish", False))

    # Run all analyses
    vol_profile = analyze_volume_profile(df, bull_trend=bull_trend)
    momentum = analyze_momentum(df, bull_trend=bull_trend)
    vwap = analyze_vwap_interaction(df, bull_trend=bull_trend)
    ad = analyze_accumulation_distribution(df, bull_trend=bull_trend)

    # Combined intraday score (0-100)
    # Weights: volume profile 30%, momentum 30%, VWAP 25%, AD 15%
    vol_score = 50 + vol_profile["volume_skew"] * 50  # skew -1..1 -> 0..100
    vol_score = max(0, min(100, vol_score))

    intraday_score = (
        vol_score * 0.30 +
        momentum["momentum_score"] * 0.30 +
        vwap["vwap_score"] * 0.25 +
        (50 + ad["ad_score"] * 50) * 0.15  # AD score -1..1 -> 0..100
    )
    intraday_score = max(0, min(100, float(intraday_score)))

    # Confirmed bull trends: intraday noise should not veto trend quality
    if bull_trend and intraday_score < 55:
        intraday_score = max(intraday_score, 55.0)

    return {
        "skipped": False,
        "ticker": ticker,
        "intraday_score": round(intraday_score, 1),
        "volume_profile": vol_profile,
        "momentum": momentum,
        "vwap": vwap,
        "accumulation_distribution": ad,
        "trend_context": trend_context,
        "bull_trend": bull_trend,
        "bars_analyzed": len(df),
    }
