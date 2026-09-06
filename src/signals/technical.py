"""
Technical Indicator Calculations for EGX Signals
Migrated from V12 tool - pure calculation, no plotting
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
FIB_LEVELS = [0.382, 0.5, 0.618]
FIB_LABELS = ["38.2%", "50%", "61.8%"]
FIB_COLORS = ["#facc15", "#f97316", "#ef4444"]

# ─────────────────────────────────────────────────────────────
# Volume Weighted Average Price (VWAP)
# ─────────────────────────────────────────────────────────────
def add_vwap(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    out = df.copy()
    typical_price = (out["High"] + out["Low"] + out["Close"]) / 3.0
    cumulative_tp_vol = (typical_price * out["Volume"]).rolling(period).sum()
    cumulative_vol = out["Volume"].rolling(period).sum()
    out["VWAP"] = cumulative_tp_vol / cumulative_vol.replace(0, np.nan)
    out["dist_VWAP"] = (out["Close"] - out["VWAP"]) / out["VWAP"].replace(0, np.nan)
    return out

# ─────────────────────────────────────────────────────────────
# Fixed Range Volume Profile
# ─────────────────────────────────────────────────────────────
def compute_volume_profile(df: pd.DataFrame, n_bins: int = 20) -> Dict:
    data = df[["High", "Low", "Close", "Volume"]].dropna().copy()
    if len(data) < 5:
        return {"poc": None, "va_high": None, "va_low": None, "bins": [], "above_poc": None}

    price_min = float(data["Low"].min())
    price_max = float(data["High"].max())
    if price_max == price_min:
        return {"poc": price_max, "va_high": price_max, "va_low": price_min, "bins": [], "above_poc": None}

    bin_size = (price_max - price_min) / n_bins
    bins = []
    for i in range(n_bins):
        lo = price_min + i * bin_size
        hi = lo + bin_size
        mask = (data["Low"] < hi) & (data["High"] >= lo)
        vol_in_bin = float(data.loc[mask, "Volume"].sum())
        bins.append({"price_low": round(lo, 3), "price_high": round(hi, 3), "volume": vol_in_bin})

    max_vol = max(b["volume"] for b in bins) if bins else 1
    for b in bins:
        b["pct_of_max"] = round(b["volume"] / max_vol * 100, 1) if max_vol > 0 else 0

    poc_bin = max(bins, key=lambda b: b["volume"])
    poc = round((poc_bin["price_low"] + poc_bin["price_high"]) / 2, 3)

    total_vol = sum(b["volume"] for b in bins)
    value_threshold = total_vol * 0.70
    sorted_bins = sorted(bins, key=lambda b: -b["volume"])
    cum_vol = 0
    va_bins = []
    for b in sorted_bins:
        cum_vol += b["volume"]
        va_bins.append(b)
        if cum_vol >= value_threshold:
            break
    va_high = round(max(b["price_high"] for b in va_bins), 3)
    va_low = round(min(b["price_low"] for b in va_bins), 3)

    last_close = float(data["Close"].iloc[-1])
    above_poc = last_close > poc

    return {
        "poc": poc,
        "va_high": va_high,
        "va_low": va_low,
        "bins": bins,
        "above_poc": above_poc,
        "volume_at_poc": poc_bin["volume"],
    }

# ─────────────────────────────────────────────────────────────
# Moving Averages
# ─────────────────────────────────────────────────────────────
def add_moving_averages(df: pd.DataFrame, periods: List[int] = [20, 50, 200]) -> pd.DataFrame:
    out = df.copy()
    for p in periods:
        if len(out) >= p:
            out[f"SMA_{p}"] = out["Close"].rolling(p).mean()
            out[f"EMA_{p}"] = out["Close"].ewm(span=p, adjust=False).mean()
    return out

# ─────────────────────────────────────────────────────────────
# RSI
# ─────────────────────────────────────────────────────────────
def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    out = df.copy()
    delta = out["Close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI"] = 100 - (100 / (1 + rs))
    return out

# ─────────────────────────────────────────────────────────────
# MACD
# ─────────────────────────────────────────────────────────────
def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    out = df.copy()
    ema_fast = out["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = out["Close"].ewm(span=slow, adjust=False).mean()
    out["MACD"] = ema_fast - ema_slow
    out["MACD_sig"] = out["MACD"].ewm(span=signal, adjust=False).mean()
    out["MACD_hist"] = out["MACD"] - out["MACD_sig"]
    return out

# ─────────────────────────────────────────────────────────────
# Bollinger Bands
# ─────────────────────────────────────────────────────────────
def add_bollinger(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    out = df.copy()
    sma = out["Close"].rolling(period).mean()
    std = out["Close"].rolling(period).std()
    out["BB_mid"] = sma
    out["BB_up"] = sma + std_dev * std
    out["BB_dn"] = sma - std_dev * std
    out["BB_pct"] = (out["Close"] - out["BB_dn"]) / (out["BB_up"] - out["BB_dn"])
    return out

# ─────────────────────────────────────────────────────────────
# ATR
# ─────────────────────────────────────────────────────────────
def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    out = df.copy()
    high = out["High"]
    low = out["Low"]
    prev = out["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev).abs(),
        (low - prev).abs(),
    ], axis=1).max(axis=1)
    out["ATR"] = tr.ewm(span=period, adjust=False).mean()
    return out

# ─────────────────────────────────────────────────────────────
# ADX
# ─────────────────────────────────────────────────────────────
def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    out = df.copy()
    high = out["High"]
    low = out["Low"]
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = out["Close"].shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr_s = tr.ewm(span=period, adjust=False).mean()
    pos_di = 100 * pd.Series(pos_dm, index=out.index).ewm(span=period, adjust=False).mean() / atr_s.replace(0, np.nan)
    neg_di = 100 * pd.Series(neg_dm, index=out.index).ewm(span=period, adjust=False).mean() / atr_s.replace(0, np.nan)
    dx = (100 * (pos_di - neg_di).abs() / (pos_di + neg_di).replace(0, np.nan))
    out["ADX"] = dx.ewm(span=period, adjust=False).mean()
    out["DI_plus"] = pos_di
    out["DI_minus"] = neg_di
    return out

# ─────────────────────────────────────────────────────────────
# Fibonacci Retracement
# ─────────────────────────────────────────────────────────────
def compute_fibonacci(df: pd.DataFrame) -> Dict:
    if "High" in df.columns and "Low" in df.columns:
        swing_high = float(df["High"].dropna().max())
        swing_low = float(df["Low"].dropna().min())
    else:
        swing_high = float(df["Close"].dropna().max())
        swing_low = float(df["Close"].dropna().min())

    diff = swing_high - swing_low
    levels = []
    for ratio, label, color in zip(FIB_LEVELS, FIB_LABELS, FIB_COLORS):
        price = swing_high - ratio * diff
        levels.append({
            "ratio": ratio,
            "label": f"Fib {label}",
            "price": round(price, 3),
            "color": color,
        })

    return {
        "swing_high": swing_high,
        "swing_low": swing_low,
        "levels": levels,
        "n_bars": len(df),
    }

# ─────────────────────────────────────────────────────────────
# Support & Resistance (Pivot-point cluster method)
# ─────────────────────────────────────────────────────────────
def compute_support_resistance(df: pd.DataFrame, n_levels: int = 5, tolerance_pct: float = 1.5,
                                lookback: int = 90, max_distance_pct: float = 25.0) -> Dict:
    """Compute support/resistance using swing points and volume clusters.
    
    Combines:
    1. Swing point detection (local highs/lows)
    2. Volume-weighted price zones (where most trading occurred)
    3. Fibonacci levels
    """
    data = df[["High", "Low", "Close", "Volume"]].dropna()
    if len(data) < 15:
        lp = float(data["Close"].iloc[-1]) if len(data) > 0 else 0
        return {"support": [], "resistance": [], "last_price": lp}

    recent = data.tail(lookback)
    closes = recent["Close"].values
    highs = recent["High"].values
    lows = recent["Low"].values
    volumes = recent["Volume"].values.astype(float)
    n = len(recent)
    last_price = float(closes[-1])
    max_dist = last_price * (max_distance_pct / 100)

    # --- Method 1: Swing points at multiple timeframes ---
    swing_prices = []
    for order in [2, 3, 5]:
        if n < 2 * order + 1:
            continue
        for i in range(order, n - order):
            if highs[i] == max(highs[i - order: i + order + 1]):
                swing_prices.append(("R", i, highs[i]))
            if lows[i] == min(lows[i - order: i + order + 1]):
                swing_prices.append(("S", i, lows[i]))

    # --- Method 2: Volume-weighted price zones ---
    # Divide price range into bins and weight by volume
    price_min = min(lows)
    price_max = max(highs)
    n_bins = min(50, max(20, n // 2))
    bin_edges = np.linspace(price_min, price_max, n_bins + 1)
    bin_volume = np.zeros(n_bins)
    bin_mid = (bin_edges[:-1] + bin_edges[1:]) / 2

    for i in range(n):
        lo_idx = max(0, int((lows[i] - price_min) / (price_max - price_min) * (n_bins - 1)))
        hi_idx = min(n_bins - 1, int((highs[i] - price_min) / (price_max - price_min) * (n_bins - 1)))
        for j in range(lo_idx, hi_idx + 1):
            bin_volume[j] += volumes[i] / max(1, hi_idx - lo_idx + 1)

    # Find volume peaks (local maxima in volume profile)
    vol_peaks = []
    for i in range(1, n_bins - 1):
        if bin_volume[i] > bin_volume[i - 1] and bin_volume[i] > bin_volume[i + 1]:
            vol_peaks.append((bin_mid[i], bin_volume[i]))
    # Also check edges
    if n_bins >= 2:
        if bin_volume[0] > bin_volume[1]:
            vol_peaks.append((bin_mid[0], bin_volume[0]))
        if bin_volume[-1] > bin_volume[-2]:
            vol_peaks.append((bin_mid[-1], bin_volume[-1]))

    # Combine all candidate levels
    candidates = []
    for stype, idx, price in swing_prices:
        if abs(price - last_price) <= max_dist:
            candidates.append(price)
    for price, vol in vol_peaks:
        if abs(price - last_price) <= max_dist:
            candidates.append(price)

    if not candidates:
        return {"support": [], "resistance": [], "last_price": last_price}

    # Cluster candidates
    candidates = sorted(set(candidates))
    clusters = []
    current = [candidates[0]]
    for c in candidates[1:]:
        cluster_mean = np.mean(current)
        if abs(c - cluster_mean) / cluster_mean * 100 <= tolerance_pct:
            current.append(c)
        else:
            clusters.append(current)
            current = [c]
    clusters.append(current)

    # Score clusters by how many times price touched them
    result = []
    for cluster in clusters:
        avg_price = np.mean(cluster)
        if abs(avg_price - last_price) > max_dist:
            continue

        # Count touches (bars where price came within 0.5% of this level)
        touches = 0
        for i in range(n):
            if abs(highs[i] - avg_price) / avg_price < 0.005 or abs(lows[i] - avg_price) / avg_price < 0.005:
                touches += 1

        # Volume at this level
        vol_at_level = sum(volumes[i] for i in range(n)
                          if min(highs[i], lows[i]) <= avg_price <= max(highs[i], lows[i]))

        # Recency: more recent touches count more
        recency = 0
        for i in range(n):
            if abs(highs[i] - avg_price) / avg_price < 0.005 or abs(lows[i] - avg_price) / avg_price < 0.005:
                recency += 1.0 / (1.0 + (n - i) / 10.0)

        score = touches * 2 + recency + (len(cluster) * 0.5)

        if touches >= 4 or score >= 6:
            strength = "Strong"
        elif touches >= 2 or score >= 3:
            strength = "Medium"
        else:
            strength = "Weak"

        result.append({
            "price": round(float(avg_price), 3),
            "touches": touches,
            "strength": strength,
            "recency_score": round(recency, 2),
            "score": round(score, 2),
        })

    result.sort(key=lambda x: -x["score"])

    support = [s for s in result if s["price"] < last_price][:n_levels]
    resistance = [r for r in result if r["price"] > last_price][:n_levels]

    return {
        "support": support,
        "resistance": resistance,
        "last_price": last_price,
    }

# ─────────────────────────────────────────────────────────────
# Candlestick Pattern Detection
# ─────────────────────────────────────────────────────────────
def detect_candlestick_patterns(df: pd.DataFrame) -> Dict:
    data = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if len(data) < 3:
        return {"patterns": [], "bullish_count": 0, "bearish_count": 0,
                "latest_signal": "neutral", "score_delta": 0}

    patterns = []
    window = data.iloc[-5:]
    rows = [window.iloc[i] for i in range(len(window))]

    def body(row): return abs(row["Close"] - row["Open"])
    def upper_wick(row): return row["High"] - max(row["Close"], row["Open"])
    def lower_wick(row): return min(row["Close"], row["Open"]) - row["Low"]
    def candle_range(row): return row["High"] - row["Low"]

    last = rows[-1]
    prev = rows[-2] if len(rows) >= 2 else None
    body_last = body(last)
    rng_last = candle_range(last)

    if rng_last > 0:
        if body_last < 0.10 * rng_last:
            patterns.append({"name": "Doji", "signal": "neutral", "description": "Indecision", "points": 0})
        elif (lower_wick(last) >= 2 * body_last and upper_wick(last) <= 0.3 * body_last and last["Close"] > last["Open"]):
            patterns.append({"name": "Hammer 🔨", "signal": "bullish", "description": "Bullish reversal", "points": 8})
        elif (upper_wick(last) >= 2 * body_last and lower_wick(last) <= 0.3 * body_last and last["Close"] > last["Open"]):
            patterns.append({"name": "Inverted Hammer", "signal": "bullish", "description": "Possible bullish reversal", "points": 5})
        elif (upper_wick(last) >= 2 * body_last and lower_wick(last) <= 0.3 * body_last and last["Close"] < last["Open"]):
            patterns.append({"name": "Shooting Star ⭐", "signal": "bearish", "description": "Bearish reversal", "points": -8})
        elif (lower_wick(last) >= 2 * body_last and upper_wick(last) <= 0.3 * body_last and last["Close"] < last["Open"]):
            patterns.append({"name": "Hanging Man", "signal": "bearish", "description": "Bearish warning at highs", "points": -5})
        elif (body_last > 0.7 * rng_last and last["Close"] > last["Open"]):
            patterns.append({"name": "Strong Bullish Candle", "signal": "bullish", "description": "Decisive buying", "points": 6})
        elif (body_last > 0.7 * rng_last and last["Close"] < last["Open"]):
            patterns.append({"name": "Strong Bearish Candle", "signal": "bearish", "description": "Decisive selling", "points": -6})

    if prev is not None and candle_range(prev) > 0:
        if (prev["Close"] < prev["Open"] and last["Close"] > last["Open"]
                and last["Open"] < prev["Close"] and last["Close"] > prev["Open"]):
            patterns.append({"name": "Bullish Engulfing 📈", "signal": "bullish", "description": "Strong bullish reversal", "points": 12})
        elif (prev["Close"] > prev["Open"] and last["Close"] < last["Open"]
              and last["Open"] > prev["Close"] and last["Close"] < prev["Open"]):
            patterns.append({"name": "Bearish Engulfing 📉", "signal": "bearish", "description": "Strong bearish reversal", "points": -12})
        elif (prev["Close"] < prev["Open"] and last["Close"] > last["Open"]
              and last["Open"] > prev["Close"] and last["Close"] < prev["Open"]):
            patterns.append({"name": "Bullish Harami", "signal": "bullish", "description": "Possible bullish reversal", "points": 5})

    if len(rows) >= 3:
        r1, r2, r3 = rows[-3], rows[-2], rows[-1]
        if (r1["Close"] > r1["Open"] and r2["Close"] > r2["Open"] and r3["Close"] > r3["Open"]
                and r2["Close"] > r1["Close"] and r3["Close"] > r2["Close"]
                and body(r1) > 0.5 * candle_range(r1) and body(r2) > 0.5 * candle_range(r2) and body(r3) > 0.5 * candle_range(r3)):
            patterns.append({"name": "Three White Soldiers 🪖", "signal": "bullish", "description": "Strong bullish continuation", "points": 15})
        elif (r1["Close"] < r1["Open"] and r2["Close"] < r2["Open"] and r3["Close"] < r3["Open"]
              and r2["Close"] < r1["Close"] and r3["Close"] < r2["Close"]
              and body(r1) > 0.5 * candle_range(r1) and body(r2) > 0.5 * candle_range(r2) and body(r3) > 0.5 * candle_range(r3)):
            patterns.append({"name": "Three Black Crows 🐦", "signal": "bearish", "description": "Strong bearish continuation", "points": -15})

    bullish_count = sum(1 for p in patterns if p["signal"] == "bullish")
    bearish_count = sum(1 for p in patterns if p["signal"] == "bearish")
    score_delta = sum(p.get("points", 0) for p in patterns)

    if bullish_count > bearish_count:
        latest_signal = "bullish"
    elif bearish_count > bullish_count:
        latest_signal = "bearish"
    else:
        latest_signal = "neutral"

    return {
        "patterns": patterns,
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "latest_signal": latest_signal,
        "score_delta": score_delta,
    }

# ─────────────────────────────────────────────────────────────
# Trend Regime Detection
# ─────────────────────────────────────────────────────────────
def detect_trend_regime(df: pd.DataFrame) -> Dict:
    data = df.dropna(subset=["Close"])
    if "ADX" not in data.columns or data["ADX"].isna().all():
        return {
            "regime": "unknown", "adx": 0.0, "adx_label": "N/A",
            "direction": "neutral",
            "entry_note": "Insufficient data for regime detection.",
            "hold_note": "Use default 5-day forecast window.",
        }

    last = data.iloc[-1]
    adx = float(last["ADX"]) if not pd.isna(last["ADX"]) else 0.0

    close = float(last["Close"])
    sma50 = float(last.get("SMA_50", close))
    sma200 = float(last.get("SMA_200", close))

    di_plus = float(last.get("DI_plus", 0))
    di_minus = float(last.get("DI_minus", 0))

    if di_plus > di_minus and close > sma50:
        direction = "bullish"
    elif di_minus > di_plus and close < sma50:
        direction = "bearish"
    else:
        direction = "neutral"

    if adx >= 25:
        regime = "trending_up" if direction == "bullish" else ("trending_down" if direction == "bearish" else "trending")
        adx_label = "Strong Trend"
    elif adx >= 20:
        regime = "transitioning"
        adx_label = "Weak Trend / Transitioning"
    else:
        regime = "ranging"
        adx_label = "Ranging / Consolidating"

    if "trending_up" in regime:
        entry_note = ("Trending up: favour buying pullbacks to EMA20 or SMA50. "
                      "Avoid chasing breakouts far from the moving averages.")
        hold_note = ("In a strong uptrend, consider extending hold beyond 5 days — "
                     "trend followers often target 10–20 days or until EMA20 is breached.")
    elif "trending_down" in regime:
        entry_note = ("Trending down: exercise caution on long entries. "
                      "Wait for a confirmed reversal signal before entering long.")
        hold_note = ("In a downtrend, keep holding periods short (3–5 days max) "
                     "and tighten stops.")
    elif regime == "ranging":
        entry_note = ("Ranging market: buy near support, sell near resistance. "
                      "Avoid trend-following entries in the middle of the range.")
        hold_note = ("In a ranging market, target the range boundary as TP and exit "
                     "within 5–8 days.")
    else:
        entry_note = ("Market is transitioning — trend forming or reversing. "
                      "Reduce position size by 30–50% until regime clarifies (ADX > 25).")
        hold_note = ("Uncertain regime: stick to the 5-day ML forecast window "
                     "and exit at TP1 rather than holding for TP2 or TP3.")

    return {
        "regime": regime,
        "adx": round(adx, 1),
        "adx_label": adx_label,
        "direction": direction,
        "entry_note": entry_note,
        "hold_note": hold_note,
    }

# ─────────────────────────────────────────────────────────────
# Master Enricher
# ─────────────────────────────────────────────────────────────
def enrich(df: pd.DataFrame, ma_periods: List[int] = [20, 50, 200], rsi_period: int = 14, atr_period: int = 14) -> Dict:
    enriched = df.copy()
    enriched = add_moving_averages(enriched, ma_periods)
    enriched = add_rsi(enriched, rsi_period)
    enriched = add_macd(enriched)
    enriched = add_bollinger(enriched)
    enriched = add_atr(enriched, atr_period)
    enriched = add_adx(enriched, atr_period)
    enriched = add_vwap(enriched)

    fib = compute_fibonacci(df)
    sr = compute_support_resistance(df)
    patterns = detect_candlestick_patterns(enriched)
    regime = detect_trend_regime(enriched)
    # VP POC uses last 1 month (~21 trading days) for current relevance
    df_1m = df.tail(21) if len(df) > 21 else df
    vol_profile = compute_volume_profile(df_1m)

    return {
        "df_enriched": enriched,
        "fibonacci": fib,
        "sr": sr,
        "patterns": patterns,
        "regime": regime,
        "volume_profile": vol_profile,
    }