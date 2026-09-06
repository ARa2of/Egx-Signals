"""
Unified Data Loader for EGX Signals
Combines TradingView batch scanning (primary) with yfinance fallback
"""
from __future__ import annotations
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from tradingview_ta import Interval, TradingView

from src.config import load_params

log = logging.getLogger(__name__)

params = load_params()
tv_cfg = params["tradingview"]
data_cfg = params["data"]
enhanced_cfg = params["enhanced_entry"]

# ─────────────────────────────────────────────────────────────
# Constants from config
# ─────────────────────────────────────────────────────────────
HISTORY_PERIOD = data_cfg["history_period"]
MIN_TRADING_DAYS = data_cfg["min_trading_days"]
ADL_WINDOW = data_cfg["adl_window"]
EGX_SUFFIX = data_cfg["egx_suffix"]
AUTO_ADJUST = data_cfg["auto_adjust"]
FX_TICKER = data_cfg["fx_ticker"]
EGP_SIMILARITY_BAND = data_cfg["egp_similarity_band"]
SWING_ORDER = data_cfg["swing_order"]
SR_LOOKBACK_DAYS = data_cfg["sr_lookback_days"]
BUY_VOL_AVG_DAYS = data_cfg["buy_vol_avg_days"]

TA_EXCHANGE = tv_cfg["exchange"]
TA_SCREENER = tv_cfg["screener"]
TA_EXTRA_COLUMNS = tv_cfg["extra_columns"]
TA_SYMBOLS_PER_REQUEST = tv_cfg["symbols_per_request"]
TA_BATCH_RETRIES = tv_cfg["batch_retries"]
TA_BATCH_RETRY_DELAY = tv_cfg["batch_retry_delay"]
TA_INTER_CHUNK_DELAY_MIN = tv_cfg["inter_chunk_delay_min"]
TA_INTER_CHUNK_DELAY_MAX = tv_cfg["inter_chunk_delay_max"]
TA_USER_AGENT = tv_cfg["user_agent"]

REFERENCE_PE_EGX = params["fundamentals"]["reference_pe_egx"]

# ─────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────
@dataclass
class TickerData:
    raw_ticker: str
    yf_ticker: str
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    ok: bool = False
    reason: str = ""

@dataclass
class TickerTA:
    raw_ticker: str
    indicators: dict = field(default_factory=dict)
    ok: bool = False
    reason: str = ""
    fetch_time: Optional[datetime] = None

# ─────────────────────────────────────────────────────────────
# Input Reading
# ─────────────────────────────────────────────────────────────
def read_ticker_list(path: str, sheet_name: str = "Selected_Stocks") -> List[str]:
    df = pd.read_excel(path, sheet_name=sheet_name, usecols=[0])
    col = df.columns[0]
    tickers = (
        df[col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    )
    tickers = [t for t in tickers if t]
    seen = set()
    ordered = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered

def read_ticker_index_map(path: str, sheet_name: str = "Selected_Stocks") -> Dict[str, str]:
    try:
        df = pd.read_excel(path, sheet_name=sheet_name)
    except Exception as e:
        log.warning("Could not read '%s' sheet for index mapping: %s", sheet_name, e)
        return {}

    if df.empty or df.shape[1] < 1:
        return {}

    ticker_col = df.columns[0]
    index_col = next((c for c in df.columns[1:] if str(c).strip().upper() == "INDEX"), None)
    if index_col is None:
        log.warning("No 'INDEX' column found in '%s'", sheet_name)
        return {}

    mapping = {}
    for _, row in df.iterrows():
        raw = row.get(ticker_col)
        if pd.isna(raw):
            continue
        ticker = str(raw).strip().upper()
        if not ticker:
            continue
        idx_val = row.get(index_col)
        mapping[ticker] = str(idx_val).strip().upper() if not pd.isna(idx_val) else "UNINDEX"
    return mapping

def to_yf_ticker(raw: str) -> str:
    raw = raw.strip().upper()
    if raw.endswith(EGX_SUFFIX):
        return raw
    return f"{raw}{EGX_SUFFIX}"

# ─────────────────────────────────────────────────────────────
# yfinance Download
# ─────────────────────────────────────────────────────────────
def download_all(tickers: List[str], cache: Dict[str, TickerData]) -> None:
    for raw in tickers:
        if raw in cache:
            continue
        yf_ticker = to_yf_ticker(raw)
        entry = TickerData(raw_ticker=raw, yf_ticker=yf_ticker)

        try:
            hist = yf.download(
                yf_ticker,
                period=HISTORY_PERIOD,
                interval="1d",
                auto_adjust=AUTO_ADJUST,
                progress=False,
            )
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)

            if hist.empty:
                entry.reason = "no data returned"
                log.warning("Skipping %s: %s", raw, entry.reason)
            else:
                hist = hist.dropna(subset=["Close"])
                if len(hist) < MIN_TRADING_DAYS:
                    entry.reason = f"insufficient history ({len(hist)} bars)"
                    log.warning("Skipping %s: %s", raw, entry.reason)
                else:
                    entry.history = hist
                    entry.ok = True

        except Exception as e:
            entry.reason = f"download error: {e}"
            log.warning("Skipping %s: %s", raw, entry.reason)

        cache[raw] = entry

# ─────────────────────────────────────────────────────────────
# TradingView Batch Fetching
# ─────────────────────────────────────────────────────────────
def chunk_list(items: List[str], size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]

def fetch_ta_chunk_raw(symbols: List[str]) -> Dict[str, Optional[dict]]:
    scan_url = f"{TradingView.scan_url}{TA_SCREENER.lower()}/scan"
    tv_symbols = [f"{TA_EXCHANGE}:{s}" for s in symbols]
    indicator_columns = TradingView.indicators + TA_EXTRA_COLUMNS
    payload = {
        "symbols": {"tickers": [s.upper() for s in tv_symbols], "query": {"types": []}},
        "columns": indicator_columns,
    }
    headers = {"User-Agent": TA_USER_AGENT}
    response = requests.post(scan_url, json=payload, headers=headers, timeout=30)

    if response.status_code == 429:
        raise RuntimeError("429 rate limit")
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")

    data = response.json().get("data", [])
    by_symbol = {item["s"].upper(): item["d"] for item in data}

    result: Dict[str, Optional[dict]] = {}
    for raw, tv_symbol in zip(symbols, tv_symbols):
        row = by_symbol.get(tv_symbol.upper())
        result[raw] = dict(zip(indicator_columns, row)) if row is not None else None
    return result

def fetch_ta_batch(tickers: List[str]) -> Dict[str, TickerTA]:
    cache: Dict[str, TickerTA] = {}
    chunks = list(chunk_list(tickers, TA_SYMBOLS_PER_REQUEST))
    total_chunks = len(chunks)

    for idx, chunk in enumerate(chunks):
        chunk_result: Optional[Dict[str, Optional[dict]]] = None
        fetch_time = None

        for attempt in range(TA_BATCH_RETRIES):
            try:
                chunk_result = fetch_ta_chunk_raw(chunk)
                fetch_time = datetime.now()
                break
            except Exception as e:
                reason = str(e)
                is_last_attempt = attempt == TA_BATCH_RETRIES - 1
                if "429" in reason and not is_last_attempt:
                    wait = TA_BATCH_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 2)
                    log.warning(
                        "TA batch %d/%d: 429 rate limit, retrying in %.1fs (attempt %d/%d)",
                        idx + 1, total_chunks, wait, attempt + 1, TA_BATCH_RETRIES
                    )
                    time.sleep(wait)
                    continue
                log.warning("TA batch %d/%d failed: %s", idx + 1, total_chunks, reason)
                chunk_result = None
                break

        for raw in chunk:
            entry = TickerTA(raw_ticker=raw)
            indicators = chunk_result.get(raw) if chunk_result else None
            if indicators and indicators.get("close") is not None:
                entry.indicators = indicators
                entry.ok = True
                entry.fetch_time = fetch_time
                log.info("%s: TA data loaded (batch %d/%d)", raw, idx + 1, total_chunks)
            else:
                entry.reason = "no TA data in batch response"
                log.warning("%s: TA fetch failed - %s", raw, entry.reason)
            cache[raw] = entry

        if idx < total_chunks - 1:
            pause = random.uniform(TA_INTER_CHUNK_DELAY_MIN, TA_INTER_CHUNK_DELAY_MAX)
            log.info(
                "Processed chunk %d/%d (%d symbols), pausing %.1fs before next chunk...",
                idx + 1, total_chunks, len(chunk), pause
            )
            time.sleep(pause)

    return cache

def fetch_all_ta(tickers: List[str], cache: Dict[str, TickerTA]) -> None:
    to_fetch = [t for t in tickers if t not in cache]
    if not to_fetch:
        return
    cache.update(fetch_ta_batch(to_fetch))

# ─────────────────────────────────────────────────────────────
# Index Sentiment Analysis
# ─────────────────────────────────────────────────────────────
INDEX_SYMBOLS = {"EGX30": "EGX30", "EGX70": "EGX70EWI", "EGX100": "EGX100EWI"}

def fetch_index_sentiment() -> Dict[str, Dict]:
    """Fetch TA for EGX30, EGX70, EGX100 and classify as Bullish/Bearish/Neutral."""
    symbols = list(INDEX_SYMBOLS.values())
    scan_url = f"{TradingView.scan_url}{TA_SCREENER.lower()}/scan"
    tv_symbols = [f"{TA_EXCHANGE}:{s}" for s in symbols]
    columns = ["close", "RSI", "MACD.macd", "MACD.signal", "ADX", "ADX+DI", "ADX-DI",
               "EMA20", "EMA50", "SMA50", "SMA200", "recommendation_all"]

    payload = {
        "symbols": {"tickers": [s.upper() for s in tv_symbols], "query": {"types": []}},
        "columns": columns,
    }
    headers = {"User-Agent": TA_USER_AGENT}

    try:
        response = requests.post(scan_url, json=payload, headers=headers, timeout=15)
        if response.status_code != 200:
            log.warning("Index TA fetch failed: HTTP %d", response.status_code)
            return {}
        data = response.json().get("data", [])
    except Exception as e:
        log.warning("Index TA fetch error: %s", e)
        return {}

    by_symbol = {item["s"].upper(): dict(zip(columns, item["d"])) for item in data}

    result = {}
    for label, tv_sym in INDEX_SYMBOLS.items():
        tv_key = f"{TA_EXCHANGE}:{tv_sym}".upper()
        ind = by_symbol.get(tv_key)
        if not ind or ind.get("close") is None:
            result[label] = {"close": None, "sentiment": "Unknown", "rsi": None, "reason": "no data"}
            continue

        close = ind["close"]
        rsi = ind.get("RSI")
        macd = ind.get("MACD.macd")
        macd_sig = ind.get("MACD.signal")
        adx = ind.get("ADX")
        adx_plus = ind.get("ADX+DI")
        adx_minus = ind.get("ADX-DI")
        ema20 = ind.get("EMA20")
        ema50 = ind.get("EMA50")
        sma50 = ind.get("SMA50")
        sma200 = ind.get("SMA200")

        # Classify sentiment
        bullish = 0
        bearish = 0
        reasons = []

        # RSI
        if rsi is not None:
            if rsi > 55:
                bullish += 1
                reasons.append(f"RSI {rsi:.0f} > 55")
            elif rsi < 45:
                bearish += 1
                reasons.append(f"RSI {rsi:.0f} < 45")
            else:
                reasons.append(f"RSI {rsi:.0f} neutral")

        # MACD
        if macd is not None and macd_sig is not None:
            if macd > macd_sig:
                bullish += 1
                reasons.append("MACD bullish crossover")
            else:
                bearish += 1
                reasons.append("MACD bearish")

        # Trend (EMA/SMA alignment)
        if close and ema20 and ema50:
            if close > ema20 > ema50:
                bullish += 1
                reasons.append("Price > EMA20 > EMA50 (uptrend)")
            elif close < ema20 < ema50:
                bearish += 1
                reasons.append("Price < EMA20 < EMA50 (downtrend)")

        if sma50 and sma200:
            if sma50 > sma200:
                bullish += 1
                reasons.append("Golden cross (SMA50 > SMA200)")
            else:
                bearish += 1
                reasons.append("Death cross (SMA50 < SMA200)")

        # ADX strength
        if adx is not None and adx > 25:
            reasons.append(f"Strong trend (ADX {adx:.0f})")
        elif adx is not None:
            reasons.append(f"Weak trend (ADX {adx:.0f})")

        # Final classification
        if bullish > bearish + 1:
            sentiment = "Bullish"
        elif bearish > bullish + 1:
            sentiment = "Bearish"
        else:
            sentiment = "Neutral"

        result[label] = {
            "close": close,
            "sentiment": sentiment,
            "rsi": rsi,
            "adx": adx,
            "bullish_score": bullish,
            "bearish_score": bearish,
            "reasons": reasons,
        }
        log.info("Index %s: %s (close=%.1f, RSI=%.1f, ADX=%.1f)", label, sentiment, close or 0, rsi or 0, adx or 0)

    return result

# ─────────────────────────────────────────────────────────────
# FX Rate
# ─────────────────────────────────────────────────────────────
def download_fx(period: str = HISTORY_PERIOD) -> Optional[pd.Series]:
    try:
        fx = yf.download(FX_TICKER, period=period, interval="1d",
                          auto_adjust=AUTO_ADJUST, progress=False)
        if isinstance(fx.columns, pd.MultiIndex):
            fx.columns = fx.columns.get_level_values(0)
        if fx.empty:
            log.warning("No FX data returned for %s", FX_TICKER)
            return None
        return fx["Close"].dropna()
    except Exception as exc:
        log.warning("Could not download FX rate %s: %s", FX_TICKER, exc)
        return None

# ─────────────────────────────────────────────────────────────
# Fundamentals
# ─────────────────────────────────────────────────────────────
def fetch_fundamentals(raw_ticker: str, yf_ticker: str, current_egp: float) -> Optional[dict]:
    try:
        ticker_obj = yf.Ticker(yf_ticker)
        info = ticker_obj.info
        if not info:
            return None

        eps = info.get("trailingEps")
        book_value = info.get("bookValue")
        pe_ratio = info.get("trailingPE")
        sector = info.get("sector", "Unknown")

        if eps is None and pe_ratio is not None and pe_ratio > 0 and current_egp is not None:
            eps = current_egp / pe_ratio
            log.debug("%s: Derived EPS %.4f from price %.2f / PE %.2f",
                      raw_ticker, eps, current_egp, pe_ratio)

        if eps is None or eps <= 0:
            log.debug("%s: No usable EPS (raw=%s, derived=%s)", raw_ticker,
                      info.get("trailingEps"), eps)
            return None

        return {
            "eps": float(eps),
            "book_value": float(book_value) if book_value is not None else None,
            "pe_ratio": float(pe_ratio) if pe_ratio is not None else None,
            "sector": sector,
        }
    except Exception as e:
        log.debug("%s: Could not fetch fundamentals: %s", raw_ticker, e)
        return None

def compute_fair_value(fundamentals: Optional[dict], current_egp: Optional[float]) -> Tuple[Optional[float], str]:
    if fundamentals is None:
        return None, "N/A"

    eps = fundamentals.get("eps")
    book_value = fundamentals.get("book_value")

    if eps is None or eps <= 0:
        return None, "N/A (negative earnings)"

    graham_value = None
    pe_value = None

    if book_value is not None and book_value > 0:
        graham_value = (params["fundamentals"]["graham_multiplier"] * eps * book_value) ** 0.5

    pe_value = eps * REFERENCE_PE_EGX

    if graham_value is not None:
        fair_value = (graham_value + pe_value) / 2
        method = "Blended (Graham+PE)"
    else:
        fair_value = pe_value
        method = "PE-Based"

    if fair_value <= 0:
        return None, "N/A (negative fair value)"

    return round(fair_value, 4), method

# ─────────────────────────────────────────────────────────────
# USD Valuation + Fair Value
# ─────────────────────────────────────────────────────────────
def usd_valuation(
    raw_ticker: str,
    cache: Dict[str, TickerData],
    fx_series: Optional[pd.Series],
    ta_cache: Optional[Dict[str, TickerTA]] = None,
) -> dict:
    result = {
        "current_egp": None, "current_usd": None,
        "hist_min_usd": None, "hist_max_usd": None,
        "undervalued": "No",
        "implied_fair_value_egp": None,
        "fair_value_method": "N/A",
        "pe_ratio_ttm": None,
        "eps_ttm": None,
    }

    entry = cache.get(raw_ticker)
    if entry is None or not entry.ok or fx_series is None:
        return result

    egp_close = entry.history["Close"]
    aligned = pd.concat(
        [egp_close.rename("egp"), fx_series.rename("fx")], axis=1, join="inner"
    ).dropna()
    if aligned.empty:
        return result

    aligned["usd"] = aligned["egp"] / aligned["fx"]

    current_egp = float(aligned["egp"].iloc[-1])
    current_usd = float(aligned["usd"].iloc[-1])
    current_fx = float(aligned["fx"].iloc[-1])
    hist_min_usd = float(aligned["usd"].min())
    hist_max_usd = float(aligned["usd"].max())

    result.update(current_egp=current_egp, current_usd=current_usd,
                   hist_min_usd=hist_min_usd, hist_max_usd=hist_max_usd)

    # Primary: Fundamental fair value
    fundamentals = fetch_fundamentals(raw_ticker, entry.yf_ticker, current_egp)
    fair_value, method = compute_fair_value(fundamentals, current_egp)

    if fair_value is not None:
        result["implied_fair_value_egp"] = fair_value
        result["fair_value_method"] = method
        if fundamentals:
            result["pe_ratio_ttm"] = fundamentals.get("pe_ratio")
            result["eps_ttm"] = fundamentals.get("eps")
        if current_egp < fair_value:
            result["undervalued"] = "Yes"
        log.info("%s: Fair value = %.4f (%s), current = %.4f, undervalued = %s",
                 raw_ticker, fair_value, method, current_egp, result["undervalued"])
        return result

    # Fallback: USD-comparison method
    pe_ratio = None
    eps = None
    if ta_cache:
        ta_entry = ta_cache.get(raw_ticker)
        if ta_entry and ta_entry.ok:
            ind = ta_entry.indicators
            pe_ratio = ind.get("price_earnings_ttm")
            eps = ind.get("earnings_per_share_basic_ttm")
    if eps is not None:
        result["eps_ttm"] = float(eps)
    if pe_ratio is not None:
        result["pe_ratio_ttm"] = float(pe_ratio)

    lower = current_egp * (1 - EGP_SIMILARITY_BAND)
    upper = current_egp * (1 + EGP_SIMILARITY_BAND)
    comparable = aligned[(aligned["egp"] >= lower) & (aligned["egp"] <= upper)]
    comparable = comparable.iloc[:-2] if len(comparable) > 2 else comparable

    if len(comparable) >= 10:
        historical_median_usd = float(comparable["usd"].median())
        result["implied_fair_value_egp"] = historical_median_usd * current_fx
        result["fair_value_method"] = "USD-Comparison (fallback)"
        if current_usd < historical_median_usd:
            result["undervalued"] = "Yes"
        log.info("%s: Fair value = %.4f (USD-Comparison fallback), current = %.4f",
                 raw_ticker, result["implied_fair_value_egp"], current_egp)

    return result

# ─────────────────────────────────────────────────────────────
# Volume Analysis
# ─────────────────────────────────────────────────────────────
def volume_analysis(raw_ticker: str, yf_cache: Dict[str, TickerData], ta_cache: Dict[str, TickerTA]) -> dict:
    result = {"avg_vol_1y": None, "last_day_vol": None, "vol_multiplier": None}

    ta_entry = ta_cache.get(raw_ticker)
    if ta_entry and ta_entry.ok:
        ta_volume = ta_entry.indicators.get("volume")
        if ta_volume is not None:
            result["last_day_vol"] = float(ta_volume)

    entry = yf_cache.get(raw_ticker)
    if entry is None or not entry.ok or "Volume" not in entry.history.columns:
        if result["last_day_vol"] is not None:
            return result
        return result

    vol = entry.history["Volume"].dropna()
    if vol.empty:
        return result

    last_year = vol.tail(252)
    avg_1y = float(last_year.mean()) if not last_year.empty else None

    result["avg_vol_1y"] = avg_1y

    if result["last_day_vol"] is None:
        result["last_day_vol"] = float(vol.iloc[-1])

    if avg_1y and avg_1y > 0:
        result["vol_multiplier"] = round(result["last_day_vol"] / avg_1y, 3)

    return result

# ─────────────────────────────────────────────────────────────
# Money Flow Volume (Buy Volume)
# ─────────────────────────────────────────────────────────────
def money_flow_volume_analysis(raw_ticker: str, yf_cache: Dict[str, TickerData]) -> dict:
    result = {
        "buy_vol_avg_2mo": None,
        "buy_vol_last_day": None,
        "buy_vol_multiplier": None
    }
    entry = yf_cache.get(raw_ticker)
    required = {"High", "Low", "Close", "Volume"}
    if entry is None or not entry.ok or not required.issubset(entry.history.columns):
        return result

    hist = entry.history.dropna(subset=list(required))
    if hist.empty:
        return result

    high, low, close, volume = hist["High"], hist["Low"], hist["Close"], hist["Volume"]
    day_range = high - low

    with np.errstate(divide="ignore", invalid="ignore"):
        mfm = ((close - low) - (high - close)) / day_range
    mfm = mfm.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-1, 1)

    buy_fraction = (mfm + 1) / 2
    est_buy_vol = volume * buy_fraction

    buy_recent = est_buy_vol.tail(BUY_VOL_AVG_DAYS)
    if len(buy_recent) < BUY_VOL_AVG_DAYS // 2:
        return result

    avg_buy = float(buy_recent.mean())
    last_buy = float(est_buy_vol.iloc[-1])

    result["buy_vol_avg_2mo"] = avg_buy
    result["buy_vol_last_day"] = last_buy
    if avg_buy > 0:
        result["buy_vol_multiplier"] = round(last_buy / avg_buy, 3)

    return result

# ─────────────────────────────────────────────────────────────
# Support & Resistance
# ─────────────────────────────────────────────────────────────
def find_swing_points(prices: pd.Series, order: int = SWING_ORDER):
    values = prices.values
    n = len(values)
    swing_highs, swing_lows = [], []

    for i in range(order, n - order):
        window = values[i - order: i + order + 1]
        if values[i] == window.max() and np.argmax(window) == order:
            swing_highs.append((i, values[i]))
        if values[i] == window.min() and np.argmin(window) == order:
            swing_lows.append((i, values[i]))

    return swing_highs, swing_lows

def support_resistance(raw_ticker: str, yf_cache: Dict[str, TickerData]) -> dict:
    result = {"support": None, "resistance": None}
    entry = yf_cache.get(raw_ticker)
    if entry is None or not entry.ok:
        return result

    hist = entry.history.tail(SR_LOOKBACK_DAYS)
    if len(hist) < (2 * SWING_ORDER + 1):
        return result

    highs = hist["High"] if "High" in hist.columns else hist["Close"]
    lows = hist["Low"] if "Low" in hist.columns else hist["Close"]

    swing_highs, _ = find_swing_points(highs, SWING_ORDER)
    _, swing_lows = find_swing_points(lows, SWING_ORDER)

    current_price = float(hist["Close"].iloc[-1])

    above = [v for _, v in swing_highs if v > current_price]
    resistance = min(above) if above else None

    below = [v for _, v in swing_lows if v < current_price]
    support = max(below) if below else None

    result["support"] = float(support) if support is not None else None
    result["resistance"] = float(resistance) if resistance is not None else None
    return result

# ─────────────────────────────────────────────────────────────
# ADL, MFI, BB Squeeze
# ─────────────────────────────────────────────────────────────
def _adl_series(history: pd.DataFrame) -> Optional[np.ndarray]:
    if history is None or len(history) < 5:
        return None
    try:
        high = history["High"].values.astype(float)
        low = history["Low"].values.astype(float)
        close = history["Close"].values.astype(float)
        vol = history["Volume"].values.astype(float)
        hl_range = high - low
        hl_range[hl_range == 0] = 1e-10
        clv = ((close - low) - (high - close)) / hl_range
        return np.cumsum(clv * vol)
    except Exception:
        return None

def compute_adl(history: pd.DataFrame) -> Optional[float]:
    adl = _adl_series(history)
    if adl is None:
        return None
    return float(adl[-1])

def compute_adl_trend(history: pd.DataFrame, window: int = ADL_WINDOW) -> Optional[float]:
    adl = _adl_series(history)
    if adl is None or len(adl) < window + 1:
        return None
    return float(adl[-1] - adl[-1 - window])

def compute_mfi(history: pd.DataFrame, period: int = 14) -> Optional[float]:
    if history is None or len(history) < period + 2:
        return None
    try:
        tp = ((history["High"] + history["Low"] + history["Close"]) / 3).values
        vol = history["Volume"].values.astype(float)
        raw_mf = tp * vol
        pos_flow = 0.0
        neg_flow = 0.0
        for i in range(1, len(tp)):
            if tp[i] > tp[i - 1]:
                pos_flow += raw_mf[i]
            elif tp[i] < tp[i - 1]:
                neg_flow += raw_mf[i]
        if neg_flow == 0:
            return 100.0
        mfr = pos_flow / neg_flow
        mfi = 100.0 - (100.0 / (1.0 + mfr))
        return float(mfi)
    except Exception:
        return None

def compute_bb_squeeze(bb_lower: float, bb_upper: float, atr: float) -> Optional[bool]:
    if bb_lower is None or bb_upper is None or atr is None or atr <= 0:
        return None
    bb_width = bb_upper - bb_lower
    return bb_width < (0.5 * atr)

# ─────────────────────────────────────────────────────────────
# Fallback Indicators (yfinance)
# ─────────────────────────────────────────────────────────────
SMA_SHORT_WINDOW = 50
SMA_LONG_WINDOW = 200
EMA_XSHORT_WINDOW = 20
EMA_SHORT_WINDOW = 50
EMA_LONG_WINDOW = 200
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_PERIOD = data_cfg["rsi_period"]

def compute_sma_ema_rsi_from_yf(history: pd.DataFrame) -> dict:
    close = history["Close"].dropna()
    result = {
        "sma20": None, "sma50": None, "sma200": None, "ema20": None, "ema50": None, "ema200": None,
        "rsi": None, "macd": None, "macd_signal": None,
    }

    if len(close) >= EMA_XSHORT_WINDOW:
        result["ema20"] = float(close.ewm(span=EMA_XSHORT_WINDOW, adjust=False).mean().iloc[-1])
        result["sma20"] = float(close.rolling(EMA_XSHORT_WINDOW).mean().iloc[-1])

    if len(close) >= SMA_SHORT_WINDOW:
        result["sma50"] = float(close.rolling(SMA_SHORT_WINDOW).mean().iloc[-1])
        result["ema50"] = float(close.ewm(span=EMA_SHORT_WINDOW, adjust=False).mean().iloc[-1])

    if len(close) >= SMA_LONG_WINDOW:
        result["sma200"] = float(close.rolling(SMA_LONG_WINDOW).mean().iloc[-1])
        result["ema200"] = float(close.ewm(span=EMA_LONG_WINDOW, adjust=False).mean().iloc[-1])

    if len(close) >= RSI_PERIOD + 1:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        last_rsi = rsi.iloc[-1]
        if not pd.isna(last_rsi):
            result["rsi"] = float(last_rsi)

    if len(close) >= MACD_SLOW + MACD_SIGNAL:
        ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
        ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
        result["macd"] = float(macd_line.iloc[-1])
        result["macd_signal"] = float(signal_line.iloc[-1])

    return result