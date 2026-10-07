"""
Base Recommendation Scoring for EGX Signals
6-category weighted scoring system
"""
from typing import Dict, List, Tuple, Optional
import pandas as pd
from src.config import load_params

params = load_params()
weights_cfg = params["weights"]
thresholds_cfg = params["thresholds"]
rsi_cfg = params["rsi"]
volume_cfg = params["volume"]
trade_cfg = params["trade"]
data_cfg = params["data"]

# ─────────────────────────────────────────────────────────────
# Weights & Thresholds
# ─────────────────────────────────────────────────────────────
SCORE_WEIGHT_TREND = weights_cfg["trend"]
SCORE_WEIGHT_MACD = weights_cfg["macd"]
SCORE_WEIGHT_RSI = weights_cfg["rsi"]
SCORE_WEIGHT_VOLUME = weights_cfg["volume"]
SCORE_WEIGHT_ADI = weights_cfg["adi"]
SCORE_WEIGHT_SUPPORT = weights_cfg["support"]
SCORE_WEIGHT_VWAP = weights_cfg.get("vwap", 5)
SCORE_WEIGHT_VP = weights_cfg.get("volume_profile", 5)
SCORE_WEIGHT_INTRADAY = weights_cfg.get("intraday", 0)

SCORE_BUY_THRESHOLD = thresholds_cfg["buy"]
SCORE_WATCH_THRESHOLD = thresholds_cfg["watch"]

# Fallbacks only — regime zones from params.yaml take priority in score_rsi.
# Derived from params rsi.ranging defaults so config and code never diverge.
_RSI_FALLBACK = params.get("rsi", {}).get("ranging", {})
RSI_OVERSOLD = _RSI_FALLBACK.get("oversold", 28)
RSI_OVERBOUGHT = _RSI_FALLBACK.get("overbought", 70)
ADX_TREND_THRESHOLD = data_cfg["adx_trend_threshold"]
MFI_OVERBOUGHT = data_cfg["mfi_overbought"]
MFI_OVERSOLD = data_cfg["mfi_oversold"]
VOLUME_SPIKE_MULTIPLIER = volume_cfg["spike_multiplier"]
NEAR_SUPPORT_PCT = volume_cfg["near_support_pct"]
ADL_WINDOW = data_cfg["adl_window"]

# ─────────────────────────────────────────────────────────────
# Regime → RSI Config Mapping
# ─────────────────────────────────────────────────────────────
def resolve_rsi_regime(regime: str) -> str:
    """Map detector regime labels onto params.yaml rsi zones.

    detect_trend_regime can emit 'trending' (ADX strong, DI neutral) and
    'transitioning', which are not keys in rsi_cfg. A strong-ADX 'trending'
    regime with price above the 50-MA is an uptrend structure — use
    trending_up bands, not ranging.
    """
    if not regime:
        return "ranging"
    if regime in rsi_cfg:
        return regime
    if regime == "trending":
        return "trending_up"
    # transitioning / unknown → ranging
    return "ranging"


# ─────────────────────────────────────────────────────────────
# Scoring Functions
# ─────────────────────────────────────────────────────────────
def score_trend(current_price: Optional[float], ema20: Optional[float],
                ema50: Optional[float], ema200: Optional[float]) -> Tuple[float, List[str]]:
    reasons = []
    if current_price is None or ema50 is None or ema200 is None:
        return 0.0, reasons

    score = 0.0
    ema_bullish = ema50 > ema200
    price_above_50 = current_price > ema50
    price_above_200 = current_price > ema200

    if price_above_50 and ema_bullish:
        score += 0.55
        reasons.append("price > EMA50 > EMA200 (strong bullish alignment)")
    elif price_above_200 and ema_bullish:
        score += 0.35
        reasons.append("EMA50 > EMA200, price above EMA200")
    elif ema_bullish:
        score += 0.15
        reasons.append("EMA50 > EMA200 but price below EMA50")
    else:
        reasons.append("EMA50 <= EMA200 (bearish structure)")

    if ema20 is not None and ema20 > ema50:
        score += 0.20
        reasons.append("Diamond Cross (EMA20>EMA50)")

    if price_above_200:
        distance_pct = (current_price - ema200) / ema200
        slope_bonus = min(0.25, max(0.0, distance_pct * 1.0))
        if slope_bonus > 0:
            score += slope_bonus
            reasons.append(f"price {distance_pct * 100:.1f}% above EMA200")

    return min(1.0, max(0.0, score)), reasons

def score_macd(macd: Optional[float], macd_signal: Optional[float]) -> Tuple[float, List[str]]:
    reasons = []
    if macd is None or macd_signal is None:
        return 0.0, reasons

    score = 0.0
    if macd > macd_signal:
        hist = macd - macd_signal
        mag = abs(macd_signal) if macd_signal != 0 else 1.0
        strength = min(1.0, abs(hist) / mag)
        score = 0.4 + 0.6 * strength
        reasons.append(f"MACD bullish (strength {strength:.0%})")
        if macd > 0:
            score = min(1.0, score + 0.1)
            reasons.append("MACD above zero (confirmed momentum)")
    else:
        reasons.append("MACD below signal (bearish)")

    return min(1.0, max(0.0, score)), reasons

def score_rsi(rsi: Optional[float], adx: Optional[float], regime: str = "unknown") -> Tuple[float, List[str]]:
    reasons = []
    if rsi is None:
        return 0.0, reasons

    # Get regime-specific RSI thresholds (map ambiguous regime labels)
    regime_key = resolve_rsi_regime(regime)
    regime_rsi = rsi_cfg.get(regime_key, rsi_cfg.get("ranging", {}))
    oversold = regime_rsi.get("oversold", RSI_OVERSOLD)
    healthy_low = regime_rsi.get("healthy_low", 40)
    healthy_high = regime_rsi.get("healthy_high", 70)
    overbought = regime_rsi.get("overbought", RSI_OVERBOUGHT)

    if healthy_low <= rsi <= healthy_high:
        score = 1.0
        reasons.append(f"RSI in healthy bullish zone ({rsi:.1f})")
    elif oversold <= rsi < healthy_low:
        score = 0.6
        reasons.append(f"RSI neutral-firm ({rsi:.1f})")
    elif rsi < oversold:
        score = 0.8
        reasons.append(f"RSI oversold, potential reversal ({rsi:.1f})")
    elif rsi <= overbought:
        strong_trend = adx is not None and adx >= ADX_TREND_THRESHOLD
        score = 0.5 if strong_trend else 0.25
        note = " (strong trend maintained)" if strong_trend else ""
        reasons.append(f"RSI mildly overbought ({rsi:.1f}){note}")
    else:
        strong_trend = adx is not None and adx >= ADX_TREND_THRESHOLD
        score = 0.35 if strong_trend else 0.1
        note = " (strong trend, penalty reduced)" if strong_trend else ""
        reasons.append(f"RSI overbought ({rsi:.1f}){note}")

    return score, reasons

def score_volume(vol_multiplier: Optional[float], buy_vol_multiplier: Optional[float]) -> Tuple[float, List[str]]:
    reasons = []
    score = 0.0

    if vol_multiplier is not None:
        if vol_multiplier >= 3.0:
            vol_score = 1.0
        elif vol_multiplier >= 2.0:
            vol_score = 0.6 + 0.4 * (vol_multiplier - 2.0)
        elif vol_multiplier >= VOLUME_SPIKE_MULTIPLIER:
            vol_score = 0.3 + 0.3 * (vol_multiplier - VOLUME_SPIKE_MULTIPLIER) / (2.0 - VOLUME_SPIKE_MULTIPLIER)
        elif vol_multiplier >= 1.0:
            vol_score = 0.1 * (vol_multiplier - 1.0) / (VOLUME_SPIKE_MULTIPLIER - 1.0)
        else:
            vol_score = 0.0
        score += vol_score
        if vol_multiplier >= VOLUME_SPIKE_MULTIPLIER:
            reasons.append(f"volume {vol_multiplier:.2f}x average")

    if buy_vol_multiplier is not None:
        if buy_vol_multiplier >= 3.0:
            buy_score = 1.0
        elif buy_vol_multiplier >= VOLUME_SPIKE_MULTIPLIER:
            buy_score = 0.3 + 0.7 * (buy_vol_multiplier - VOLUME_SPIKE_MULTIPLIER) / (3.0 - VOLUME_SPIKE_MULTIPLIER)
        elif buy_vol_multiplier >= 1.0:
            buy_score = 0.1 * (buy_vol_multiplier - 1.0) / (VOLUME_SPIKE_MULTIPLIER - 1.0)
        else:
            buy_score = 0.0
        score += buy_score
        if buy_vol_multiplier >= VOLUME_SPIKE_MULTIPLIER:
            reasons.append(f"buy-volume spike ({buy_vol_multiplier:.2f}x)")

    return min(1.0, max(0.0, score)), reasons

def score_adi(adl_trend: Optional[float], mfi: Optional[float]) -> Tuple[float, List[str]]:
    reasons = []
    score = 0.0

    if adl_trend is not None:
        if adl_trend > 0:
            score += 0.5
            reasons.append(f"accumulation (ADL rising over last {ADL_WINDOW}d)")
        elif adl_trend < 0:
            score -= 0.5
            reasons.append(f"distribution (ADL falling over last {ADL_WINDOW}d)")

    if mfi is not None:
        if mfi <= MFI_OVERSOLD:
            score += 0.3
            reasons.append(f"MFI oversold ({mfi:.1f})")
        elif mfi >= MFI_OVERBOUGHT:
            score -= 0.3
            reasons.append(f"MFI overbought ({mfi:.1f})")
        elif 40 <= mfi <= 60:
            score += 0.1

    return min(1.0, max(0.0, score)), reasons

def score_support(is_near_support: bool, volume_confirmed: bool,
                  current_price: Optional[float] = None, support: Optional[float] = None) -> Tuple[float, List[str]]:
    if not is_near_support:
        return 0.0, []

    if current_price is not None and support is not None and support > 0:
        distance_pct = (current_price - support) / support
        proximity_score = max(0.3, 1.0 - (distance_pct / NEAR_SUPPORT_PCT) * 0.7)
    else:
        proximity_score = 0.6

    if volume_confirmed:
        proximity_score = min(1.0, proximity_score + 0.2)
        return proximity_score, ["support bounce confirmed by volume"]

    return proximity_score, ["price near support"]

def score_vwap(dist_vwap: Optional[float], vwap: Optional[float],
               current_price: Optional[float] = None,
               adx: Optional[float] = None,
               ema_bullish: Optional[bool] = None) -> Tuple[float, List[str]]:
    reasons = []
    if dist_vwap is None or vwap is None or vwap <= 0:
        return 0.0, reasons

    # In a confirmed uptrend, trading above VWAP is participation, not a
    # mean-reversion sell signal. Only penalise truly extreme extensions.
    strong_trend = (
        (adx is not None and adx >= ADX_TREND_THRESHOLD)
        or ema_bullish is True
    )

    score = 0.0
    pct = dist_vwap * 100

    if pct > 0:
        if pct <= 2.0:
            score = 0.6
            reasons.append(f"price {pct:.1f}% above VWAP (healthy premium)")
        elif pct <= 5.0:
            score = 0.8
            reasons.append(f"price {pct:.1f}% above VWAP (strong momentum)")
        elif pct <= 8.0:
            if strong_trend:
                score = 0.65
                reasons.append(f"price {pct:.1f}% above VWAP (trend extension)")
            else:
                score = 0.3
                reasons.append(f"price {pct:.1f}% above VWAP (extended, mean-reversion risk)")
        else:
            if strong_trend:
                score = 0.45
                reasons.append(f"price {pct:.1f}% above VWAP (deep extension, trend intact)")
            else:
                score = 0.3
                reasons.append(f"price {pct:.1f}% above VWAP (extended, mean-reversion risk)")
    else:
        abs_pct = abs(pct)
        if abs_pct <= 2.0:
            score = 0.4
            reasons.append(f"price {abs_pct:.1f}% below VWAP (approaching fair value)")
        elif abs_pct <= 5.0:
            score = 0.2
            reasons.append(f"price {abs_pct:.1f}% below VWAP (weak)")
        else:
            score = 0.1
            reasons.append(f"price {abs_pct:.1f}% below VWAP (deep discount)")

    return score, reasons

def score_volume_profile(above_poc: Optional[bool], poc: Optional[float],
                         va_high: Optional[float], va_low: Optional[float],
                         current_price: Optional[float] = None,
                         strong_trend: bool = False) -> Tuple[float, List[str]]:
    reasons = []
    if poc is None or above_poc is None:
        return 0.0, reasons

    score = 0.0
    if above_poc:
        if current_price is not None and va_high is not None and current_price <= va_high:
            score = 0.8
            reasons.append(f"price above POC ({poc:.3f}), within value area")
        elif current_price is not None and va_high is not None and current_price > va_high:
            # Breaking out above the value area — bullish when trend is confirmed
            if strong_trend:
                score = 0.7
                reasons.append(f"price breaking above value area ({va_high:.3f}) in strong trend")
            else:
                score = 0.5
                reasons.append(f"price above POC ({poc:.3f}), outside value area")
        else:
            score = 0.5
            reasons.append(f"price above POC ({poc:.3f})")
    else:
        if current_price is not None and va_low is not None and current_price >= va_low:
            score = 0.6
            reasons.append(f"price below POC ({poc:.3f}), near value area low (potential support)")
        else:
            score = 0.2
            reasons.append(f"price below POC ({poc:.3f}), below value area")

    return score, reasons

# ─────────────────────────────────────────────────────────────
# Master Scoring Function
# ─────────────────────────────────────────────────────────────
def compute_base_score(current_price: Optional[float],
                       ema20: Optional[float], ema50: Optional[float], ema200: Optional[float],
                       macd: Optional[float], macd_signal: Optional[float],
                       rsi: Optional[float], adx: Optional[float], regime: str,
                       vol_multiplier: Optional[float], buy_vol_multiplier: Optional[float],
                       adl_trend: Optional[float], mfi: Optional[float],
                       is_near_support: bool, volume_confirmed: bool,
                       support: Optional[float],
                       dist_vwap: Optional[float] = None, vwap: Optional[float] = None,
                       above_poc: Optional[bool] = None, poc: Optional[float] = None,
                       va_high: Optional[float] = None, va_low: Optional[float] = None,
                       intraday_score: Optional[float] = None,
                       intraday_details: Optional[Dict] = None) -> Dict:
    """
    Compute the 0-100 base score from all categories (including intraday).
    Returns dict with score, breakdown, and recommendation.
    """
    reasons = []

    ema_bullish_flag = (
        ema50 is not None and ema200 is not None and ema50 > ema200
    )
    strong_trend = (
        (adx is not None and adx >= ADX_TREND_THRESHOLD) or ema_bullish_flag
    )

    # Score each category
    trend_val, trend_reasons = score_trend(current_price, ema20, ema50, ema200)
    macd_val, macd_reasons = score_macd(macd, macd_signal)
    rsi_val, rsi_reasons = score_rsi(rsi, adx, regime)
    volume_val, volume_reasons = score_volume(vol_multiplier, buy_vol_multiplier)
    adi_val, adi_reasons = score_adi(adl_trend, mfi)
    support_val, support_reasons = score_support(is_near_support, volume_confirmed, current_price, support)
    vwap_val, vwap_reasons = score_vwap(
        dist_vwap, vwap, current_price, adx=adx, ema_bullish=ema_bullish_flag
    )
    vp_val, vp_reasons = score_volume_profile(
        above_poc, poc, va_high, va_low, current_price, strong_trend=strong_trend
    )

    # Intraday score (0-100) — normalize to 0-1
    if intraday_score is not None:
        intraday_val = intraday_score / 100.0
        # Intraday reasons
        if intraday_details:
            vp = intraday_details.get("volume_profile", {})
            mom = intraday_details.get("momentum", {})
            vw = intraday_details.get("vwap", {})
            ad = intraday_details.get("accumulation_distribution", {})

            if vp.get("volume_skew", 0) > 0.2:
                reasons.append(f"Intraday volume skew bullish ({vp['volume_skew']:.2f})")
            elif vp.get("volume_skew", 0) < -0.2:
                reasons.append(f"Intraday volume skew bearish ({vp['volume_skew']:.2f})")

            if mom.get("session_trend") == "accelerating":
                reasons.append(f"Intraday momentum accelerating (RSI accel +{mom.get('rsi_acceleration', 0):.1f})")
            elif mom.get("session_trend") == "decelerating":
                reasons.append(f"Intraday momentum decelerating")

            if vw.get("vwap_trend") == "above":
                reasons.append(f"Price sustained above VWAP ({vw.get('avg_distance_from_vwap', 0):+.1f}%)")
            elif vw.get("vwap_trend") == "below":
                reasons.append(f"Price sustained below VWAP ({vw.get('avg_distance_from_vwap', 0):+.1f}%)")

            if ad.get("ad_signal") == "bullish":
                reasons.append(f"Intraday accumulation detected (MFI={ad.get('money_flow_index', 0):.0f})")
            elif ad.get("ad_signal") == "bearish":
                reasons.append(f"Intraday distribution detected (MFI={ad.get('money_flow_index', 0):.0f})")
    else:
        intraday_val = 0.5  # Neutral if not available
        reasons.append("Intraday data not available (neutral)")

    # Convert to weighted points (total: 100 points)
    trend_pts = trend_val * SCORE_WEIGHT_TREND
    macd_pts = macd_val * SCORE_WEIGHT_MACD
    rsi_pts = rsi_val * SCORE_WEIGHT_RSI
    volume_pts = volume_val * SCORE_WEIGHT_VOLUME
    adi_pts = adi_val * SCORE_WEIGHT_ADI
    support_pts = support_val * SCORE_WEIGHT_SUPPORT
    vwap_pts = vwap_val * SCORE_WEIGHT_VWAP
    vp_pts = vp_val * SCORE_WEIGHT_VP
    intraday_pts = intraday_val * SCORE_WEIGHT_INTRADAY

    raw_score = trend_pts + macd_pts + rsi_pts + volume_pts + adi_pts + support_pts + vwap_pts + vp_pts + intraday_pts
    raw_score = max(0.0, min(100.0, raw_score))

    reasons.extend(trend_reasons)
    reasons.extend(macd_reasons)
    reasons.extend(rsi_reasons)
    reasons.extend(volume_reasons)
    reasons.extend(adi_reasons)
    reasons.extend(support_reasons)
    reasons.extend(vwap_reasons)
    reasons.extend(vp_reasons)

    score_breakdown = {
        "trend": round(trend_pts, 2),
        "macd": round(macd_pts, 2),
        "rsi": round(rsi_pts, 2),
        "volume": round(volume_pts, 2),
        "adi": round(adi_pts, 2),
        "support": round(support_pts, 2),
        "vwap": round(vwap_pts, 2),
        "volume_profile": round(vp_pts, 2),
        "intraday": round(intraday_pts, 2),
    }

    # Recommendation
    # Death cross overrides to Avoid (handled in caller)
    if raw_score >= SCORE_BUY_THRESHOLD:
        recommendation = "Buy"
    elif raw_score >= SCORE_WATCH_THRESHOLD:
        recommendation = "Watch"
    else:
        recommendation = "Avoid"

    basis = ", ".join(reasons) if reasons else "no signals triggered"
    basis = f"{basis}; score={raw_score:.1f}/100"

    return {
        "recommendation": recommendation,
        "recommendation_basis": basis,
        "raw_score": round(raw_score, 2),
        "score_breakdown": score_breakdown,
    }