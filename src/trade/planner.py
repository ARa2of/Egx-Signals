"""
Trade Planner for EGX Signals
Migrated from V12 tool - ATR-based entry, stop, targets, sizing
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from src.config import load_params
from src.signals.ml import compute_ml_conviction

params = load_params()
trade_cfg = params["trade"]
enhanced_cfg = params["enhanced_entry"]

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
ATR_STOP_MULTIPLIER = trade_cfg["stop_loss_atr"]
ATR_ZONE_HALF = trade_cfg["atr_zone_half"]
MAX_STOP_ATR = trade_cfg["max_stop_atr"]
MAX_WAIT_ATR = trade_cfg["max_wait_atr"]
MIN_RR = trade_cfg["min_rr"]
PROXIMITY_PENALTY_PER_ATR = trade_cfg["proximity_penalty_per_atr"]
TARGET_ATR = trade_cfg["target_atr"]

# Enhanced entry indicators
ENTRY_INDICATORS = enhanced_cfg["entry_indicators"]
TP_INDICATORS = enhanced_cfg["tp_indicators"]
SL_INDICATORS = enhanced_cfg["sl_indicators"]

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _get_indicators(df_tech: pd.DataFrame) -> Dict:
    row = df_tech.dropna(subset=["Close"]).iloc[-1]
    return {
        "close": float(row["Close"]),
        "sma50": float(row.get("SMA_50", row["Close"])),
        "sma200": float(row.get("SMA_200", row["Close"])),
        "ema20": float(row.get("EMA_20", row["Close"])),
        "rsi": float(row.get("RSI", 50.0)),
        "macd": float(row.get("MACD", 0.0)),
        "macd_sig": float(row.get("MACD_sig", 0.0)),
        "bb_mid": float(row.get("BB_mid", row["Close"])),
        "bb_up": float(row.get("BB_up", row["Close"] * 1.02)),
        "bb_dn": float(row.get("BB_dn", row["Close"] * 0.98)),
        "bb_pct": float(row.get("BB_pct", 0.5)),
        "atr": float(df_tech["ATR"].dropna().iloc[-1] if "ATR" in df_tech.columns else row["Close"] * 0.02),
        "vwap": float(row.get("VWAP", row["Close"])) if "VWAP" in df_tech.columns and not pd.isna(row.get("VWAP")) else None,
    }

def _nearest_resistance_above(price: float, sr: Dict) -> Optional[float]:
    above = [r["price"] for r in sr.get("resistance", []) if r["price"] > price]
    return min(above) if above else None

def _stop_below(candidate: float, atr: float, sr: Dict) -> float:
    atr_stop = candidate - ATR_STOP_MULTIPLIER * atr
    max_stop = candidate - MAX_STOP_ATR * atr

    strong = [s for s in sr.get("support", [])
              if s["price"] < candidate and s["price"] > max_stop
              and s["strength"] in ("Strong", "Medium")]

    if strong:
        best_sup = max(strong, key=lambda s: s["price"])["price"]
        sup_stop = best_sup * 0.992
        return max(sup_stop, atr_stop)

    return atr_stop

# ─────────────────────────────────────────────────────────────
# Candidate Scoring
# ─────────────────────────────────────────────────────────────
def _score_candidate(candidate_price: float, indic: Dict, sr: Dict, fib_levels: List,
                     forecast: Dict, candle_score_delta: int = 0, ml_conviction: Dict = None,
                     vol_profile: Dict = None) -> Optional[Dict]:
    last_price = indic["close"]
    atr = indic["atr"]

    if last_price - candidate_price > MAX_WAIT_ATR * atr:
        return None

    stop = _stop_below(candidate_price, atr, sr)
    risk = candidate_price - stop
    nearest_r = _nearest_resistance_above(candidate_price, sr)

    if risk <= 0:
        return None

    if nearest_r:
        reward = nearest_r - candidate_price
    else:
        reward = forecast.get("High", {}).get("price", candidate_price) - candidate_price

    rr = reward / risk if risk > 0 else 0
    if rr < MIN_RR:
        return None

    score = 0
    breakdown = {}

    # R/R quality (35 pts)
    rr_pts = min(35, int((rr / 3.0) * 35))
    score += rr_pts
    breakdown["R/R ratio"] = (rr_pts, 35, f"{rr:.1f}x")

    # ML agreement (30 pts)
    ml_pts = 0
    med_price = forecast.get("Medium", {}).get("price", last_price)
    high_price = forecast.get("High", {}).get("price", last_price)
    low_price = forecast.get("Low", {}).get("price", last_price)

    conv_multiplier = 1.0
    if ml_conviction:
        conv_score = ml_conviction.get("conviction_score", 50)
        conv_multiplier = max(0.2, conv_score / 100)

    if med_price > candidate_price:
        upside_pct = (med_price - candidate_price) / candidate_price * 100
        ml_pts += min(15, int(upside_pct / 2 * 15))

    if high_price > candidate_price and low_price < high_price:
        asymmetry = (high_price - candidate_price) / max(candidate_price - low_price, 0.001)
        ml_pts += min(15, int(asymmetry / 2 * 15))

    ml_pts = int(ml_pts * conv_multiplier)
    score += ml_pts
    breakdown["ML agreement"] = (ml_pts, 30,
        f"Med {med_price:.2f} EGP ({(med_price-candidate_price)/candidate_price*100:+.1f}%) [conviction ×{conv_multiplier:.1f}]")

    # Momentum (20 pts)
    mom_pts = 0
    rsi = indic["rsi"]
    if 35 <= rsi <= 60: mom_pts += 10
    elif 25 <= rsi < 35 or 60 < rsi <= 70: mom_pts += 5

    if indic["macd"] > indic["macd_sig"]: mom_pts += 10
    elif indic["macd"] > 0: mom_pts += 5

    score += mom_pts
    breakdown["Momentum"] = (mom_pts, 20, f"RSI {rsi:.0f} | MACD {'bull' if indic['macd'] > indic['macd_sig'] else 'bear'}")

    # Level confluence (15 pts)
    lvl_pts = 0
    tol = atr * 0.4

    for s in sr.get("support", []):
        if abs(candidate_price - s["price"]) <= tol:
            lvl_pts += 8 if s["strength"] == "Strong" else (5 if s["strength"] == "Medium" else 3)
            break

    for lvl in fib_levels:
        if abs(candidate_price - lvl["price"]) <= tol:
            lvl_pts += 7
            break

    for ma_price in [indic["ema20"], indic["sma50"]]:
        if abs(candidate_price - ma_price) <= tol:
            lvl_pts += 5
            break

    lvl_pts = min(15, lvl_pts)
    score += lvl_pts
    breakdown["Level confluence"] = (lvl_pts, 15, f"candidate {candidate_price:.2f} EGP")

    # VWAP & Volume Profile confluence (up to 10 pts)
    vp_pts = 0
    vwap = indic.get("vwap")
    if vwap and vwap > 0:
        vwap_tol = atr * 0.3
        if abs(candidate_price - vwap) <= vwap_tol:
            vp_pts += 5
        elif candidate_price > vwap:
            vp_pts += 2

    if vol_profile:
        poc = vol_profile.get("poc")
        va_low = vol_profile.get("va_low")
        va_high = vol_profile.get("va_high")
        if poc and abs(candidate_price - poc) <= atr * 0.3:
            vp_pts += 5
        elif va_low and va_high and va_low <= candidate_price <= va_high:
            vp_pts += 3

    vp_pts = min(10, vp_pts)
    score += vp_pts
    vp_detail = f"VWAP {vwap:.2f}" if vwap else "N/A"
    breakdown["VWAP/VP confluence"] = (vp_pts, 10, vp_detail)

    # Candlestick adjustment
    candle_adj = max(-15, min(15, candle_score_delta))
    score += candle_adj
    breakdown["Candlestick patterns"] = (candle_adj, 15, f"net delta {candle_score_delta:+d} pts (capped at ±15)")

    # Proximity penalty
    atr_distance = max(0.0, (last_price - candidate_price) / atr)
    proximity_penalty = int(atr_distance * PROXIMITY_PENALTY_PER_ATR)

    if abs(candidate_price - last_price) / last_price < 0.005:
        proximity_penalty = max(0, proximity_penalty - 8)

    score = max(0, score - proximity_penalty)
    breakdown["Proximity"] = (-proximity_penalty, 0, f"{atr_distance:.1f}×ATR from current price → -{proximity_penalty} pts")

    score = max(0, min(100, score))

    return {
        "price": round(candidate_price, 3),
        "score": score,
        "breakdown": breakdown,
        "stop": round(stop, 3),
        "rr_ratio": round(rr, 2),
        "nearest_resistance": round(nearest_r, 3) if nearest_r else None,
        "risk_per_share": round(risk, 3),
    }

# ─────────────────────────────────────────────────────────────
# Entry Zone
# ─────────────────────────────────────────────────────────────
def compute_entry_zone(indic: Dict, sr: Dict, fib: Dict, forecast: Dict,
                       candle_score_delta: int = 0, ml_conviction: Dict = None,
                       vol_profile: Dict = None) -> Dict:
    last_price = indic["close"]
    atr = indic["atr"]
    fib_levels = fib.get("levels", [])
    zone_half = ATR_ZONE_HALF * atr

    raw_candidates = []

    # Support levels
    for s in sr.get("support", []):
        raw_candidates.append((s["price"], f"Support ({s['strength']})"))

    # Dynamic MAs
    if indic["ema20"] < last_price:
        raw_candidates.append((indic["ema20"], "EMA 20"))
    if indic["sma50"] < last_price:
        raw_candidates.append((indic["sma50"], "SMA 50"))

    # Fibonacci levels below price
    for lvl in fib_levels:
        if lvl["price"] < last_price * 0.995:
            raw_candidates.append((lvl["price"], lvl["label"]))

    # BB lower band (mean-reversion)
    if indic["rsi"] < 45:
        raw_candidates.append((indic["bb_dn"], "BB Lower Band"))

    # VWAP (pullback to VWAP)
    vwap = indic.get("vwap")
    if vwap and vwap < last_price and vwap > 0:
        raw_candidates.append((vwap, "VWAP"))

    # Volume Profile POC and VA Low
    if vol_profile:
        poc = vol_profile.get("poc")
        va_low = vol_profile.get("va_low")
        if poc and poc < last_price:
            raw_candidates.append((poc, "VP POC"))
        if va_low and va_low < last_price:
            raw_candidates.append((va_low, "VP VA Low"))

    # Breakout candidates — resistance levels above current price
    # These are valid when the stock is expected to rise but needs to clear resistance first
    for r in sr.get("resistance", []):
        if r["price"] > last_price and r["price"] < last_price * 1.05:
            # Only if ML forecast supports upside beyond this level
            med_price = forecast.get("Medium", {}).get("price", last_price)
            if med_price > r["price"] * 1.02:
                raw_candidates.append((r["price"], f"Breakout ({r['strength']})"))

    # Current price
    raw_candidates.append((last_price, "Current Price"))

    # Score all candidates
    scored = []
    for price, label in raw_candidates:
        result = _score_candidate(price, indic, sr, fib_levels, forecast,
                                  candle_score_delta=candle_score_delta,
                                  ml_conviction=ml_conviction,
                                  vol_profile=vol_profile)
        if result is not None:
            result["source"] = label
            scored.append(result)

    if not scored:
        atr_stop = last_price - ATR_STOP_MULTIPLIER * atr
        nearest_r = _nearest_resistance_above(last_price, sr)
        risk = last_price - atr_stop
        reward = (nearest_r - last_price) if nearest_r else atr
        return {
            "entry_ideal": round(last_price, 3),
            "entry_low": round(last_price - zone_half, 3),
            "entry_high": round(last_price + zone_half, 3),
            "entry_score": 0,
            "entry_source": "Fallback (no clean setup)",
            "entry_action": "AVOID",
            "distance_pct": 0.0,
            "all_candidates": [],
            "stop_price": round(atr_stop, 3),
            "rr_ratio": round(reward / risk, 2) if risk > 0 else 0,
        }

    scored.sort(key=lambda c: (c["score"], c["price"]), reverse=True)
    best = scored[0]

    distance_pct = (last_price - best["price"]) / last_price * 100
    atr_pct = (atr / last_price) * 100

    if distance_pct > 0:
        # Entry is below current price (pullback)
        if distance_pct <= max(1.5, atr_pct * 0.5):
            action = "BUY NOW"
        elif distance_pct <= max(5.0, atr_pct * 1.5):
            action = "WAIT FOR PULLBACK"
        else:
            action = "WAIT — SIGNIFICANT DIP REQUIRED"
    elif distance_pct < 0:
        # Entry is above current price (breakout)
        breakout_pct = abs(distance_pct)
        if breakout_pct <= 3.0:
            action = "BUY ON BREAKOUT"
        else:
            action = "WAIT FOR BREAKOUT"
    else:
        action = "BUY NOW"

    if best["score"] < 20:
        action = "AVOID — POOR SETUP"

    return {
        "entry_ideal": best["price"],
        "entry_low": round(best["price"] - zone_half, 3),
        "entry_high": round(best["price"] + zone_half, 3),
        "entry_score": best["score"],
        "entry_source": best["source"],
        "entry_action": action,
        "distance_pct": round(distance_pct, 2),
        "all_candidates": scored,
        "stop_price": best["stop"],
        "rr_ratio": best["rr_ratio"],
    }

# ─────────────────────────────────────────────────────────────
# Stop Loss
# ─────────────────────────────────────────────────────────────
def compute_stop_loss(entry_price: float, stop_price: float, atr: float, sr: Dict) -> Dict:
    atr_stop = entry_price - ATR_STOP_MULTIPLIER * atr
    stop_pct = (entry_price - stop_price) / entry_price * 100

    if abs(stop_price - atr_stop) / atr < 0.1:
        method = "ATR (1.5×)"
        stop_type = "trailing"
    else:
        method = "Below support floor"
        stop_type = "structural"

    # Determine exit urgency based on stop distance
    if stop_pct <= 2.0:
        urgency = "tight"
        exit_rule = "Exit immediately on intraday touch — do not wait for close."
        wait_days = 0
    elif stop_pct <= 4.0:
        urgency = "normal"
        exit_rule = "Exit if price closes below stop for 1 consecutive session."
        wait_days = 1
    else:
        urgency = "wide"
        exit_rule = "Exit if price closes below stop for 2 consecutive sessions."
        wait_days = 2

    return {
        "stop_price": round(stop_price, 3),
        "stop_pct": round(stop_pct, 2),
        "method": method,
        "stop_type": stop_type,
        "atr_stop": round(atr_stop, 3),
        "urgency": urgency,
        "exit_rule": exit_rule,
        "wait_days": wait_days,
        "partial_exit": f"Reduce 50% position on first touch, exit remainder on close below.",
    }

# ─────────────────────────────────────────────────────────────
# Take Profit Targets
# ─────────────────────────────────────────────────────────────
def compute_take_profits(entry: float, stop: float, forecast: Dict, sr: Dict, fib: Dict, last_price: float = None) -> List[Dict]:
    risk = entry - stop
    targets = []

    # ML forecast
    for name, f in forecast.items():
        price = f["price"]
        if price > entry:
            rr = (price - entry) / risk if risk > 0 else 0
            if rr >= 1.0:
                targets.append({
                    "source": f"ML {name}",
                    "price": round(price, 3),
                    "pct_from_entry": round((price - entry) / entry * 100, 2),
                    "rr_ratio": round(rr, 2),
                    "above_last_close": (last_price is None or price > last_price),
                })

    # Resistance levels
    for r in sr.get("resistance", []):
        price = r["price"]
        if price > entry:
            rr = (price - entry) / risk if risk > 0 else 0
            if rr >= 1.0:
                targets.append({
                    "source": f"Resistance ({r['strength']})",
                    "price": round(price, 3),
                    "pct_from_entry": round((price - entry) / entry * 100, 2),
                    "rr_ratio": round(rr, 2),
                    "above_last_close": (last_price is None or price > last_price),
                })

    # Fibonacci levels
    for lvl in fib.get("levels", []):
        price = lvl["price"]
        if price > entry:
            rr = (price - entry) / risk if risk > 0 else 0
            if rr >= 1.0:
                targets.append({
                    "source": lvl["label"],
                    "price": round(price, 3),
                    "pct_from_entry": round((price - entry) / entry * 100, 2),
                    "rr_ratio": round(rr, 2),
                    "above_last_close": (last_price is None or price > last_price),
                })

    # Deduplicate
    targets.sort(key=lambda t: t["price"])
    merged = []
    for t in targets:
        if merged and abs(t["price"] - merged[-1]["price"]) / merged[-1]["price"] < 0.005:
            priority = {"ML": 0, "Resistance": 1, "Fib": 2}
            existing_p = next((k for k in priority if merged[-1]["source"].startswith(k)), 2)
            new_p = next((k for k in priority if t["source"].startswith(k)), 2)
            if new_p < existing_p:
                merged[-1] = t
        else:
            merged.append(t)

    # Fallback: if no targets found, use ATR-based targets from entry
    if not merged and entry and stop:
        risk = entry - stop
        if risk > 0:
            for multiplier, label in [(1.5, "ATR 1.5R"), (2.5, "ATR 2.5R"), (4.0, "ATR 4.0R")]:
                price = entry + risk * multiplier
                merged.append({
                    "source": label,
                    "price": round(price, 3),
                    "pct_from_entry": round((price - entry) / entry * 100, 2),
                    "rr_ratio": round(multiplier, 2),
                    "above_last_close": True,
                })

    return merged

# ─────────────────────────────────────────────────────────────
# Position Sizing
# ─────────────────────────────────────────────────────────────
def compute_position_size(capital: float, entry: float, stop: float, risk_pct: float = 1.0) -> Dict:
    risk_per_share = entry - stop
    if risk_per_share <= 0:
        return {"error": "Stop is above or equal to entry — invalid trade setup."}

    risk_amount = capital * (risk_pct / 100)
    shares = max(1, int(risk_amount / risk_per_share))
    position_val = shares * entry
    position_pct = position_val / capital * 100

    return {
        "risk_amount": round(risk_amount, 2),
        "shares": shares,
        "position_val": round(position_val, 2),
        "position_pct": round(position_pct, 2),
        "risk_pct": risk_pct,
    }

# ─────────────────────────────────────────────────────────────
# Signal Score
# ─────────────────────────────────────────────────────────────
def compute_signal_score(last_price: float, df_tech: pd.DataFrame, sr: Dict, forecast: Dict, entry_price: float) -> Dict:
    score = 0
    details = []
    indic = _get_indicators(df_tech)

    def check(label, pts, condition, note=""):
        nonlocal score
        earned = pts if condition else 0
        score += earned
        details.append({
            "Signal": label, "Max": pts, "Earned": earned,
            "✓": "✅" if condition else "❌", "Note": note,
        })

    check("Price / Entry above SMA 50", 10, entry_price > indic["sma50"], f"SMA50={indic['sma50']:.2f}")
    check("Price / Entry above SMA 200", 10, entry_price > indic["sma200"], f"SMA200={indic['sma200']:.2f}")

    rsi = indic["rsi"]
    check("RSI in healthy zone (35–65)", 10, 35 <= rsi <= 65, f"RSI={rsi:.1f}")
    check("MACD > Signal (bullish)", 10, indic["macd"] > indic["macd_sig"], f"MACD={indic['macd']:.3f} Sig={indic['macd_sig']:.3f}")

    check("Entry at/near support zone", 10,
          any(abs(s["price"] - entry_price) / entry_price < 0.02 for s in sr.get("support", [])),
          "within 2% of a support level")

    med_price = forecast.get("Medium", {}).get("price", last_price)
    high_price = forecast.get("High", {}).get("price", last_price)
    low_price = forecast.get("Low", {}).get("price", last_price)

    check("ML Medium forecast above entry", 20, med_price > entry_price, f"Medium={med_price:.2f} vs entry={entry_price:.2f}")

    cone_up = high_price - entry_price
    cone_down = entry_price - low_price
    check("ML upside > downside from entry", 10, cone_up > cone_down, f"Up={cone_up:.2f} EGP  Down={cone_down:.2f} EGP")

    nearest_r = _nearest_resistance_above(entry_price, sr)
    stop_p = _stop_below(entry_price, indic["atr"], sr)
    risk = entry_price - stop_p
    reward = (nearest_r - entry_price) if nearest_r else 0
    rr = reward / risk if risk > 0 else 0

    check("R/R ratio ≥ 2.0x", 15, rr >= 2.0, f"R/R={rr:.1f}x (reward={reward:.2f} / risk={risk:.2f})")

    close_res = any(r["strength"] == "Strong" and (r["price"] - entry_price) / entry_price < 0.03 for r in sr.get("resistance", []))
    check("No strong resistance within 3%", 5, not close_res, "clear path to first target")

    if score >= 75: rating = "Strong 🟢"
    elif score >= 55: rating = "Moderate 🟡"
    elif score >= 35: rating = "Weak 🔴"
    else: rating = "Avoid ⛔"

    return {"score": score, "rating": rating, "details": details}

# ─────────────────────────────────────────────────────────────
# Holding Duration Recommendation
# ─────────────────────────────────────────────────────────────
def compute_holding_recommendation(forecast: Dict, trade: Dict, regime: Dict, patterns: Dict, conviction: Dict, last_price: float) -> Dict:
    signal_score = trade.get("signal", {}).get("score", 0)
    regime_type = regime.get("regime", "unknown")
    conv_score = conviction.get("conviction_score", 0)
    pattern_sig = patterns.get("latest_signal", "neutral")
    rr_ratio = trade.get("entry", {}).get("rr_ratio", 0)
    targets = trade.get("targets", [])
    entry_action = trade.get("entry", {}).get("entry_action", "")
    adx = regime.get("adx", 0)

    if "AVOID" in entry_action or "POOR" in entry_action:
        return {"min_weeks": 0, "max_weeks": 0, "target_weeks": 0, "duration_label": "Do not enter",
                "calendar_note": "", "exit_strategy": "Setup is not recommended.", "rationale": ["Signal score or R/R too low."],
                "exit_triggers": []}

    rationale = []
    exit_triggers = []

    # Regime base range
    if "trending_up" in regime_type:
        min_weeks, max_weeks = 1, 4
        rationale.append(f"Confirmed uptrend (ADX {adx:.0f} > 25) — holds typically run 2–4 weeks.")
    elif "trending_down" in regime_type:
        min_weeks, max_weeks = 1, 2
        rationale.append(f"Downtrend (ADX {adx:.0f} > 25, bearish) — counter-trend long, keep short.")
    elif regime_type == "ranging":
        min_weeks, max_weeks = 1, 3
        rationale.append("Ranging market (ADX < 20) — range trades complete in 1–3 weeks.")
    else:
        min_weeks, max_weeks = 1, 2
        rationale.append("Regime transitioning — hold conservatively 1–2 weeks.")

    # ML conviction gates upper range
    if conv_score >= 70:
        target_weeks = max_weeks
        rationale.append(f"High ML conviction ({conv_score}/100) — target upper range ({max_weeks} weeks).")
    elif conv_score >= 45:
        target_weeks = round((min_weeks + max_weeks) / 2)
        rationale.append(f"Moderate ML conviction ({conv_score}/100) — aim for mid-range ({target_weeks} weeks).")
    else:
        target_weeks = min_weeks
        max_weeks = min(max_weeks, min_weeks + 1)
        rationale.append(f"Low ML conviction ({conv_score}/100) — target TP1 within {target_weeks} week.")

    # Signal score gates multi-target holding
    if signal_score >= 75:
        rationale.append(f"Strong confluence ({signal_score}/100) — holding partial toward TP2 reasonable.")
    elif signal_score < 45:
        target_weeks = min_weeks
        max_weeks = min(max_weeks, min_weeks + 1)
        rationale.append(f"Weak confluence ({signal_score}/100) — exit at TP1.")

    # Candlestick adjustment
    if pattern_sig == "bearish":
        pnames = [p["name"] for p in patterns.get("patterns", []) if p["signal"] == "bearish"]
        target_weeks = min_weeks
        max_weeks = min(max_weeks, min_weeks + 1)
        rationale.append(f"Bearish candle ({', '.join(pnames)}) — reduce hold target.")
        exit_triggers.append(f"Bearish candle ({', '.join(pnames)}) confirmed — tighten stop or exit.")
    elif pattern_sig == "bullish":
        pnames = [p["name"] for p in patterns.get("patterns", []) if p["signal"] == "bullish"]
        rationale.append(f"Bullish candle ({', '.join(pnames)}) — momentum supports holding.")

    # Exit strategy — dynamic based on individual stock factors
    entry_price = trade.get("entry", {}).get("entry_ideal", last_price)
    if targets:
        tp1 = targets[0]["price"]
        tp2 = targets[1]["price"] if len(targets) >= 2 else None
        tp3 = targets[2]["price"] if len(targets) >= 3 else None
        tp1_pct = (tp1 - entry_price) / entry_price * 100 if entry_price else 0
        tp2_pct = (tp2 - entry_price) / entry_price * 100 if tp2 and entry_price else 0
        tp3_pct = (tp3 - entry_price) / entry_price * 100 if tp3 and entry_price else 0

        # Determine exit style based on multiple factors
        # Strong: high R/R, high conviction, trending, volume confirmed
        # Moderate: decent R/R, moderate conviction
        # Conservative: low R/R, weak conviction, ranging, or bearish candles
        strength_score = 0
        if rr_ratio >= 3.0: strength_score += 2
        elif rr_ratio >= 2.0: strength_score += 1

        if conv_score >= 60: strength_score += 2
        elif conv_score >= 40: strength_score += 1

        if signal_score >= 70: strength_score += 1
        elif signal_score < 45: strength_score -= 1

        if "trending_up" in regime_type: strength_score += 1
        elif "trending_down" in regime_type: strength_score -= 1

        if pattern_sig == "bullish": strength_score += 1
        elif pattern_sig == "bearish": strength_score -= 1

        # Build dynamic exit strategy
        stop_urgency = trade.get("stop", {}).get("urgency", "normal")
        stop_exit_rule = trade.get("stop", {}).get("exit_rule", "")
        stop_partial = trade.get("stop", {}).get("partial_exit", "")

        if strength_score >= 4:
            # Strong setup: scale out gradually, let winners run
            parts = [f"Strong setup (score {strength_score}/7). Scale out approach:"]
            parts.append(f"Sell 30% at TP1 ({tp1:.2f}, +{tp1_pct:.1f}%) — move stop to breakeven.")
            if tp2:
                parts.append(f"Sell 30% at TP2 ({tp2:.2f}, +{tp2_pct:.1f}%) — trail stop to TP1.")
            if tp3:
                parts.append(f"Trail final 40% toward TP3 ({tp3:.2f}, +{tp3_pct:.1f}%).")
            else:
                parts.append("Trail final 40% with EMA20 trailing stop.")
            parts.append(f"Stop-loss: {stop_exit_rule} {stop_partial}")
            exit_strategy = " ".join(parts)

        elif strength_score >= 2:
            # Moderate setup: take most profit at TP1, trail remainder
            parts = [f"Moderate setup (score {strength_score}/7)."]
            parts.append(f"Sell 60% at TP1 ({tp1:.2f}, +{tp1_pct:.1f}%).")
            if tp2:
                parts.append(f"Sell 30% at TP2 ({tp2:.2f}, +{tp2_pct:.1f}%).")
            parts.append("Trail final 10% — exit if price closes below EMA20.")
            parts.append(f"Stop-loss: {stop_exit_rule}")
            exit_strategy = " ".join(parts)

        elif strength_score >= 0:
            # Conservative: take profit early, tight trail
            parts = [f"Conservative setup (score {strength_score}/7)."]
            parts.append(f"Sell 70% at TP1 ({tp1:.2f}, +{tp1_pct:.1f}%).")
            if tp2:
                parts.append(f"Sell remaining 30% at TP2 ({tp2:.2f}, +{tp2_pct:.1f}%).")
            else:
                parts.append("Trail remaining 30% — exit if price drops 2% from peak.")
            parts.append(f"Stop-loss: {stop_exit_rule}")
            exit_strategy = " ".join(parts)

        else:
            # Weak: exit quickly
            parts = [f"Weak setup (score {strength_score}/7). Exit fast."]
            parts.append(f"Sell 100% at TP1 ({tp1:.2f}, +{tp1_pct:.1f}%).")
            parts.append("Do not hold for TP2 — momentum is against you.")
            parts.append(f"Stop-loss: {stop_exit_rule}")
            exit_strategy = " ".join(parts)
    else:
        med_price = forecast.get("Medium", {}).get("price", last_price)
        exit_strategy = (f"No resistance target. Use ML Medium ({med_price:.2f}) as exit guide. "
                        f"Exit within {target_weeks} week{'s' if target_weeks > 1 else ''}.")

    calendar_note = "EGX trades Sun–Thu. 1 week = 5 sessions. Plan around holidays."
    if "WAIT" in entry_action or "BREAKOUT" in entry_action:
        calendar_note += f" Note: waiting for {'breakout' if 'BREAKOUT' in entry_action else 'pullback'} — {target_weeks}-week hold starts FROM entry date."

    stop_price = trade.get("stop", {}).get("stop_price", 0)
    stop_pct = trade.get("stop", {}).get("stop_pct", 0)
    stop_urgency = trade.get("stop", {}).get("urgency", "normal")
    stop_exit_rule = trade.get("stop", {}).get("exit_rule", "")
    stop_partial = trade.get("stop", {}).get("partial_exit", "")
    med_fc = forecast.get("Medium", {}).get("price", 0)
    exit_triggers.extend([
        f"STOP-LOSS at {stop_price:.2f} ({stop_pct:.1f}% below entry) — {stop_exit_rule}",
        f"Stop urgency: {stop_urgency.upper()} — {stop_partial}",
        "Price closes below EMA20 for two consecutive sessions — tighten or exit.",
        f"ML Medium reached ({med_fc:.2f}) — reassess or take partial profit.",
        "High volume bearish day near entry/resistance — cut or reduce position.",
        f"Hold exceeded {max_weeks} week{'s' if max_weeks > 1 else ''} without TP1 — exit.",
    ])

    if "BREAKOUT" in entry_action:
        exit_triggers.insert(0, "Breakout failed — price fell back below breakout level after entry. Exit immediately.")
        exit_triggers.insert(1, "Volume on breakout day was below 1.5x average — weak breakout, tighten stop.")

    duration_label = f"{min_weeks}–{max_weeks} week{'s' if max_weeks > 1 else ''} (~{min_weeks*5}–{max_weeks*5} EGX sessions)"

    return {
        "min_weeks": min_weeks, "max_weeks": max_weeks, "target_weeks": target_weeks,
        "duration_label": duration_label, "calendar_note": calendar_note,
        "exit_strategy": exit_strategy, "rationale": rationale, "exit_triggers": exit_triggers,
    }

# ─────────────────────────────────────────────────────────────
# Master Trade Plan Builder
# ─────────────────────────────────────────────────────────────
def build_trade_plan(df_tech: pd.DataFrame, sr: Dict, fib: Dict, forecast: Dict,
                     capital: float = 10000.0, risk_pct: float = 1.0,
                     regime: Dict = None, patterns: Dict = None, mae_pct: float = None,
                     vol_profile: Dict = None, current_price: float = None) -> Dict:
    indic = _get_indicators(df_tech)
    last_price = indic["close"]

    # Override with TradingView real-time price if available
    if current_price is not None and current_price > 0:
        last_price = float(current_price)
        indic["close"] = last_price

    regime = regime or {"regime": "unknown", "adx": 0.0, "direction": "neutral"}
    patterns = patterns or {"patterns": [], "latest_signal": "neutral", "score_delta": 0}

    # ML conviction first
    conviction = compute_ml_conviction(forecast, last_price, mae_pct)

    # Entry zone (uses conviction + candle delta)
    entry = compute_entry_zone(indic, sr, fib, forecast,
                               candle_score_delta=patterns.get("score_delta", 0),
                               ml_conviction=conviction,
                               vol_profile=vol_profile)

    stop = compute_stop_loss(entry["entry_ideal"], entry["stop_price"], indic["atr"], sr)
    tps = compute_take_profits(entry["entry_ideal"], stop["stop_price"], forecast, sr, fib, last_price=last_price)
    pos = compute_position_size(capital, entry["entry_ideal"], stop["stop_price"], risk_pct)
    signal = compute_signal_score(last_price, df_tech, sr, forecast, entry["entry_ideal"])

    partial_trade = {"entry": entry, "stop": stop, "targets": tps, "signal": signal}
    holding = compute_holding_recommendation(forecast, partial_trade, regime, patterns, conviction, last_price)

    return {
        "last_price": last_price, "atr": round(indic["atr"], 3),
        "entry": entry, "stop": stop, "targets": tps, "position": pos,
        "signal": signal, "conviction": conviction, "holding": holding,
        "regime": regime, "patterns": patterns,
    }