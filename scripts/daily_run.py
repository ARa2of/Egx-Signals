#!/usr/bin/env python3
"""
Daily EGX Signal Generator
Runs at 5:00 PM UK time via GitHub Actions
"""
import argparse
import logging
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_params, load_enhancement_config
from src.data.loader import (
    read_ticker_list, read_ticker_index_map, download_all, fetch_all_ta,
    download_fx, usd_valuation, volume_analysis, money_flow_volume_analysis,
    support_resistance, compute_adl, compute_adl_trend, compute_mfi,
    compute_bb_squeeze, compute_sma_ema_rsi_from_yf,
    fetch_index_sentiment, INDEX_SYMBOLS,
)
from src.signals.technical import enrich
from src.signals.ml import run_ml_pipeline, compute_ml_conviction
from src.signals.scoring import compute_base_score
from src.signals.chartscan import init_chartscan, chartscan_analyze, is_enabled as chartscan_enabled
from src.trade.planner import build_trade_plan
from src.store.signal_store import append_signals, export_latest_csv
from src.output.excel import export_analysis, append_daily_history
from src.output.alerts import send_daily_alert, send_error_alert

# ─────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("daily_run")

# ─────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────
def run_daily_analysis(input_file: str, output_dir: str = "output") -> List[Dict]:
    """Run the complete daily analysis pipeline."""
    log.info("=" * 60)
    log.info("EGX DAILY SIGNAL GENERATOR - %s", date.today().strftime("%d %b %Y"))
    log.info("=" * 60)

    # Load params
    params = load_params()
    enhance_cfg = load_enhancement_config()
    data_cfg = params["data"]
    indices_cfg = params["indices"]
    chartscan_cfg = params["chartscan"]

    # 1. Read tickers
    log.info("Reading ticker list from %s", input_file)
    tickers = read_ticker_list(input_file, "Selected_Stocks")
    if not tickers:
        raise ValueError("No tickers found in 'Selected_Stocks' sheet.")

    index_map = read_ticker_index_map(input_file, "Selected_Stocks")
    log.info("Loaded %d tickers", len(tickers))

    # Read INDEX membership from sector groups
    sector_file = Path(__file__).parent.parent / params.get("ml", {}).get("sector_groups_file", "config/EGX_Stock_Groups.xlsx")
    ticker_index_map = {}
    if sector_file.exists():
        sector_df = pd.read_excel(sector_file)
        ticker_index_map = dict(zip(sector_df["Ticker"], sector_df["INDEX"]))

    # 2. Download yfinance data
    log.info("Downloading yfinance data for %d tickers...", len(tickers))
    yf_cache: Dict = {}
    download_all(tickers, yf_cache)

    # Also download sector peers for ML
    ml_cfg = params.get("ml", {})
    if ml_cfg.get("peer_selection") == "sector":
        sector_file = Path(__file__).parent.parent / ml_cfg.get("sector_groups_file", "config/EGX_Stock_Groups.xlsx")
        if sector_file.exists():
            sector_df = pd.read_excel(sector_file)
            sector_tickers = [t for t in sector_df["Ticker"].tolist() if t not in yf_cache]
            if sector_tickers:
                log.info("Downloading %d sector peer tickers for ML...", len(sector_tickers))
                download_all(sector_tickers, yf_cache)

    valid_tickers = [t for t in tickers if yf_cache.get(t) and yf_cache[t].ok]
    all_valid = [t for t in yf_cache if yf_cache[t].ok]  # includes sector peers
    log.info("Valid yfinance data for %d tickers (+ sector peers)", len(valid_tickers))

    # 3. Fetch TradingView TA (batched)
    log.info("Fetching TradingView TA data...")
    ta_cache: Dict = {}
    fetch_all_ta(valid_tickers, ta_cache)

    tv_success = sum(1 for t in valid_tickers if ta_cache.get(t) and ta_cache[t].ok)
    log.info("TradingView data: %d/%d successful", tv_success, len(valid_tickers))

    # 3b. Fetch index sentiment
    log.info("Fetching index sentiment (EGX30, EGX70, EGX100)...")
    index_sentiment = fetch_index_sentiment()

    # 4. Download FX rate
    log.info("Downloading USD/EGP exchange rate...")
    fx_series = download_fx(period=data_cfg["history_period"])

    # 5. Initialize ChartScan AI
    chartscan_cfg = params.get("chartscan", {})
    model_path = chartscan_cfg.get("model_path", "weights/custom_yolov8.pt")
    chartscan_loaded = init_chartscan(model_path, chartscan_cfg.get("enabled", True))
    if chartscan_loaded:
        log.info("ChartScan AI initialized successfully")
    else:
        log.info("ChartScan AI disabled or model not found")

    # 5. Process each ticker
    rows = []
    index_rows = []

    for raw in valid_tickers:
        try:
            yf_entry = yf_cache[raw]
            ta_entry = ta_cache.get(raw)

            # Get indicators (TV or fallback)
            if ta_entry and ta_entry.ok:
                ind = ta_entry.indicators
                # SMA 20 not in TV TA data — compute from yfinance
                sma20 = float(yf_entry.history["Close"].rolling(20).mean().iloc[-1]) if len(yf_entry.history) >= 20 else None
                sma50 = ind.get("SMA50")
                sma200 = ind.get("SMA200")
                ema20 = ind.get("EMA20")
                ema50 = ind.get("EMA50")
                ema200 = ind.get("EMA200")
                rsi = ind.get("RSI")
                close_ta = ind.get("close")
                vwma = ind.get("VWMA")
                macd = ind.get("MACD.macd")
                macd_signal = ind.get("MACD.signal")
                adx = ind.get("ADX")
                adx_plus = ind.get("ADX+DI")
                adx_minus = ind.get("ADX-DI")
                bb_lower = ind.get("BB.lower")
                bb_upper = ind.get("BB.upper")
                ta_fetch_time = ta_entry.fetch_time.strftime("%Y-%m-%d %H:%M:%S") if ta_entry.fetch_time else None
                ta_source = "TradingView"
            else:
                # Fallback to yfinance
                log.info("%s: Using yfinance fallback", raw)
                fallback = compute_sma_ema_rsi_from_yf(yf_entry.history)
                sma20 = fallback["sma20"]
                sma50 = fallback["sma50"]
                sma200 = fallback["sma200"]
                ema20 = fallback["ema20"]
                ema50 = fallback["ema50"]
                ema200 = fallback["ema200"]
                rsi = fallback["rsi"]
                close_ta = None
                vwma = None
                macd = fallback["macd"]
                macd_signal = fallback["macd_signal"]
                adx = None
                adx_plus = None
                adx_minus = None
                bb_lower = None
                bb_upper = None
                ta_fetch_time = None
                ta_source = "yfinance_fallback"
                ind = {}

            # Compute analytics
            val = usd_valuation(raw, yf_cache, fx_series, ta_cache)
            vol = volume_analysis(raw, yf_cache, ta_cache)
            mf = money_flow_volume_analysis(raw, yf_cache)
            sr = support_resistance(raw, yf_cache)

            # ADL, MFI, BB Squeeze
            adl = compute_adl(yf_entry.history)
            adl_trend = compute_adl_trend(yf_entry.history)
            mfi = compute_mfi(yf_entry.history)

            atr_val = None
            if "High" in yf_entry.history.columns and "Low" in yf_entry.history.columns:
                h = yf_entry.history["High"].values.astype(float)
                lo = yf_entry.history["Low"].values.astype(float)
                c = yf_entry.history["Close"].values.astype(float)
                if len(c) >= 2:
                    trs = []
                    for i in range(1, len(h)):
                        tr = max(h[i] - lo[i], abs(h[i] - c[i-1]), abs(lo[i] - c[i-1]))
                        trs.append(tr)
                    atr_val = float(pd.Series(trs[-14:]).mean()) if trs else None

            bb_squeeze = compute_bb_squeeze(bb_lower, bb_upper, atr_val)

            # Current price (prefer TV)
            current_price = close_ta or val.get("current_egp")

            # Cross signals
            golden_cross = death_cross = None
            if sma50 is not None and sma200 is not None:
                if sma50 > sma200:
                    golden_cross, death_cross = "Yes", "No"
                elif sma50 < sma200:
                    golden_cross, death_cross = "No", "Yes"
                else:
                    golden_cross, death_cross = "No", "No"

            ema_bullish = "Yes" if (ema50 and ema200 and ema50 > ema200) else ("No" if ema50 and ema200 else None)
            diamond_cross = "Yes" if (ema20 and ema50 and ema20 > ema50) else ("No" if ema20 and ema50 else None)
            macd_bullish = "Yes" if (macd and macd_signal and macd > macd_signal) else ("No" if macd and macd_signal else None)

            # Technical enrichment
            raw_df = yf_entry.history[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
            tech = enrich(raw_df)
            df_tech = tech["df_enriched"]
            fib = tech["fibonacci"]
            sr_tech = tech["sr"]
            patterns = tech["patterns"]
            regime = tech["regime"]
            vol_profile = tech["volume_profile"]

            # VWAP
            last_vwap = float(df_tech["VWAP"].iloc[-1]) if "VWAP" in df_tech.columns and not pd.isna(df_tech["VWAP"].iloc[-1]) else None
            last_dist_vwap = float(df_tech["dist_VWAP"].iloc[-1]) if "dist_VWAP" in df_tech.columns and not pd.isna(df_tech["dist_VWAP"].iloc[-1]) else None

            # ML Pipeline
            ml_cfg = params.get("ml", {})
            peer_selection = ml_cfg.get("peer_selection", "sector")
            max_peers = ml_cfg.get("max_peers", 4)
            
            if peer_selection == "sector":
                from src.signals.ml import get_sector_peers
                peer_tickers = get_sector_peers(raw, all_valid, max_peers)
            elif peer_selection == "custom":
                custom_peers = ml_cfg.get("custom_peers", [])
                peer_tickers = [t for t in custom_peers if t in all_valid]
            else:
                peer_tickers = all_valid
            
            peer_close = pd.DataFrame({t: yf_cache[t].history["Close"] for t in peer_tickers if t in yf_cache and yf_cache[t].ok})
            ml_result = run_ml_pipeline(df_tech, peer_close, raw, {})

            if "error" in ml_result:
                log.warning("%s: ML failed - %s", raw, ml_result["error"])
                ml_result = {}

            # ML Conviction
            if ml_result and "eval" in ml_result:
                eval_res = ml_result.get("eval", {})
                avg_mae = eval_res.get("avg_mae", {})
                xgb_mae = avg_mae.get("xgb", {}).get("Medium", 0)
                lgbm_mae = avg_mae.get("lgbm", {}).get("Medium", 0)
                mae_pct = ((xgb_mae + lgbm_mae) / 2) * 100 if (xgb_mae or lgbm_mae) else 0
            else:
                mae_pct = 0
            conviction = compute_ml_conviction(
                ml_result.get("forecast", {}),
                current_price or 0,
                mae_pct,
                direction_accuracy=ml_result.get("direction_accuracy"),
                ensemble_weights=ml_result.get("ensemble_weights"),
            )

            # ChartScan AI
            cs_result = None
            if chartscan_enabled():
                cs_result = chartscan_analyze(yf_entry.history, raw)
                if cs_result:
                    log.info("%s: ChartScanAI signal=%s, conf=%.2f", raw, cs_result.get("signal"), cs_result.get("confidence"))

            # Base Score
            is_near_support = False
            if current_price and sr.get("support"):
                distance_pct = (current_price - sr["support"]) / sr["support"]
                is_near_support = 0 <= distance_pct <= params["volume"]["near_support_pct"]

            buy_multiplier = mf.get("buy_vol_multiplier")
            is_volume_spike = buy_multiplier is not None and buy_multiplier >= params["volume"]["spike_multiplier"]

            base_score = compute_base_score(
                current_price=current_price,
                ema20=ema20, ema50=ema50, ema200=ema200,
                macd=macd, macd_signal=macd_signal,
                rsi=rsi, adx=adx, regime=regime.get("regime", "unknown"),
                vol_multiplier=vol.get("vol_multiplier"),
                buy_vol_multiplier=buy_multiplier,
                adl_trend=adl_trend, mfi=mfi,
                is_near_support=is_near_support,
                volume_confirmed=is_volume_spike,
                support=sr.get("support"),
                dist_vwap=last_dist_vwap, vwap=last_vwap,
                above_poc=vol_profile.get("above_poc"),
                poc=vol_profile.get("poc"),
                va_high=vol_profile.get("va_high"),
                va_low=vol_profile.get("va_low"),
            )

            # Death cross override
            if death_cross == "Yes":
                base_score["recommendation"] = "Avoid"
                base_score["recommendation_basis"] += "; Death Cross overrides to Avoid"

            # Trade Plan
            trade = build_trade_plan(
                df_tech=df_tech, sr=sr_tech, fib=fib,
                forecast=ml_result.get("forecast", {}),
                capital=100000, risk_pct=1.0,
                regime=regime, patterns=patterns, mae_pct=mae_pct,
                vol_profile=vol_profile,
            )

            # Build output row
            row = {
                "Analysis Run Date": date.today(),
                "Selected Stock": raw,
                "Index Membership": index_map.get(raw, "UNINDEX"),
                "Data As Of": yf_entry.history.index[-1].strftime("%Y-%m-%d"),
                "Current EGP Price": round(current_price, 4) if current_price else None,
                "Current USD Price": round(val["current_usd"], 4) if val["current_usd"] else None,
                "Historical Min USD Price": round(val["hist_min_usd"], 4) if val["hist_min_usd"] else None,
                "Historical Max USD Price": round(val["hist_max_usd"], 4) if val["hist_max_usd"] else None,
                "Undervalued (Yes/No)": val["undervalued"],
                "Implied Fair Value (EGP)": round(val["implied_fair_value_egp"], 4) if val["implied_fair_value_egp"] else None,
                "Fair Value Method": val["fair_value_method"],
                "P/E Ratio (TTM)": round(val["pe_ratio_ttm"], 2) if val["pe_ratio_ttm"] else None,
                "EPS (TTM)": round(val["eps_ttm"], 4) if val["eps_ttm"] else None,
                "1-Year Avg Volume": round(vol["avg_vol_1y"], 0) if vol["avg_vol_1y"] else None,
                "Last Day Volume": round(vol["last_day_vol"], 0) if vol["last_day_vol"] else None,
                "Volume Multiplier (vs 1Y)": vol["vol_multiplier"],
                "Est. Buy Volume (2-Month Avg)": round(mf["buy_vol_avg_2mo"], 0) if mf["buy_vol_avg_2mo"] else None,
                "Est. Buy Volume (Last Day)": round(mf["buy_vol_last_day"], 0) if mf["buy_vol_last_day"] else None,
                "Buy Volume Multiplier (vs 2-Month)": mf["buy_vol_multiplier"],
                "Support": round(sr["support"], 4) if sr["support"] else None,
                "Resistance": round(sr["resistance"], 4) if sr["resistance"] else None,
                "50 SMA": round(sma50, 4) if sma50 else None,
                "200 SMA": round(sma200, 4) if sma200 else None,
                "20 SMA": round(sma20, 4) if sma20 else None,
                "Golden Cross (Yes/No)": golden_cross,
                "Death Cross (Yes/No)": death_cross,
                "20 EMA": round(ema20, 4) if ema20 else None,
                "50 EMA": round(ema50, 4) if ema50 else None,
                "200 EMA": round(ema200, 4) if ema200 else None,
                "EMA Bullish (50>200) (Yes/No)": ema_bullish,
                "Diamond Cross (20>50) (Yes/No)": diamond_cross,
                "MACD": round(macd, 4) if macd else None,
                "MACD Signal": round(macd_signal, 4) if macd_signal else None,
                "MACD Bullish (Yes/No)": macd_bullish,
                "RSI (%)": round(rsi, 2) if rsi else None,
                "VWMA": round(vwma, 4) if vwma else None,
                "ADX": round(adx, 2) if adx else None,
                "ADX +DI": round(adx_plus, 2) if adx_plus else None,
                "ADX -DI": round(adx_minus, 2) if adx_minus else None,
                "MFI": round(mfi, 2) if mfi else None,
                "BB Squeeze": "Yes" if bb_squeeze else ("No" if bb_squeeze is not None else None),
                "ADL": round(adl, 0) if adl else None,
                "ADL Trend (20d)": round(adl_trend, 0) if adl_trend else None,
                "TA Data As Of": ta_fetch_time,
                "Optimal Entry Price": trade["entry"]["entry_ideal"],
                "Entry Action": trade["entry"]["entry_action"],
                "Entry Source": trade["entry"]["entry_source"],
                "Entry Score": trade["entry"]["entry_score"],
                "Entry Distance %": trade["entry"]["distance_pct"],
                "Entry High": trade["entry"]["entry_high"],
                "Stop Loss": trade["stop"]["stop_price"],
                "Stop Loss Basis": trade["stop"]["method"],
                "Take Profit 1": trade["targets"][0]["price"] if trade["targets"] else None,
                "Take Profit 2": trade["targets"][1]["price"] if len(trade["targets"]) > 1 else None,
                "Take Profit 3": trade["targets"][2]["price"] if len(trade["targets"]) > 2 else None,
                "Take Profit Basis": ", ".join([f"{t['source']}: {t['price']:.3f}" for t in trade["targets"][:3]]),
                "TP1 Risk/Reward": trade["targets"][0]["rr_ratio"] if trade["targets"] else None,
                "TP2 Risk/Reward": trade["targets"][1]["rr_ratio"] if len(trade["targets"]) > 1 else None,
                "TP3 Risk/Reward": trade["targets"][2]["rr_ratio"] if len(trade["targets"]) > 2 else None,
                "TP1 Reward %": trade["targets"][0]["pct_from_entry"] if trade["targets"] else None,
                "TP2 Reward %": trade["targets"][1]["pct_from_entry"] if len(trade["targets"]) > 1 else None,
                "TP3 Reward %": trade["targets"][2]["pct_from_entry"] if len(trade["targets"]) > 2 else None,
                "Recommendation": base_score["recommendation"],
                "Recommendation Basis": base_score["recommendation_basis"],
                "Score": base_score["raw_score"],
                "Score - Trend": base_score["score_breakdown"]["trend"],
                "Score - MACD": base_score["score_breakdown"]["macd"],
                "Score - RSI": base_score["score_breakdown"]["rsi"],
                "Score - Volume": base_score["score_breakdown"]["volume"],
                "Score - ADI": base_score["score_breakdown"]["adi"],
                "Score - Support": base_score["score_breakdown"]["support"],
                "Score - VWAP": base_score["score_breakdown"]["vwap"],
                "Score - Volume Profile": base_score["score_breakdown"]["volume_profile"],
                "VWAP": round(last_vwap, 4) if last_vwap else None,
                "Dist VWAP %": round(last_dist_vwap * 100, 2) if last_dist_vwap is not None else None,
                "Volume Profile POC": vol_profile.get("poc"),
                "Volume Profile VA High": vol_profile.get("va_high"),
                "Volume Profile VA Low": vol_profile.get("va_low"),
                "Above POC": vol_profile.get("above_poc"),
                "ChartScanAI Signal": cs_result.get("signal", "N/A") if cs_result else "N/A",
                "ChartScanAI Recommendation": "Buy" if cs_result and cs_result.get("signal") == "Buy" else ("Avoid" if cs_result and cs_result.get("signal") == "Sell" else "Hold"),
                "ChartScanAI Confidence": cs_result.get("confidence") if cs_result else None,
                "ChartScanAI Buy Patterns": cs_result.get("buy_patterns", 0) if cs_result else 0,
                "ChartScanAI Sell Patterns": cs_result.get("sell_patterns", 0) if cs_result else 0,
                # Additional fields for signal store
                "close_usd": val["current_usd"],
                "fx_rate": fx_series.iloc[-1] if fx_series is not None and not fx_series.empty else None,
                "macd_bullish": macd_bullish == "Yes",
                "golden_cross": golden_cross == "Yes",
                "death_cross": death_cross == "Yes",
                "diamond_cross": diamond_cross == "Yes",
                "adx": adx,
                "mfi": mfi,
                "bb_squeeze": bb_squeeze,
                "regime": regime.get("regime", "unknown"),
                "fair_value_egp": val["implied_fair_value_egp"],
                "fair_value_method": val["fair_value_method"],
                "pe_ttm": val["pe_ratio_ttm"],
                "undervalued": val["undervalued"] == "Yes",
                "entry_price": trade["entry"]["entry_ideal"],
                "stop_loss": trade["stop"]["stop_price"],
                "tp1": trade["targets"][0]["price"] if trade["targets"] else None,
                "tp2": trade["targets"][1]["price"] if len(trade["targets"]) > 1 else None,
                "tp3": trade["targets"][2]["price"] if len(trade["targets"]) > 2 else None,
                "tp1_rr": trade["targets"][0]["rr_ratio"] if trade["targets"] else None,
                "tp2_rr": trade["targets"][1]["rr_ratio"] if len(trade["targets"]) > 1 else None,
                "tp3_rr": trade["targets"][2]["rr_ratio"] if len(trade["targets"]) > 2 else None,
                "entry_action": trade["entry"]["entry_action"],
                "entry_source": trade["entry"]["entry_source"],
                "entry_score": trade["entry"]["entry_score"],
                # Multi-level support/resistance
                "support_levels": [{"price": s["price"], "strength": s["strength"], "touches": s["touches"]} for s in sr_tech.get("support", [])[:5]],
                "resistance_levels": [{"price": r["price"], "strength": r["strength"], "touches": r["touches"]} for r in sr_tech.get("resistance", [])[:5]],
                "ml_signal": "Buy" if conviction.get("conviction_score", 0) >= 45 else "Avoid",
                "ml_confidence": conviction.get("conviction_score", 0) / 100,
                "ml_medium_price": ml_result.get("forecast", {}).get("Medium", {}).get("price"),
                "ml_low_price": ml_result.get("forecast", {}).get("Low", {}).get("price"),
                "ml_high_price": ml_result.get("forecast", {}).get("High", {}).get("price"),
                "ml_last_date": ml_result.get("last_date", "").strftime("%Y-%m-%d") if hasattr(ml_result.get("last_date", ""), "strftime") else str(ml_result.get("last_date", "")),
                "ml_forecast_end": ml_result.get("forecast_end", "").strftime("%Y-%m-%d") if hasattr(ml_result.get("forecast_end", ""), "strftime") else str(ml_result.get("forecast_end", "")),
                "ml_cone_pct": conviction.get("cone_pct"),
                "candle_signal": patterns.get("latest_signal", "neutral"),
                "candle_score_delta": patterns.get("score_delta", 0),
                "chartscan_signal": cs_result.get("signal", "N/A") if cs_result else "N/A",
                "chartscan_confidence": cs_result.get("confidence") if cs_result else None,
                "chartscan_buy_patterns": cs_result.get("buy_patterns", 0) if cs_result else 0,
                "chartscan_sell_patterns": cs_result.get("sell_patterns", 0) if cs_result else 0,
                "ML Signal": "Buy" if conviction.get("conviction_score", 0) >= 45 else "Avoid",
                "ML Confidence": conviction.get("conviction_score", 0) / 100,
                "ML Medium Price": ml_result.get("forecast", {}).get("Medium", {}).get("price"),
                "ML Conviction": conviction.get("conviction_score", 0),
                "ta_source": ta_source,
                "ta_fetch_time": ta_fetch_time,
                "params_version": params.get("version", "unknown"),
                # Index membership & sentiment
                "index_membership": ticker_index_map.get(raw, "UNINDEX"),
                # Holding / duration
                "hold_weeks_min": trade.get("holding", {}).get("min_weeks"),
                "hold_weeks_max": trade.get("holding", {}).get("max_weeks"),
                "hold_weeks_target": trade.get("holding", {}).get("target_weeks"),
                "hold_label": trade.get("holding", {}).get("duration_label", ""),
                "hold_exit_strategy": trade.get("holding", {}).get("exit_strategy", ""),
                "hold_exit_triggers": trade.get("holding", {}).get("exit_triggers", []),
            }

            # ── OHLCV history for chart (last 3 months) ──
            hist_3m = yf_entry.history[["Open", "High", "Low", "Close", "Volume"]].tail(63).dropna()
            if not hist_3m.empty:
                row["chart_dates"] = [d.strftime("%Y-%m-%d") for d in hist_3m.index]
                row["chart_open"] = [round(float(v), 4) for v in hist_3m["Open"].values]
                row["chart_high"] = [round(float(v), 4) for v in hist_3m["High"].values]
                row["chart_low"] = [round(float(v), 4) for v in hist_3m["Low"].values]
                row["chart_close"] = [round(float(v), 4) for v in hist_3m["Close"].values]
                row["chart_volume"] = [int(v) for v in hist_3m["Volume"].values]

                # If TradingView close differs from yfinance's last close,
                # append today's session with the real-time TV price
                yf_last_close = row["chart_close"][-1]
                today_str = date.today().strftime("%Y-%m-%d")
                if close_ta and row["chart_dates"][-1] != today_str:
                    if round(float(close_ta), 4) != yf_last_close:
                        row["chart_dates"].append(today_str)
                        tv_c = round(float(close_ta), 4)
                        row["chart_open"].append(yf_last_close)
                        row["chart_high"].append(max(yf_last_close, tv_c))
                        row["chart_low"].append(min(yf_last_close, tv_c))
                        row["chart_close"].append(tv_c)
                        row["chart_volume"].append(0)
                        # Update current_price so downstream (trade plan, scoring, etc.)
                        # uses the real-time TradingView price
                        current_price = tv_c
                        row["Current EGP Price"] = tv_c

            # ── Consensus-based recommendation ──
            # Base (technical) signal
            base_rec = base_score["recommendation"]
            # ML signal
            ml_rec = "Buy" if conviction.get("conviction_score", 0) >= 45 else ("Watch" if conviction.get("conviction_score", 0) >= 25 else "Avoid")
            # ChartScan AI signal
            cs_rec = "Buy" if cs_result and cs_result.get("signal") == "Buy" else ("Avoid" if cs_result and cs_result.get("signal") == "Sell" else "Watch")

            buy_votes = sum(1 for r in [base_rec, ml_rec, cs_rec] if r == "Buy")
            watch_votes = sum(1 for r in [base_rec, ml_rec, cs_rec] if r == "Watch")
            avoid_votes = sum(1 for r in [base_rec, ml_rec, cs_rec] if r == "Avoid")

            if buy_votes == 3:
                consensus_rec = "Strong Buy"
                consensus_basis = "All 3 methods agree: Buy"
            elif buy_votes == 2:
                consensus_rec = "Buy"
                agree = [n for n, r in [("Base", base_rec), ("ML", ml_rec), ("ChartScan", cs_rec)] if r == "Buy"]
                consensus_basis = f"2/3 agree Buy: {', '.join(agree)}"
            elif buy_votes == 1:
                consensus_rec = "Watch"
                agree = [n for n, r in [("Base", base_rec), ("ML", ml_rec), ("ChartScan", cs_rec)] if r == "Buy"]
                consensus_basis = f"1/3 Buy ({', '.join(agree)}), others disagree"
            elif watch_votes == 3:
                consensus_rec = "Watch"
                consensus_basis = "All 3 methods: Watch/Neutral"
            else:
                consensus_rec = "Avoid"
                consensus_basis = f"Base={base_rec}, ML={ml_rec}, ChartScan={cs_rec}"

            row["Recommendation"] = consensus_rec
            row["Recommendation Basis"] = consensus_basis
            row["Base Rec"] = base_rec
            row["ML Rec"] = ml_rec
            row["ChartScan Rec"] = cs_rec

            # Add index sentiment
            mem = ticker_index_map.get(raw, "UNINDEX")
            idx_label = "EGX100" if mem == "UNINDEX" else mem
            idx_data = index_sentiment.get(idx_label, {})
            row["index_sentiment"] = idx_data.get("sentiment", "Unknown")
            row["index_close"] = idx_data.get("close")
            row["index_rsi"] = idx_data.get("rsi")
            row["index_adx"] = idx_data.get("adx")
            row["index_reasons"] = idx_data.get("reasons", [])

            rows.append(row)

        except Exception as e:
            log.error("Error processing %s: %s", raw, e, exc_info=True)
            continue

    # 6. Fetch Index Data
    log.info("Fetching EGX index snapshots...")
    for idx_name, idx_symbol in indices_cfg.items():
        # Would fetch from TV cache - simplified for now
        index_rows.append({
            "Index": idx_name,
            "TradingView Symbol": idx_symbol,
            "Status": "pending",
        })

    # 7. Export
    output_path = Path(output_dir) / f"EGX_Signals_{date.today().strftime('%Y%m%d')}.xlsx"
    export_analysis(rows, index_rows, str(output_path))
    append_daily_history(rows, str(output_path))
    export_latest_csv()

    # Generate HTML report for GitHub Pages
    from src.output.html_report import generate_html_report
    html_path = Path(output_dir).parent / "docs" / "index.html"
    generate_html_report(rows, str(html_path))

    # 8. Append to signal store (transform to schema format)
    signal_records = []
    for row in rows:
        signal_records.append({
            "run_date": row.get("Analysis Run Date"),
            "ticker": row.get("Selected Stock"),
            "data_asof": row.get("Data As Of"),
            "close": row.get("Current EGP Price"),
            "close_usd": row.get("Current USD Price"),
            "fx_rate": row.get("fx_rate"),
            "recommendation": row.get("Recommendation"),
            "score": row.get("Score"),
            "score_trend": row.get("Score - Trend"),
            "score_macd": row.get("Score - MACD"),
            "score_rsi": row.get("Score - RSI"),
            "score_volume": row.get("Score - Volume"),
            "score_adi": row.get("Score - ADI"),
            "score_support": row.get("Score - Support"),
            "rsi": row.get("RSI (%)"),
            "macd_bullish": row.get("MACD Bullish (Yes/No)") == "Yes",
            "golden_cross": row.get("Golden Cross (Yes/No)") == "Yes",
            "death_cross": row.get("Death Cross (Yes/No)") == "Yes",
            "diamond_cross": row.get("Diamond Cross (20>50) (Yes/No)") == "Yes",
            "adx": row.get("ADX"),
            "mfi": row.get("MFI"),
            "bb_squeeze": row.get("BB Squeeze") == "Yes",
            "regime": row.get("regime"),
            "fair_value_egp": row.get("Implied Fair Value (EGP)"),
            "fair_value_method": row.get("Fair Value Method"),
            "pe_ttm": row.get("P/E Ratio (TTM)"),
            "undervalued": row.get("Undervalued (Yes/No)") == "Yes",
            "entry_price": row.get("Optimal Entry Price"),
            "stop_loss": row.get("Stop Loss"),
            "tp1": row.get("Take Profit 1"),
            "tp2": row.get("Take Profit 2"),
            "tp3": row.get("Take Profit 3"),
            "tp1_rr": row.get("TP1 Risk/Reward"),
            "tp2_rr": row.get("TP2 Risk/Reward"),
            "tp3_rr": row.get("TP3 Risk/Reward"),
            "ml_signal": row.get("ml_signal"),
            "ml_confidence": row.get("ml_confidence"),
            "ml_medium_price": row.get("ml_medium_price"),
            "ml_low_price": row.get("ml_low_price"),
            "ml_high_price": row.get("ml_high_price"),
            "ml_last_date": row.get("ml_last_date"),
            "ml_forecast_end": row.get("ml_forecast_end"),
            "ml_cone_pct": row.get("ml_cone_pct"),
            "candle_signal": row.get("candle_signal"),
            "candle_score_delta": row.get("candle_score_delta"),
            "chartscan_signal": row.get("chartscan_signal"),
            "chartscan_confidence": row.get("chartscan_confidence"),
            "chartscan_buy_patterns": row.get("chartscan_buy_patterns"),
            "chartscan_sell_patterns": row.get("chartscan_sell_patterns"),
            "score_vwap": row.get("Score - VWAP"),
            "score_volume_profile": row.get("Score - Volume Profile"),
            "vwap": row.get("VWAP"),
            "dist_vwap_pct": row.get("Dist VWAP %"),
            "vp_poc": row.get("Volume Profile POC"),
            "vp_va_high": row.get("Volume Profile VA High"),
            "vp_va_low": row.get("Volume Profile VA Low"),
            "above_poc": row.get("Above POC"),
            "ta_source": row.get("ta_source"),
            "ta_fetch_time": row.get("TA Data As Of"),
            "params_version": row.get("params_version"),
            # Chart data
            "chart_dates": row.get("chart_dates"),
            "chart_open": row.get("chart_open"),
            "chart_high": row.get("chart_high"),
            "chart_low": row.get("chart_low"),
            "chart_close": row.get("chart_close"),
            "chart_volume": row.get("chart_volume"),
        })
    append_signals(signal_records)

    # 9. Send alerts
    send_daily_alert(rows)

    log.info("=" * 60)
    log.info("DAILY ANALYSIS COMPLETE - %d signals generated", len(rows))
    log.info("Output: %s", output_path)
    log.info("=" * 60)

    return rows

# ─────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="EGX Daily Signal Generator")
    parser.add_argument("input_file", nargs="?", default="config/tickers.xlsx",
                        help="Path to input Excel workbook with Selected_Stocks sheet")
    parser.add_argument("-o", "--output-dir", default="output",
                        help="Output directory for Excel files")
    args = parser.parse_args()

    try:
        run_daily_analysis(args.input_file, args.output_dir)
    except Exception as e:
        log.error("Fatal error: %s", e, exc_info=True)
        send_error_alert(str(e), "daily_run")
        sys.exit(1)

if __name__ == "__main__":
    main()