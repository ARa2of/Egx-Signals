#!/usr/bin/env python3
"""
Intraday Predictive Validation
Checks if intraday signals predict future price movement
"""
import sys; sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from src.store.signal_store import load_store
from src.data.intraday_analysis import run_intraday_analysis

store = load_store()
buys = store[store["recommendation"].isin(["Buy", "Strong Buy", "Watch"])].copy()

print("=" * 70)
print("INTRADAY PREDICTIVE VALIDATION")
print("=" * 70)
print(f"\nTotal Buy/Watch signals: {len(buys)}")

# Sample signals to validate
sample = buys.sample(n=min(50, len(buys)), random_state=42)

results = []
for _, sig in sample.iterrows():
    ticker = sig["ticker"]
    run_date = sig["run_date"]
    if hasattr(run_date, "date"):
        run_date = run_date.date()
    
    # Run intraday analysis
    intraday = run_intraday_analysis(ticker, sig["recommendation"])
    if intraday.get("skipped"):
        continue
    
    # Get future price data (5 days after signal)
    from datetime import timedelta
    future_start = run_date + timedelta(days=1)
    future_end = run_date + timedelta(days=6)
    
    try:
        import yfinance as yf
        yf_ticker = f"{ticker}.CA"
        future = yf.download(yf_ticker, start=str(future_start), end=str(future_end),
                              progress=False, auto_adjust=True)
        if future.empty:
            continue
        if isinstance(future.columns, pd.MultiIndex):
            future.columns = future.columns.get_level_values(0)
        
        future_close = future["Close"].iloc[-1]
        signal_close = sig.get("close", 0) or 0
        if signal_close <= 0:
            continue
        
        future_return = (future_close - signal_close) / signal_close * 100
        
        results.append({
            "ticker": ticker,
            "intraday_score": intraday.get("intraday_score", 50),
            "volume_skew": intraday.get("volume_profile", {}).get("volume_skew", 0),
            "momentum": intraday.get("momentum", {}).get("momentum_score", 50),
            "vwap_score": intraday.get("vwap", {}).get("vwap_score", 50),
            "ad_signal": intraday.get("accumulation_distribution", {}).get("ad_signal", "neutral"),
            "ad_score": intraday.get("accumulation_distribution", {}).get("ad_score", 0),
            "future_return": future_return,
        })
    except Exception as e:
        continue

if not results:
    print("\nNo valid results for validation.")
    sys.exit(0)

rdf = pd.DataFrame(results)
print(f"\nValidated {len(rdf)} signals")

# ── Correlation Analysis ──
print("\n" + "=" * 70)
print("CORRELATION WITH FUTURE RETURNS")
print("=" * 70)

for col, label in [("intraday_score", "Intraday Score"),
                    ("volume_skew", "Volume Skew"),
                    ("momentum", "Momentum"),
                    ("vwap_score", "VWAP Score"),
                    ("ad_score", "AD Score")]:
    if col in rdf.columns:
        corr = rdf[col].corr(rdf["future_return"])
        print(f"  {label}: {corr:+.3f}")

# ── AD Signal Performance ──
print("\n" + "=" * 70)
print("AD SIGNAL vs FUTURE RETURNS")
print("=" * 70)

for signal in ["bullish", "bearish", "neutral"]:
    subset = rdf[rdf["ad_signal"] == signal]
    if len(subset) > 0:
        avg_return = subset["future_return"].mean()
        win_rate = (subset["future_return"] > 0).mean()
        print(f"  {signal:10s}: {len(subset):3d} trades, avg_return={avg_return:+.2f}%, win_rate={win_rate:.0%}")

# ── Intraday Score Quartiles ──
print("\n" + "=" * 70)
print("INTRADAY SCORE QUARTILES vs FUTURE RETURNS")
print("=" * 70)

rdf["score_quartile"] = pd.qcut(rdf["intraday_score"], q=4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
for q in ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]:
    subset = rdf[rdf["score_quartile"] == q]
    if len(subset) > 0:
        avg_return = subset["future_return"].mean()
        win_rate = (subset["future_return"] > 0).mean()
        print(f"  {q:10s}: {len(subset):3d} trades, avg_return={avg_return:+.2f}%, win_rate={win_rate:.0%}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("Positive correlation = intraday signal predicts future returns")
print("Higher Q4 return = intraday score is predictive")
