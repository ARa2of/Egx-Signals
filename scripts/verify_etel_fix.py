#!/usr/bin/env python3
"""Verify fixed scoring/consensus on ETEL-like uptrend names.

Replays the daily_run scoring path (yfinance bars + enrich + compute_base_score
+ consensus) against live ETEL.CA data and prints the vote breakdown.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import reload_configs, load_params
from src.signals.technical import enrich
from src.signals.scoring import compute_base_score, resolve_rsi_regime

reload_configs()
params = load_params()
print(f"params version: {params.get('version')}  weights.sum={sum(params['weights'].values())}")
print(f"rsi trending_up: {params['rsi']['trending_up']}")
print()


def fetch_history(ticker: str, period: str = "10y") -> pd.DataFrame:
    hist = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    return hist.dropna(subset=["Close"])


def simulate_ticker(ticker: str, tv_close: float = None) -> dict:
    hist = fetch_history(f"{ticker}.CA")
    raw_df = hist[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()

    current_price = float(tv_close) if tv_close else float(raw_df["Close"].iloc[-1])
    # Mirror daily_run: merge live close into last bar before enrich
    raw_df.iloc[-1, raw_df.columns.get_loc("Close")] = current_price

    tech = enrich(raw_df)
    df_tech = tech["df_enriched"]
    regime = tech["regime"]
    vol_profile = tech["volume_profile"]
    patterns = tech["patterns"]

    def last(col):
        if col in df_tech.columns:
            v = df_tech[col].iloc[-1]
            try:
                fv = float(v)
                return None if pd.isna(fv) else fv
            except (TypeError, ValueError):
                return None
        return None

    ema20, ema50, ema200 = last("EMA_20"), last("EMA_50"), last("EMA_200")
    sma50, sma200 = last("SMA_50"), last("SMA_200")
    rsi, adx = last("RSI"), last("ADX")
    macd, macd_signal = last("MACD"), last("MACD_sig")
    vw = last("VWAP")
    dist_vwap = (current_price - vw) / vw if vw else None

    golden = death = diamond = macd_bull = None
    if sma50 and sma200:
        golden, death = ("Yes", "No") if sma50 > sma200 else ("No", "Yes")
    if ema50 and ema200:
        bull_ema = "Yes" if ema50 > ema200 else "No"
    else:
        bull_ema = None
    if ema20 and ema50:
        diamond = "Yes" if ema20 > ema50 else "No"
    if macd and macd_signal:
        macd_bull = "Yes" if macd > macd_signal else "No"

    # Simple volume multipliers from history
    vol = raw_df["Volume"]
    avg3m = float(vol.tail(63).mean()) if len(vol) >= 63 else float(vol.mean())
    last_vol = float(vol.iloc[-1])
    vol_mult = last_vol / avg3m if avg3m else None
    buy_mult = vol_mult  # approximation for simulation

    # ADL / MFI quick versions
    clv = ((raw_df["Close"] - raw_df["Low"]) - (raw_df["High"] - raw_df["Close"])) / (
        raw_df["High"] - raw_df["Low"] + 1e-10
    )
    adl = (clv * vol).cumsum()
    adl_trend = float(adl.iloc[-1] - adl.iloc[-21]) if len(adl) > 21 else 0.0
    tp = (raw_df["High"] + raw_df["Low"] + raw_df["Close"]) / 3
    mf_pos = (tp * vol).where(tp > tp.shift(1), 0.0).rolling(14).sum()
    mf_neg = (tp * vol).where(tp < tp.shift(1), 0.0).rolling(14).sum()
    mfi = float(100 - 100 / (1 + mf_pos.iloc[-1] / (mf_neg.iloc[-1] + 1e-10)))

    is_near_support = False
    base = compute_base_score(
        current_price=current_price,
        ema20=ema20, ema50=ema50, ema200=ema200,
        macd=macd, macd_signal=macd_signal,
        rsi=rsi, adx=adx, regime=regime.get("regime", "unknown"),
        vol_multiplier=vol_mult,
        buy_vol_multiplier=buy_mult,
        adl_trend=adl_trend, mfi=mfi,
        is_near_support=is_near_support,
        volume_confirmed=False,
        support=None,
        dist_vwap=dist_vwap, vwap=vw,
        above_poc=vol_profile.get("above_poc"),
        poc=vol_profile.get("poc"),
        va_high=vol_profile.get("va_high"),
        va_low=vol_profile.get("va_low"),
        intraday_score=None,
        intraday_details=None,
    )

    base_rec = base["recommendation"]
    # Simulated ML: ETEL-like mean-reversion Avoid (worst case for consensus)
    ml_rec = "Avoid"
    cs_rec = None  # abstain - model file missing

    named = [("Base", base_rec), ("ML", ml_rec), ("ChartScan", cs_rec)]
    active = [(n, r) for n, r in named if r is not None]
    buy_v = sum(1 for _, r in active if r == "Buy")
    avoid_v = sum(1 for _, r in active if r == "Avoid")
    watch_v = sum(1 for _, r in active if r == "Watch")
    agree_buy = [n for n, r in active if r == "Buy"]

    strong = (
        base_rec == "Buy"
        and death != "Yes"
        and golden == "Yes"
        and diamond == "Yes"
        and (adx or 0) >= 30
        and macd_bull == "Yes"
    )

    if strong and cs_rec != "Avoid":
        cons = "Buy"
        basis = (
            f"Strong technical base (score={base['raw_score']:.1f}, golden+diamond, "
            f"ADX={adx:.0f}, MACD bullish) - ML={ml_rec}, ChartScan=abstain"
        )
    elif buy_v >= 2 and buy_v == len(active) and len(active) >= 2:
        cons, basis = "Strong Buy", f"All {len(active)} active Buy: {agree_buy}"
    elif buy_v >= 2:
        cons, basis = "Buy", f"{buy_v}/{len(active)} active Buy: {agree_buy}"
    elif buy_v == 1 and avoid_v == 0:
        cons, basis = "Buy", f"1 Buy, no Avoid - upgraded"
    elif buy_v == 1:
        cons, basis = "Watch", f"1 Buy, others disagree"
    elif watch_v >= 1 and avoid_v == 0:
        cons, basis = "Watch", "No Buy, no Avoid"
    elif avoid_v >= 1 and buy_v == 0:
        cons, basis = "Avoid", ", ".join(f"{n}={r}" for n, r in named if r)
    else:
        cons, basis = "Watch", "Insufficient signals"

    # RSI hard cap with regime map + trend exception
    rsi_cfg = params["rsi"]
    regime_key = regime.get("regime", "ranging")
    if regime_key not in rsi_cfg:
        regime_key = "trending_up" if regime_key == "trending" else "ranging"
    ob = rsi_cfg.get(regime_key, rsi_cfg["ranging"]).get("overbought", 80)
    ema_bull_flag = bool(ema50 and ema200 and ema50 > ema200)
    trend_exc = bool((adx or 0) >= 30 and ema_bull_flag and current_price > (ema50 or 0))
    capped = False
    if rsi and rsi > ob and cons in ("Buy", "Strong Buy"):
        if trend_exc:
            basis += f"; RSI {rsi:.1f}>{ob} tolerated (ADX trend exception)"
        else:
            cons, capped = "Watch", True
            basis = f"RSI overbought {rsi:.1f}>{ob} - downgraded"

    return {
        "ticker": ticker,
        "price": current_price,
        "regime": regime.get("regime"),
        "resolve_rsi_regime": resolve_rsi_regime(regime.get("regime")),
        "rsi": rsi,
        "adx": adx,
        "golden": golden,
        "diamond": diamond,
        "macd_bull": macd_bull,
        "death": death,
        "dist_vwap_pct": None if dist_vwap is None else dist_vwap * 100,
        "base_rec": base_rec,
        "score": base["raw_score"],
        "breakdown": base["score_breakdown"],
        "ml_rec": ml_rec,
        "cs_rec": cs_rec,
        "strong_technicals": strong,
        "trend_exception": trend_exc,
        "consensus": cons,
        "basis": basis,
        "rsi_capped": capped,
        "reasons": base["recommendation_basis"],
    }


def show(r: dict) -> None:
    print("=" * 72)
    print(f"{r['ticker']}  price={r['price']:.2f}  regime={r['regime']} "
          f"(rsi_regime->{r['resolve_rsi_regime']})")
    print(f"  RSI={r['rsi']:.1f}  ADX={r['adx']:.1f}  golden={r['golden']} "
          f"diamond={r['diamond']} macd_bull={r['macd_bull']} death={r['death']}")
    print(f"  dist_VWAP={r['dist_vwap_pct']:.1f}%" if r['dist_vwap_pct'] is not None else "  dist_VWAP=n/a")
    print(f"  score={r['score']:.2f}  base_rec={r['base_rec']}")
    bd = r["breakdown"]
    print("  breakdown: " + ", ".join(f"{k}={v}" for k, v in bd.items()))
    print(f"  votes: Base={r['base_rec']} ML={r['ml_rec']} ChartScan={r['cs_rec'] or 'abstain'}")
    print(f"  strong_technicals={r['strong_technicals']}  rsi_trend_exception={r['trend_exception']}")
    print(f"  CONSENSUS -> {r['consensus']}  ({r['basis']})")
    print(f"  base reasons: {r['reasons'][:200]}")
    print()


if __name__ == "__main__":
    # ETEL: try live TV-like close from yfinance last bar (already merged above)
    results = []
    for t in ["ETEL", "CPCI", "SWDY", "AMOC", "MASR"]:
        try:
            results.append(simulate_ticker(t))
        except Exception as e:
            print(f"{t}: FAILED {e}")

    for r in results:
        show(r)

    buys = [r["ticker"] for r in results if r["consensus"] in ("Buy", "Strong Buy")]
    print("=" * 72)
    print(f"Buy/Strong Buy from sample: {buys or 'NONE'}")
    etel = next((r for r in results if r["ticker"] == "ETEL"), None)
    if etel:
        ok = etel["consensus"] in ("Buy", "Strong Buy")
        print(f"ETEL verification: {'PASS' if ok else 'FAIL'} -> {etel['consensus']}")
        sys.exit(0 if ok else 1)
    sys.exit(2)
