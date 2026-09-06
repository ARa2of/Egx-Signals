#!/usr/bin/env python3
"""
Standalone ML Runner
Runs ML forecasting for specific tickers with detailed output
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_params
from src.data.loader import read_ticker_list, download_all, download_fx
from src.signals.technical import enrich
from src.signals.ml import run_ml_pipeline, compute_ml_conviction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ml_runner")

def main():
    parser = argparse.ArgumentParser(description="EGX ML Forecast Runner")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols (e.g., OBRI TMGH)")
    parser.add_argument("--input", default="config/tickers.xlsx",
                        help="Input Excel file with ticker list")
    parser.add_argument("--peers", nargs="*", default=[],
                        help="Peer tickers for ML features (default: all from input)")
    parser.add_argument("--horizon", type=int, default=5,
                        help="Forecast horizon in days (default: 5)")
    parser.add_argument("--lookback", type=int, default=730,
                        help="Training lookback in days (default: 730)")
    parser.add_argument("--plot", action="store_true",
                        help="Show forecast chart (requires matplotlib)")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("EGX ML FORECAST RUNNER")
    log.info("=" * 60)

    # Load params
    params = load_params()

    # Resolve input path relative to project root
    project_root = Path(__file__).parent.parent
    input_path = project_root / args.input
    
    # Load tickers from input file
    all_tickers = read_ticker_list(str(input_path), "Selected_Stocks")
    
    # Also load sector groups tickers for ML peer data
    sector_file = project_root / params["ml"].get("sector_groups_file", "config/EGX_Stock_Groups.xlsx")
    if sector_file.exists():
        import pandas as pd
        sector_df = pd.read_excel(sector_file)
        sector_tickers = sector_df["Ticker"].tolist()
        all_tickers = list(set(all_tickers + sector_tickers))
        log.info("Loaded %d tickers from input + %d from sector groups", len(read_ticker_list(str(input_path), "Selected_Stocks")), len(sector_tickers))
    
    # Download data for all tickers
    log.info("Downloading data for %d tickers...", len(all_tickers))
    yf_cache = {}
    download_all(all_tickers, yf_cache)
    
    valid_tickers = [t for t in all_tickers if yf_cache.get(t) and yf_cache[t].ok]
    log.info("Valid data for %d tickers", len(valid_tickers))

    # Determine peer tickers (if explicitly provided via --peers)
    explicit_peers = None
    if args.peers:
        explicit_peers = [t for t in args.peers if t in valid_tickers]
        log.info("Using %d explicit peers: %s", len(explicit_peers), explicit_peers)
    else:
        log.info("Peer selection: sector-based (max %d per sector)", params["ml"].get("max_peers", 4))

    # Process each target ticker
    for target in args.tickers:
        target = target.upper()
        if target not in valid_tickers:
            log.error("No valid data for %s", target)
            continue

        # Build peer close DataFrame (sector-based or explicit)
        if explicit_peers:
            peer_tickers = explicit_peers
        else:
            from src.signals.ml import get_sector_peers
            peer_tickers = get_sector_peers(target, valid_tickers)

        peer_close = pd.DataFrame({
            t: yf_cache[t].history["Close"] for t in peer_tickers
            if t in yf_cache and yf_cache[t].ok
        })

        yf_entry = yf_cache[target]
        log.info("\n--- Processing %s ---", target)

        # Technical enrichment
        raw_df = yf_entry.history[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
        tech = enrich(raw_df)
        df_tech = tech["df_enriched"]

        # Run ML pipeline
        log.info("Running ML pipeline...")
        ml_result = run_ml_pipeline(df_tech, peer_close, target, {})

        if "error" in ml_result:
            log.error("ML failed: %s", ml_result["error"])
            continue

        # Print results
        forecast = ml_result.get("forecast", {})
        eval_results = ml_result.get("eval", {}).get("results", {})
        last_price = ml_result.get("last_price", 0)
        last_date = ml_result.get("last_date", "N/A")

        log.info("Last Price: %.3f EGP (as of %s)", last_price, last_date)
        log.info("Forecast Horizon: %d days", args.horizon)
        log.info("Training Rows: %d", ml_result.get("n_train_rows", 0))
        log.info("Features: %d", ml_result.get("n_features", 0))

        print("\nPRICE FORECAST:")
        for name in ["Low", "Medium", "High"]:
            f = forecast.get(name, {})
            if f:
                print(f"  {name:6s}: {f['price']:>8.3f} EGP  ({f['return_pct']:+.2f}%)")

        print("\nBACKTEST MAE:")
        for name in ["Low", "Medium", "High"]:
            mae = eval_results.get(name, {}).get("mae", 0)
            print(f"  {name:6s}: {mae*100:.2f}%")

        # ML Conviction
        avg_mae = ml_result.get("eval", {}).get("avg_mae", {})
        xgb_mae = avg_mae.get("xgb", {}).get("Medium", 0)
        lgbm_mae = avg_mae.get("lgbm", {}).get("Medium", 0)
        mae_pct = ((xgb_mae + lgbm_mae) / 2) * 100 if (xgb_mae or lgbm_mae) else 0
        conviction = compute_ml_conviction(
            forecast, last_price, mae_pct,
            direction_accuracy=ml_result.get("direction_accuracy"),
            ensemble_weights=ml_result.get("ensemble_weights"),
        )
        print(f"\nML CONVICTION: {conviction['conviction_score']}/100 — {conviction['conviction_label']}")
        print(f"   Upside: {conviction['upside_pct']:+.1f}% | Cone: {conviction['cone_pct']:.1f}% | Asymmetry: {conviction['asymmetry']:.2f}x")
        print(f"   {conviction['recommendation']}")

        # Feature importances
        if "importances" in ml_result:
            print("\nTOP 10 FEATURES:")
            imp = ml_result["importances"].head(10)
            for _, row in imp.iterrows():
                print(f"  {row['Feature']:30s}: {row['Importance']:.4f}")

        if args.plot:
            try:
                import matplotlib.pyplot as plt
                # Plot forecast cone
                fig, ax = plt.subplots(figsize=(10, 5))
                # Historical prices
                hist = yf_entry.history["Close"].tail(60)
                ax.plot(hist.index, hist.values, label="Historical", color="gray")
                # Forecast
                from pandas.tseries.offsets import BDay
                f_dates = [last_date + BDay(i+1) for i in range(args.horizon)]
                for name, color in [("Low", "red"), ("Medium", "yellow"), ("High", "green")]:
                    f = forecast.get(name, {})
                    if f:
                        ax.axhline(f['price'], color=color, linestyle="--", alpha=0.5, label=name)
                ax.set_title(f"{target} - ML Forecast ({args.horizon}d)")
                ax.legend()
                plt.show()
            except Exception as e:
                log.warning("Plotting failed: %s", e)

    log.info("\n" + "=" * 60)
    log.info("ML FORECAST COMPLETE")

if __name__ == "__main__":
    main()