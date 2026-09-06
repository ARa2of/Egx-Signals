#!/usr/bin/env python3
"""
Monthly Fundamentals Updater
Refreshes fundamental data from yfinance and caches to Excel
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.fundamentals import (
    fetch_fundamentals, save_fundamentals, load_fundamentals,
    get_sector_pe_stats, is_stale, SECTOR_GROUPS_PATH
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fundamentals")


def main():
    parser = argparse.ArgumentParser(description="EGX Fundamentals Updater")
    parser.add_argument("--force", action="store_true",
                        help="Force refresh even if cache is fresh")
    parser.add_argument("--tickers", nargs="*",
                        help="Specific tickers to update (default: all from sector groups)")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("EGX FUNDAMENTALS UPDATER")
    log.info("=" * 60)

    # Check staleness
    if not args.force and not is_stale(max_age_days=25):
        log.info("Cache is fresh (less than 25 days old). Use --force to refresh.")
        return

    # Get tickers from sector groups
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        if not SECTOR_GROUPS_PATH.exists():
            log.error("Sector groups file not found: %s", SECTOR_GROUPS_PATH)
            sys.exit(1)
        import pandas as pd
        df = pd.read_excel(SECTOR_GROUPS_PATH)
        tickers = df["Ticker"].tolist()

    log.info("Fetching fundamentals for %d tickers...", len(tickers))

    # Fetch from yfinance
    fundamentals = fetch_fundamentals(tickers)

    # Save to cache
    save_fundamentals(fundamentals)

    # Print summary
    log.info("=" * 60)
    log.info("FUNDAMENTALS SUMMARY:")
    log.info("  Total tickers: %d", len(fundamentals))

    # Sector P/E stats
    pe_stats = get_sector_pe_stats(fundamentals)
    if pe_stats:
        log.info("  Sector P/E Stats:")
        for sector, stats in sorted(pe_stats.items()):
            log.info("    %s: avg=%.1f, median=%.1f, count=%d",
                     sector, stats["avg_pe"], stats["median_pe"], stats["count"])

    # Top/bottom P/E
    valid_pe = fundamentals[fundamentals["trailingPE"].notna()].copy()
    if not valid_pe.empty:
        valid_pe = valid_pe.sort_values("trailingPE")
        log.info("  Lowest P/E:")
        for _, row in valid_pe.head(5).iterrows():
            log.info("    %s: %.1f (sector avg: %.1f, vs sector: %+.1f%%)",
                     row["ticker"], row["trailingPE"],
                     row.get("sector_avg_trailingPE", 0),
                     row.get("pe_vs_sector", 0))
        log.info("  Highest P/E:")
        for _, row in valid_pe.tail(5).iterrows():
            log.info("    %s: %.1f (sector avg: %.1f, vs sector: %+.1f%%)",
                     row["ticker"], row["trailingPE"],
                     row.get("sector_avg_trailingPE", 0),
                     row.get("pe_vs_sector", 0))

    log.info("=" * 60)
    log.info("UPDATE COMPLETE")


if __name__ == "__main__":
    main()
