"""
Fundamentals Cache for EGX Signals
Monthly-refreshed fundamental data stored in Excel
"""
import logging
from pathlib import Path
from datetime import date, timedelta
from typing import Dict, Optional, List
import pandas as pd

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
FUNDAMENTALS_PATH = _PROJECT_ROOT / "data" / "fundamentals.xlsx"
SECTOR_GROUPS_PATH = _PROJECT_ROOT / "config" / "EGX_Stock_Groups.xlsx"

FUNDAMENTAL_FIELDS = [
    "trailingPE", "priceToBook", "marketCap", "sharesOutstanding",
    "bookValue", "dividendYield", "trailingAnnualDividendYield",
    "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
    "fiftyDayAverage", "twoHundredDayAverage",
    "priceToSalesTrailing12Months", "enterpriseToEbitda",
]


def _load_sector_map() -> Dict[str, str]:
    """Load ticker -> sector mapping."""
    if not SECTOR_GROUPS_PATH.exists():
        return {}
    df = pd.read_excel(SECTOR_GROUPS_PATH)
    return dict(zip(df["Ticker"], df["Sector"]))


def fetch_fundamentals(tickers: List[str]) -> pd.DataFrame:
    """Fetch fundamental data from yfinance for a list of tickers."""
    import yfinance as yf

    records = []
    for ticker in tickers:
        try:
            yf_ticker = f"{ticker}.CA"
            t = yf.Ticker(yf_ticker)
            info = t.info or {}

            row = {"ticker": ticker}
            for field in FUNDAMENTAL_FIELDS:
                row[field] = info.get(field)
            records.append(row)
        except Exception as e:
            log.warning("Failed to fetch fundamentals for %s: %s", ticker, e)
            records.append({"ticker": ticker})

    df = pd.DataFrame(records)

    # Add sector mapping
    sector_map = _load_sector_map()
    df["sector"] = df["ticker"].map(sector_map)

    # Compute sector averages
    numeric_fields = ["trailingPE", "priceToBook", "marketCap", "dividendYield",
                      "trailingAnnualDividendYield"]
    for field in numeric_fields:
        if field in df.columns:
            df[f"sector_avg_{field}"] = df.groupby("sector")[field].transform("mean")
            df[f"sector_median_{field}"] = df.groupby("sector")[field].transform("median")
            if field == "trailingPE":
                df["pe_vs_sector"] = df.apply(
                    lambda r: round((r["trailingPE"] - r["sector_avg_trailingPE"]) / r["sector_avg_trailingPE"] * 100, 1)
                    if pd.notna(r.get("trailingPE")) and pd.notna(r.get("sector_avg_trailingPE")) and r["sector_avg_trailingPE"] != 0
                    else None, axis=1
                )

    df["last_updated"] = date.today().isoformat()
    return df


def save_fundamentals(df: pd.DataFrame) -> None:
    """Save fundamentals to Excel cache."""
    FUNDAMENTALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(FUNDAMENTALS_PATH, index=False, sheet_name="Fundamentals")
    log.info("Saved fundamentals for %d tickers to %s", len(df), FUNDAMENTALS_PATH)


def load_fundamentals() -> pd.DataFrame:
    """Load cached fundamentals. Returns empty DataFrame if not found or stale."""
    if not FUNDAMENTALS_PATH.exists():
        log.warning("No fundamentals cache found at %s", FUNDAMENTALS_PATH)
        return pd.DataFrame()

    try:
        df = pd.read_excel(FUNDAMENTALS_PATH, sheet_name="Fundamentals")
        if "last_updated" in df.columns:
            last_update = pd.to_datetime(df["last_updated"]).max()
            age_days = (pd.Timestamp.now() - last_update).days
            if age_days > 35:
                log.warning("Fundamentals cache is %d days old (stale)", age_days)
            else:
                log.info("Loaded fundamentals cache: %d tickers, %d days old", len(df), age_days)
        return df
    except Exception as e:
        log.warning("Failed to load fundamentals: %s", e)
        return pd.DataFrame()


def get_fundamental(ticker: str, field: str, df: Optional[pd.DataFrame] = None) -> Optional[float]:
    """Get a specific fundamental field for a ticker."""
    if df is None:
        df = load_fundamentals()
    if df.empty:
        return None
    row = df[df["ticker"] == ticker]
    if row.empty:
        return None
    val = row.iloc[0].get(field)
    return val if pd.notna(val) else None


def get_sector_pe_stats(df: Optional[pd.DataFrame] = None) -> Dict[str, Dict]:
    """Get P/E stats by sector."""
    if df is None:
        df = load_fundamentals()
    if df.empty or "sector" not in df.columns:
        return {}

    stats = {}
    for sector, group in df.groupby("sector"):
        pe_values = group["trailingPE"].dropna()
        if len(pe_values) > 0:
            stats[sector] = {
                "avg_pe": round(pe_values.mean(), 2),
                "median_pe": round(pe_values.median(), 2),
                "min_pe": round(pe_values.min(), 2),
                "max_pe": round(pe_values.max(), 2),
                "count": len(pe_values),
            }
    return stats


def is_stale(max_age_days: int = 35) -> bool:
    """Check if the fundamentals cache is stale."""
    if not FUNDAMENTALS_PATH.exists():
        return True
    try:
        df = pd.read_excel(FUNDAMENTALS_PATH, sheet_name="Fundamentals", nrows=1)
        if "last_updated" in df.columns:
            last_update = pd.to_datetime(df["last_updated"].iloc[0])
            return (pd.Timestamp.now() - last_update).days > max_age_days
    except Exception:
        return True
    return True
