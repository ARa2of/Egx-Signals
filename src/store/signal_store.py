"""
Signal Store for EGX Signals
Append-only Parquet storage with fast queries
"""
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, date, timedelta
import logging

log = logging.getLogger(__name__)

# Resolve paths relative to project root
_PROJECT_ROOT = Path(__file__).parent.parent.parent
SIGNAL_STORE_PATH = _PROJECT_ROOT / "data" / "signals.parquet"
LATEST_CSV_PATH = _PROJECT_ROOT / "output" / "latest_signals.csv"

# ─────────────────────────────────────────────────────────────
# Schema Definition
# ─────────────────────────────────────────────────────────────
SIGNAL_SCHEMA = pa.schema([
    # Identity
    ("run_date", pa.date32()),
    ("ticker", pa.string()),
    ("data_asof", pa.date32()),

    # Price context
    ("close", pa.float64()),
    ("close_usd", pa.float64()),
    ("fx_rate", pa.float64()),

    # Core signals
    ("recommendation", pa.string()),
    ("score", pa.float64()),
    ("score_trend", pa.float64()),
    ("score_macd", pa.float64()),
    ("score_rsi", pa.float64()),
    ("score_volume", pa.float64()),
    ("score_adi", pa.float64()),
    ("score_support", pa.float64()),

    # Technical state
    ("rsi", pa.float64()),
    ("macd_bullish", pa.bool_()),
    ("golden_cross", pa.bool_()),
    ("death_cross", pa.bool_()),
    ("diamond_cross", pa.bool_()),
    ("adx", pa.float64()),
    ("mfi", pa.float64()),
    ("bb_squeeze", pa.bool_()),
    ("regime", pa.string()),

    # Fundamentals
    ("fair_value_egp", pa.float64()),
    ("fair_value_method", pa.string()),
    ("pe_ttm", pa.float64()),
    ("undervalued", pa.bool_()),

    # Trade plan
    ("entry_price", pa.float64()),
    ("stop_loss", pa.float64()),
    ("tp1", pa.float64()),
    ("tp2", pa.float64()),
    ("tp3", pa.float64()),
    ("tp1_rr", pa.float64()),
    ("tp2_rr", pa.float64()),
    ("tp3_rr", pa.float64()),

    # ML
    ("ml_signal", pa.string()),
    ("ml_confidence", pa.float64()),
    ("ml_medium_price", pa.float64()),
    ("ml_cone_pct", pa.float64()),

    # Patterns
    ("candle_signal", pa.string()),
    ("candle_score_delta", pa.int32()),

    # ChartScan AI
    ("chartscan_signal", pa.string()),
    ("chartscan_confidence", pa.float64()),
    ("chartscan_buy_patterns", pa.int32()),
    ("chartscan_sell_patterns", pa.int32()),

    # VWAP & Volume Profile
    ("score_vwap", pa.float64()),
    ("score_volume_profile", pa.float64()),
    ("vwap", pa.float64()),
    ("dist_vwap_pct", pa.float64()),
    ("vp_poc", pa.float64()),
    ("vp_va_high", pa.float64()),
    ("vp_va_low", pa.float64()),
    ("above_poc", pa.bool_()),

    # Metadata
    ("ta_source", pa.string()),
    ("ta_fetch_time", pa.string()),
    ("params_version", pa.string()),
    ("stale", pa.bool_()),

    # Outcome tracking
    ("outcome", pa.string()),
    ("outcome_date", pa.date32()),
    ("r_multiple", pa.float64()),
    ("hold_days", pa.int32()),
    ("mfe_pct", pa.float64()),
    ("mae_pct", pa.float64()),
    ("max_price", pa.float64()),
    ("min_price", pa.float64()),
    ("pnl_pct", pa.float64()),
])

# ─────────────────────────────────────────────────────────────
# Store Operations
# ─────────────────────────────────────────────────────────────
def _ensure_store_exists():
    """Create empty store with schema if it doesn't exist."""
    if not SIGNAL_STORE_PATH.exists():
        SIGNAL_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        empty_table = pa.Table.from_pydict({field.name: [] for field in SIGNAL_SCHEMA}, schema=SIGNAL_SCHEMA)
        pq.write_table(empty_table, SIGNAL_STORE_PATH)
        log.info("Created new signal store at %s", SIGNAL_STORE_PATH)

def append_signals(signals: List[Dict]) -> int:
    """Append new signals to the store. Returns number of rows added."""
    _ensure_store_exists()

    if not signals:
        return 0

    # Convert to DataFrame
    df = pd.DataFrame(signals)

    # Ensure all schema columns exist
    for field in SIGNAL_SCHEMA:
        col = field.name
        if col not in df.columns:
            df[col] = None

    # Convert date columns
    for col in ["run_date", "data_asof"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.date

    # Convert bool columns
    for col in ["macd_bullish", "golden_cross", "death_cross", "diamond_cross", "bb_squeeze", "undervalued", "stale"]:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    # Reorder columns to match schema
    df = df[[f.name for f in SIGNAL_SCHEMA]]

    # Read existing store
    existing = pq.read_table(SIGNAL_STORE_PATH).to_pandas()

    # Combine and deduplicate (keep latest for same ticker+run_date)
    combined = pd.concat([existing, df], ignore_index=True)
    combined.drop_duplicates(subset=["run_date", "ticker"], keep="last", inplace=True)

    # Sort by run_date, ticker
    combined.sort_values(["run_date", "ticker"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    # Write back
    table = pa.Table.from_pandas(combined, schema=SIGNAL_SCHEMA)
    pq.write_table(table, SIGNAL_STORE_PATH)

    log.info("Appended %d signals to store (total: %d)", len(df), len(combined))
    return len(df)

def load_store() -> pd.DataFrame:
    """Load entire signal store as DataFrame."""
    _ensure_store_exists()
    try:
        table = pq.read_table(SIGNAL_STORE_PATH)
        # Schema migration: add missing columns
        missing = [f for f in SIGNAL_SCHEMA if f.name not in table.column_names]
        if missing:
            for field in missing:
                default = pa.nulls(len(table), type=field.type)
                table = table.append_column(field.name, default)
            # Reorder columns to match schema before casting
            table = table.select([f.name for f in SIGNAL_SCHEMA])
            table = table.cast(SIGNAL_SCHEMA)
        return table.to_pandas()
    except Exception as e:
        log.warning("Signal store corrupted (%s), recreating...", e)
        SIGNAL_STORE_PATH.unlink(missing_ok=True)
        _ensure_store_exists()
        return pq.read_table(SIGNAL_STORE_PATH).to_pandas()

def load_latest_signals(run_date: Optional[date] = None) -> pd.DataFrame:
    """Load signals for a specific date (default: latest)."""
    df = load_store()
    if df.empty:
        return df

    if run_date is None:
        run_date = df["run_date"].max()

    # Ensure run_date is comparable (convert to datetime64 if needed)
    if hasattr(run_date, 'date'):
        run_date = run_date.date()
    df["run_date"] = pd.to_datetime(df["run_date"]).dt.date

    return df[df["run_date"] == run_date].copy()

def load_ticker_history(ticker: str, start_date: Optional[date] = None, end_date: Optional[date] = None) -> pd.DataFrame:
    """Load all signals for a specific ticker within date range."""
    df = load_store()
    if df.empty:
        return df

    df = df[df["ticker"] == ticker].copy()
    if start_date:
        df = df[df["run_date"] >= start_date]
    if end_date:
        df = df[df["run_date"] <= end_date]

    return df.sort_values("run_date")

def export_latest_csv(run_date: Optional[date] = None) -> str:
    """Export latest signals to CSV for dashboard consumption."""
    df = load_latest_signals(run_date)
    if df.empty:
        return ""

    LATEST_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LATEST_CSV_PATH, index=False)
    log.info("Exported %d latest signals to %s", len(df), LATEST_CSV_PATH)
    return str(LATEST_CSV_PATH)

def get_signal_stats() -> Dict[str, Any]:
    """Get summary statistics of the signal store."""
    df = load_store()
    if df.empty:
        return {"total_signals": 0, "unique_tickers": 0, "date_range": None, "recommendations": {}}

    return {
        "total_signals": len(df),
        "unique_tickers": df["ticker"].nunique(),
        "date_range": {"start": str(df["run_date"].min()), "end": str(df["run_date"].max())},
        "recommendations": df["recommendation"].value_counts().to_dict(),
        "avg_score": float(df["score"].mean()),
        "params_versions": df["params_version"].value_counts().to_dict(),
    }

def simulate_outcomes(trades: pd.DataFrame, horizon_days: int = 21,
                      cost_bps: float = 10.0, slippage_bps: float = 5.0) -> pd.DataFrame:
    """Simulate outcomes for closed trades using historical price data."""
    if trades.empty:
        return trades

    import yfinance as yf

    trades = trades.copy()
    cost_pct = (cost_bps + slippage_bps) / 10000.0

    for idx, row in trades.iterrows():
        ticker = row["ticker"]
        entry = row.get("entry_price")
        sl = row.get("stop_loss")
        tps = [row.get(f"tp{i}") for i in (1, 2, 3)]
        run_dt = row["run_date"]
        if hasattr(run_dt, "date"):
            run_dt = run_dt.date()

        if not all([entry, sl]) or entry == sl:
            trades.at[idx, "outcome"] = "invalid"
            continue

        try:
            start = run_dt + timedelta(days=1)
            end = run_dt + timedelta(days=horizon_days + 10)
            df = yf.download(ticker, start=str(start), end=str(end),
                             progress=False, auto_adjust=True)
            if df.empty:
                trades.at[idx, "outcome"] = "no_data"
                continue

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            outcome = "expired"
            exit_price = entry
            hold = len(df)
            max_p = float(df["High"].max())
            min_p = float(df["Low"].min())

            for _, bar in df.iterrows():
                low = float(bar["Low"])
                high = float(bar["High"])
                close = float(bar["Close"])

                # Check stop loss first (worst case)
                if low <= sl:
                    outcome = "stop_loss"
                    exit_price = sl
                    break

                # Check take profits (best first)
                for i, tp in enumerate(tps):
                    if tp and high >= tp:
                        outcome = f"tp{i+1}_hit"
                        exit_price = tp
                        break

                if outcome != "expired":
                    break

            if outcome == "expired":
                exit_price = float(df["Close"].iloc[-1])

            r_mult = (exit_price - entry) / abs(entry - sl) if entry != sl else 0
            r_mult -= cost_pct
            pnl = (exit_price - entry) / entry * 100 if entry else 0

            trades.at[idx, "outcome"] = outcome
            trades.at[idx, "outcome_date"] = run_dt + timedelta(days=horizon_days)
            trades.at[idx, "r_multiple"] = round(r_mult, 3)
            trades.at[idx, "hold_days"] = hold
            trades.at[idx, "mfe_pct"] = round((max_p - entry) / entry * 100, 2) if entry else 0
            trades.at[idx, "mae_pct"] = round((min_p - entry) / entry * 100, 2) if entry else 0
            trades.at[idx, "max_price"] = max_p
            trades.at[idx, "min_price"] = min_p
            trades.at[idx, "pnl_pct"] = round(pnl, 2)

        except Exception as e:
            log.warning("Outcome eval failed for %s: %s", ticker, e)
            trades.at[idx, "outcome"] = "error"

    return trades


def update_outcomes(horizon_days: int = 21, cost_bps: float = 10.0,
                    slippage_bps: float = 5.0) -> int:
    """Update pending signals with actual outcomes. Returns number of rows updated."""
    _ensure_store_exists()
    df = load_store()
    if df.empty:
        return 0

    today = date.today()
    pending_mask = df["outcome"].isna() | (df["outcome"] == "pending")
    pending = df[pending_mask].copy()

    if pending.empty:
        log.info("No pending outcomes to evaluate")
        return 0

    # Only evaluate signals old enough to have outcomes
    eval_cutoff = today - timedelta(days=horizon_days + 5)
    pending = pd.to_datetime(pending["run_date"]).dt.date
    evaluable = df[pending_mask & (pd.to_datetime(df["run_date"]).dt.date <= eval_cutoff)].copy()

    if evaluable.empty:
        log.info("No signals old enough for outcome evaluation (need %d+ days)", horizon_days)
        return 0

    log.info("Evaluating outcomes for %d signals...", len(evaluable))
    updated = simulate_outcomes(evaluable, horizon_days, cost_bps, slippage_bps)

    # Merge outcomes back into main DataFrame
    outcome_cols = ["outcome", "outcome_date", "r_multiple", "hold_days",
                    "mfe_pct", "mae_pct", "max_price", "min_price", "pnl_pct"]
    for col in outcome_cols:
        if col in updated.columns:
            df.loc[updated.index, col] = updated[col]

    # Write back
    table = pa.Table.from_pandas(df, schema=SIGNAL_SCHEMA)
    pq.write_table(table, SIGNAL_STORE_PATH)

    n = updated["outcome"].notna().sum()
    log.info("Updated %d outcomes: %s", n, updated["outcome"].value_counts().to_dict())
    return int(n)