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

    # Intraday data (tvDatafeed)
    ("intraday_rsi", pa.float64()),
    ("intraday_vwap", pa.float64()),
    ("intraday_volatility_pct", pa.float64()),
    ("intraday_data_quality", pa.string()),

    # Intraday analysis (5-min)
    ("intraday_score", pa.float64()),
    ("intraday_volume_skew", pa.float64()),
    ("intraday_momentum", pa.float64()),
    ("intraday_vwap_score", pa.float64()),
    ("intraday_ad_signal", pa.string()),
    ("intraday_ad_score", pa.float64()),
    ("intraday_adjustments", pa.string()),

    # Personalized holding period
    ("hold_personalized", pa.bool_()),
    ("hold_confidence", pa.float64()),
    ("hold_days_min", pa.int32()),
    ("hold_days_max", pa.int32()),
    ("hold_days_target", pa.int32()),

    # Outcome tracking
    ("outcome", pa.string()),
    ("outcome_date", pa.date32()),
    ("entry_date", pa.date32()),
    ("entry_price_actual", pa.float64()),
    ("r_multiple", pa.float64()),
    ("hold_days", pa.int32()),
    ("mfe_pct", pa.float64()),
    ("mae_pct", pa.float64()),
    ("max_price", pa.float64()),
    ("min_price", pa.float64()),
    ("pnl_pct", pa.float64()),
    ("exit_price", pa.float64()),
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

    # Ensure existing has all schema columns (for migration)
    for field in SIGNAL_SCHEMA:
        col = field.name
        if col not in existing.columns:
            existing[col] = None

    # Reorder existing columns to match schema
    existing = existing[[f.name for f in SIGNAL_SCHEMA]]

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
    """Simulate outcomes for closed trades using historical price data.

    Downloads one OHLCV window per ticker (not per trade) so a large pending
    backlog stays within CI time limits.
    """
    if trades.empty:
        return trades

    import yfinance as yf

    trades = trades.copy()
    cost_pct = (cost_bps + slippage_bps) / 10000.0
    today = date.today()

    def _flatten(df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    # One download per ticker spanning the union of all its trade windows.
    price_cache = {}
    for ticker in trades["ticker"].unique():
        starts, ends = [], []
        for _, r in trades[trades["ticker"] == ticker].iterrows():
            rd = r["run_date"]
            rd = rd.date() if hasattr(rd, "date") else rd
            starts.append(rd + timedelta(days=1))
            ends.append(rd + timedelta(days=horizon_days * 2 + 10))
        if not starts:
            continue
        yf_ticker = ticker if ticker.endswith(".CA") else f"{ticker}.CA"
        try:
            raw = yf.download(yf_ticker, start=str(min(starts)), end=str(max(ends)),
                              progress=False, auto_adjust=True)
            price_cache[ticker] = _flatten(raw) if raw is not None else pd.DataFrame()
        except Exception as e:
            log.warning("Price fetch failed for %s: %s", ticker, e)
            price_cache[ticker] = None  # distinct from empty: fetch error

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
            # Entry happens next trading day (not same day as recommendation)
            start = run_dt + timedelta(days=1)
            end = run_dt + timedelta(days=horizon_days * 2 + 10)

            cached = price_cache.get(ticker)
            if cached is None:
                # Fetch error for this ticker -> retryable pending
                trades.at[idx, "outcome"] = "pending"
                continue

            # Slice the cached window down to this trade's own window
            if cached.empty:
                df = cached
            else:
                df = cached[(cached.index.date >= start) & (cached.index.date <= end)]

            if df.empty:
                # Transient empty result (holiday / network blip). Only lock in
                # no_data once the horizon has fully elapsed; otherwise keep it
                # pending so a later run can retry.
                if (today - run_dt).days >= horizon_days:
                    trades.at[idx, "outcome"] = "no_data"
                else:
                    trades.at[idx, "outcome"] = "pending"
                continue

            # Limit-entry model: the recommended entry is a resting buy limit
            # below market and only fills when price trades at or through it.
            # (Taking next-day open unconditionally made gapped-up opens "fill"
            # above the TP levels and produced impossible negative-PnL "TP wins".)
            fill_i = None
            for i in range(len(df)):
                if float(df["Low"].iloc[i]) <= entry:
                    fill_i = i
                    break

            if fill_i is None:
                # Limit never filled inside the horizon window.
                if (today - run_dt).days >= horizon_days:
                    trades.at[idx, "outcome"] = "no_entry"
                    trades.at[idx, "hold_days"] = 0
                    trades.at[idx, "pnl_pct"] = 0.0
                    trades.at[idx, "r_multiple"] = 0.0
                else:
                    trades.at[idx, "outcome"] = "pending"
                continue

            fill_bar = df.iloc[fill_i]
            if "Open" in df.columns and not pd.isna(fill_bar["Open"]):
                open_px = float(fill_bar["Open"])
                # Gap-down through the limit fills at the market open
                actual_entry = min(open_px, entry)
            else:
                actual_entry = entry
            entry_date = df.index[fill_i].date()

            # Walk at most horizon_days trading bars from the fill
            trade_df = df.iloc[fill_i:fill_i + horizon_days]

            outcome = "expired"
            exit_i = len(trade_df) - 1
            exit_price = float(trade_df["Close"].iloc[-1])
            max_p = float(trade_df["High"].max())
            min_p = float(trade_df["Low"].min())

            # First-touch walk: SL and TPs are checked bar-by-bar in time
            # order. Whichever level is touched first determines the outcome.
            # If both are touched inside the same bar, SL wins (worst case).
            for i in range(len(trade_df)):
                low = float(trade_df["Low"].iloc[i])
                high = float(trade_df["High"].iloc[i])

                sl_hit = sl > 0 and low <= sl
                tp_hit_idx = None
                for j in range(len(tps) - 1, -1, -1):  # highest TP first
                    tp = tps[j]
                    if tp and high >= tp:
                        tp_hit_idx = j
                        break

                if sl_hit:
                    outcome = "stop_loss"
                    # A gap straight through the stop fills at the market
                    # (the open), not at the stop price itself
                    stop_open = trade_df["Open"].iloc[i]
                    exit_price = min(sl, float(stop_open)) if not pd.isna(stop_open) else sl
                    exit_i = i
                    break
                if tp_hit_idx is not None:
                    outcome = f"tp{tp_hit_idx+1}_hit"
                    exit_price = tps[tp_hit_idx]
                    exit_i = i
                    break

            # No touch at all: still open if the horizon hasn't elapsed yet.
            horizon_elapsed = (len(trade_df) >= horizon_days or
                               (today - entry_date).days >= horizon_days)
            if outcome == "expired" and not horizon_elapsed:
                outcome = "pending"

            # outcome_date is the actual touch (or time-exit) bar, not the
            # last bar of the download window; hold is bars held from fill.
            outcome_date = trade_df.index[exit_i].date()
            hold = exit_i

            # Use actual entry price for PnL calculations
            risk = abs(actual_entry - sl) if sl > 0 else actual_entry * 0.02
            if risk <= 0:
                risk = actual_entry * 0.02
            r_mult = (exit_price - actual_entry) / risk
            pnl = (exit_price - actual_entry) / actual_entry * 100 if actual_entry else 0
            pnl -= cost_pct * 100  # round-trip costs + slippage

            trades.at[idx, "outcome"] = outcome
            trades.at[idx, "outcome_date"] = outcome_date
            trades.at[idx, "entry_date"] = entry_date
            trades.at[idx, "entry_price_actual"] = round(actual_entry, 3)
            trades.at[idx, "exit_price"] = round(exit_price, 3)
            trades.at[idx, "r_multiple"] = round(r_mult, 3)
            trades.at[idx, "hold_days"] = hold
            trades.at[idx, "mfe_pct"] = round((max_p - actual_entry) / actual_entry * 100, 2) if actual_entry else 0
            trades.at[idx, "mae_pct"] = round((min_p - actual_entry) / actual_entry * 100, 2) if actual_entry else 0
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

    # Backfill: rows closed before the entry/exit fields existed carry no
    # entry_date/entry_price_actual, which makes the report fall back to the
    # planned entry while PnL is measured against the real fill — visibly
    # inconsistent rows (negative-PnL "TP wins", exit dates all clamped to
    # one day). Re-evaluate recent closed rows once; afterwards entry_date
    # is set and the row drops out of the mask (self-limiting).
    backfill_mask = (
        df["outcome"].isin(["tp1_hit", "tp2_hit", "tp3_hit", "stop_loss", "expired"])
        & df["entry_date"].isna()
        & (pd.to_datetime(df["run_date"]).dt.date >= today - timedelta(days=45))
    )

    # Evaluate any pending signal that has at least one post-entry bar.
    # We no longer wait the full horizon: SL/TP hits are detected as soon as
    # they occur, so a trade that already hit its stop is closed immediately
    # instead of staying "pending" for horizon_days. Trades with no touch yet
    # are returned as "pending" by simulate_outcomes (still open) and will be
    # re-evaluated on a later run.
    eval_cutoff = today - timedelta(days=2)
    evaluable = df[(pending_mask | backfill_mask) &
                   (pd.to_datetime(df["run_date"]).dt.date <= eval_cutoff)].copy()

    if evaluable.empty:
        log.info("No pending signals with post-entry data to evaluate yet")
        return 0

    log.info("Evaluating outcomes for %d signals (%d backfill)...",
             len(evaluable), int(backfill_mask.sum()))
    updated = simulate_outcomes(evaluable, horizon_days, cost_bps, slippage_bps)

    # Merge outcomes back into main DataFrame
    outcome_cols = ["outcome", "outcome_date", "entry_date", "entry_price_actual",
                    "r_multiple", "hold_days",
                    "mfe_pct", "mae_pct", "max_price", "min_price", "pnl_pct",
                    "exit_price"]
    for col in outcome_cols:
        if col in updated.columns:
            df.loc[updated.index, col] = updated[col]

    # Write back
    table = pa.Table.from_pandas(df, schema=SIGNAL_SCHEMA)
    pq.write_table(table, SIGNAL_STORE_PATH)

    n = updated["outcome"].notna().sum()
    log.info("Updated %d outcomes: %s", n, updated["outcome"].value_counts().to_dict())
    return int(n)
