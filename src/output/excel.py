"""
Excel Output Generator for EGX Signals
Creates detailed analysis workbook with multiple sheets
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import logging

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Column Definitions
# ─────────────────────────────────────────────────────────────
MAIN_COLUMNS = [
    "Analysis Run Date", "Selected Stock", "Index Membership", "Data As Of",
    "Current EGP Price", "Current USD Price",
    "Historical Min USD Price", "Historical Max USD Price",
    "Undervalued (Yes/No)", "Implied Fair Value (EGP)", "Fair Value Method",
    "P/E Ratio (TTM)", "EPS (TTM)",
    "3-Month Avg Volume", "Last Day Volume", "Volume Multiplier (vs 3M)",
    "Est. Buy Volume (2-Month Avg)", "Est. Buy Volume (Last Day)",
    "Buy Volume Multiplier (vs 2-Month)",
    "Support", "Resistance",
    "50 SMA", "200 SMA", "Golden Cross (Yes/No)", "Death Cross (Yes/No)",
    "20 EMA", "50 EMA", "200 EMA", "EMA Bullish (50>200) (Yes/No)",
    "Diamond Cross (20>50) (Yes/No)",
    "MACD", "MACD Signal", "MACD Bullish (Yes/No)",
    "RSI (%)", "VWMA",
    "ADX", "ADX +DI", "ADX -DI", "MFI", "BB Squeeze",
    "ADL", "ADL Trend (20d)",
    "TA Data As Of",
    "Optimal Entry Price", "Stop Loss", "Stop Loss Basis",
    "Take Profit 1", "Take Profit 2", "Take Profit 3", "Take Profit Basis",
    "TP1 Risk/Reward", "TP2 Risk/Reward", "TP3 Risk/Reward",
    "TP1 Reward %", "TP2 Reward %", "TP3 Reward %",
    "Recommendation", "Recommendation Basis", "Score",
    "Score - Trend", "Score - MACD", "Score - RSI", "Score - Volume", "Score - ADI", "Score - Support",
    "Score - VWAP", "Score - Volume Profile",
    "VWAP", "Dist VWAP %", "Volume Profile POC", "Volume Profile VA High", "Volume Profile VA Low", "Above POC",
    "ChartScanAI Signal", "ChartScanAI Recommendation", "ChartScanAI Confidence",
    "ChartScanAI Buy Patterns", "ChartScanAI Sell Patterns",
    "ML Signal", "ML Confidence", "ML Medium Price", "ML Conviction",
]

HISTORY_COLUMNS = [
    "Analysis Run Date", "Selected Stock", "Index Membership", "Current EGP Price", "Recommendation",
    "Score", "Score - Trend", "Score - MACD", "Score - RSI", "Score - Volume", "Score - ADI", "Score - Support",
    "Score - VWAP", "Score - Volume Profile",
    "Undervalued (Yes/No)", "Implied Fair Value (EGP)", "Fair Value Method",
    "Golden Cross (Yes/No)", "Death Cross (Yes/No)",
    "Diamond Cross (20>50) (Yes/No)", "RSI (%)",
    "ADX", "ADX +DI", "ADX -DI", "MFI", "BB Squeeze", "ADL", "ADL Trend (20d)",
    "ChartScanAI Signal", "ChartScanAI Recommendation", "ChartScanAI Confidence",
    "Volume Multiplier (vs 3M)", "Buy Volume Multiplier (vs 2-Month)",
    "Support", "Resistance",
    "P/E Ratio (TTM)", "EPS (TTM)",
    "MACD Bullish (Yes/No)",
    "Optimal Entry Price", "Stop Loss",
    "Take Profit 1", "Take Profit 2", "Take Profit 3",
]

INDEX_COLUMNS = [
    "Index", "TradingView Symbol", "Status", "Close", "Change (%)",
    "RSI (%)", "50 SMA", "200 SMA", "20 EMA", "50 EMA", "200 EMA",
    "MACD", "MACD Signal", "MACD Bullish (Yes/No)", "Data Fetched",
]

# ─────────────────────────────────────────────────────────────
# Export Functions
# ─────────────────────────────────────────────────────────────
def export_analysis(rows: List[Dict], index_rows: List[Dict], output_path: str) -> None:
    """Export main analysis and index data to Excel workbook."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame(rows)
    index_df = pd.DataFrame(index_rows)

    # Ensure all columns exist
    for col in MAIN_COLUMNS:
        if col not in out_df.columns:
            out_df[col] = None
    for col in INDEX_COLUMNS:
        if col not in index_df.columns:
            index_df[col] = None

    # Reorder columns
    out_df = out_df[MAIN_COLUMNS]
    index_df = index_df[INDEX_COLUMNS]

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        out_df.to_excel(writer, sheet_name="Stock_Analysis", index=False)
        index_df.to_excel(writer, sheet_name="Indices", index=False)

        # Auto-fit columns
        for sheet_name in ["Stock_Analysis", "Indices"]:
            ws = writer.sheets[sheet_name]
            for column in ws.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                ws.column_dimensions[column_letter].width = adjusted_width

    log.info("Analysis exported to %s (%d stocks, %d indices)", output_path, len(out_df), len(index_df))

def append_daily_history(rows: List[Dict], output_path: str, retention_days: int = 400) -> None:
    """Append today's signals to running history CSV."""
    from src.output.excel import HISTORY_COLUMNS
    from pathlib import Path

    # The signal store (Parquet) IS the history - just export latest to CSV for convenience
    out_df = pd.DataFrame(rows)
    cols = [c for c in HISTORY_COLUMNS if c in out_df.columns]
    today_snapshot = out_df[cols].copy()

    history_path = Path(__file__).parent.parent.parent / "data" / "history.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    if history_path.exists():
        try:
            existing = pd.read_csv(history_path)
        except Exception as e:
            log.warning("Could not read existing history: %s", e)
            existing = pd.DataFrame(columns=cols)
        combined = pd.concat([existing, today_snapshot], ignore_index=True)
        combined.drop_duplicates(subset=["Analysis Run Date", "Selected Stock"], keep="last", inplace=True)
    else:
        combined = today_snapshot

    # Trim old data
    if "Analysis Run Date" in combined.columns:
        # Handle both string and date formats
        combined["Analysis Run Date"] = pd.to_datetime(combined["Analysis Run Date"]).dt.date
        cutoff = datetime.now().date() - timedelta(days=retention_days)
        combined = combined[combined["Analysis Run Date"] >= cutoff]

    combined.to_csv(history_path, index=False)
    log.info("Daily history updated: %s (%d rows)", history_path, len(combined))

def export_latest_csv(rows: List[Dict], output_path: str = "output/latest_signals.csv") -> None:
    """Export latest signals to CSV for dashboard."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    log.info("Latest signals exported to %s (%d rows)", output_path, len(df))

# Need to import timedelta
from datetime import timedelta