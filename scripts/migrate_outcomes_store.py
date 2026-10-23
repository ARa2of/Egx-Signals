"""
One-off store migration (idempotent, safe to re-run):
  1. Remove junk TEST rows (not history — synthetic test artifacts).
  2. Tag rows evaluated by the old next-open simulation as sim_model=legacy
     so report KPIs can exclude them (two models in one series is noise).
     Cut-over date: 2026-10-10 (limit-entry first-touch model commit).

A timestamped backup is written next to the store before any change.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import date

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.store.signal_store import SIGNAL_STORE_PATH, SIGNAL_SCHEMA

LEGACY_CUTOVER = date(2026, 10, 10)


def main():
    if not SIGNAL_STORE_PATH.exists():
        print("Store not found:", SIGNAL_STORE_PATH)
        return

    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    backup = SIGNAL_STORE_PATH.with_suffix(f".parquet.bak_{stamp}")
    backup.write_bytes(SIGNAL_STORE_PATH.read_bytes())
    print(f"Backup: {backup}")

    df = pq.read_table(SIGNAL_STORE_PATH).to_pandas()
    n0 = len(df)

    # Schema migration: ensure sim_model exists
    if "sim_model" not in df.columns:
        df["sim_model"] = None

    # Fix 4: drop junk TEST rows
    junk_mask = df["ticker"].astype(str).str.strip().str.upper() == "TEST"
    n_junk = int(junk_mask.sum())
    df = df[~junk_mask].copy()

    # Fix 3: tag legacy-model rows (self-limiting: already-tagged or recent rows untouched)
    rd = pd.to_datetime(df["run_date"]).dt.date
    untagged = df["sim_model"].isna() | (df["sim_model"] == "")
    legacy_mask = untagged & (rd < LEGACY_CUTOVER)
    df.loc[legacy_mask, "sim_model"] = "legacy"
    n_legacy = int(legacy_mask.sum())
    # Any remaining untagged rows post-cutover are current-model
    still = df["sim_model"].isna() | (df["sim_model"] == "")
    df.loc[still, "sim_model"] = "limit_entry"
    n_new = int(still.sum())

    df = df[[f.name for f in SIGNAL_SCHEMA]]
    table = pa.Table.from_pandas(df, schema=SIGNAL_SCHEMA)
    pq.write_table(table, SIGNAL_STORE_PATH)

    print(f"Rows: {n0} -> {len(df)} (removed {n_junk} TEST)")
    print(f"Tagged legacy: {n_legacy} | tagged limit_entry: {n_new}")
    print("sim_model counts:", df["sim_model"].value_counts().to_dict())


if __name__ == "__main__":
    main()
