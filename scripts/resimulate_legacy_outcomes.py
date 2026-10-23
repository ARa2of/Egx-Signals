"""
Re-evaluate legacy-model outcome rows under the current limit-entry
first-touch simulation, then tag them sim_model=limit_entry.

September rows were evaluated with the old next-open fill model; mixing
two models in one KPI series is noise. This rewrites their outcomes with
the same model as update_outcomes (horizon/cost/slippage from config).

Idempotent: only touches sim_model == 'legacy' rows. Backup written first.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import load_enhancement_config
from src.store.signal_store import SIGNAL_STORE_PATH, SIGNAL_SCHEMA, simulate_outcomes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    cfg = load_enhancement_config()
    oc = cfg.get("outcomes", {})
    horizon = oc.get("horizon_days", 21)
    cost = oc.get("cost_bps", 10)
    slippage = oc.get("slippage_bps", 5)

    if not SIGNAL_STORE_PATH.exists():
        log.error("Store missing: %s", SIGNAL_STORE_PATH)
        return

    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    backup = SIGNAL_STORE_PATH.with_suffix(f".parquet.bak_{stamp}")
    backup.write_bytes(SIGNAL_STORE_PATH.read_bytes())
    log.info("Backup: %s", backup)

    df = pq.read_table(SIGNAL_STORE_PATH).to_pandas()
    if "sim_model" not in df.columns:
        df["sim_model"] = None

    legacy = df["sim_model"] == "legacy"
    # --all: re-walk every row with a trade plan (used when the sim model
    # itself is refined, e.g. the knife-day rule) so the store stays one model.
    all_rows = "--all" in sys.argv
    reeval = (df["entry_price"].notna() & df["stop_loss"].notna()) & (legacy | all_rows)
    n_skip = int((legacy & ~reeval).sum())
    todo = df[reeval].copy()

    log.info("Re-simulating %d legacy rows (%d skipped, no trade plan) under limit-entry model",
             len(todo), n_skip)
    if todo.empty:
        log.info("Nothing to do")
        return

    updated = simulate_outcomes(todo, horizon_days=horizon, cost_bps=cost, slippage_bps=slippage)

    outcome_cols = ["outcome", "outcome_date", "entry_date", "entry_price_actual",
                    "r_multiple", "hold_days", "mfe_pct", "mae_pct", "max_price",
                    "min_price", "pnl_pct", "exit_price"]
    for col in outcome_cols:
        if col in updated.columns:
            df.loc[updated.index, col] = updated[col]
    df.loc[updated.index, "sim_model"] = "limit_entry"

    table = pa.Table.from_pandas(df, schema=SIGNAL_SCHEMA)
    pq.write_table(table, SIGNAL_STORE_PATH)

    log.info("Outcome counts (re-simulated): %s", updated["outcome"].value_counts(dropna=False).to_dict())
    log.info("sim_model now: %s", df["sim_model"].value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
