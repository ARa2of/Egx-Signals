"""
Update trade outcomes for pending signals.
Fetches historical price data and evaluates whether TP/SL levels were hit.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from datetime import date
from src.config import load_enhancement_config
from src.store.signal_store import update_outcomes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    cfg = load_enhancement_config()
    outcomes_cfg = cfg.get("outcomes", {})
    horizon = outcomes_cfg.get("horizon_days", 21)
    cost = outcomes_cfg.get("cost_bps", 10)
    slippage = outcomes_cfg.get("slippage_bps", 5)

    log.info("Updating outcomes (horizon=%d days, cost=%d bps, slippage=%d bps)",
             horizon, cost, slippage)
    n = update_outcomes(horizon_days=horizon, cost_bps=cost, slippage_bps=slippage)
    log.info("Done. Updated %d outcomes.", n)


if __name__ == "__main__":
    main()
