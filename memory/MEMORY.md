# Long-term Memory

This file stores important information that should persist across sessions.

## Project Context

**EGX-Signals V2** (params.yaml version **v2.3**, 2026-10-07) — daily Buy/Watch/Avoid signal engine for the Egyptian Exchange. Root `D:\Stocks\EGX Project`; repo `https://github.com/ARa2of/Egx-Signals` (main; CI `daily-analysis.yml` deploys the HTML report to gh-pages; crons `0 14 * * *` safety + `0 16 * * *` main, UTC, with a git-log dedup step gating double runs). Pipeline `scripts\daily_run.py`; config `config\params.yaml`; universe `config\tickers.xlsx`; signal history `data\signals.parquet` (Git LFS).

### Architecture

- `src\signals\scoring.py` — base score + RSI regime (`resolve_rsi_regime`)
- `src\signals\technical.py` — indicators, trend regime, volume profile / support-resistance
- `src\signals\ml.py` — ML ensemble + conviction
- `src\trade\planner.py` — ATR entries/stops/targets, `compute_signal_score`, `compute_ml_conviction`
- `src\signals\chartscan.py` — YOLOv8 chart-pattern votes; `src\output\html_report.py` / `agent_json.py` — outputs
- `src\store\signal_store.py` — signal history store; trade-outcome updates (`update_outcomes`, `simulate_outcomes`)
- Consensus ladder, ChartScan gates, verification and git runbooks: `skills\egx-signals\SKILL.md`

### v2.3 engine decisions (landed 2026-10-07/08, supersede v2.2)

- Trade plan passes `trend_context` / `direction_accuracy` / `direction_proba` / `ensemble_weights` into ML conviction (fallback derived from regime).
- Breakout gate: Medium OR High forecast quantile > 1.02×resistance OR `ensemble_p_up >= 0.55`; dead `entry_adjustment` param removed.
- Take-profit targets sourced from params.yaml trade constants (tp1/2/3_pct = 1.0/2.0/6.0, min_rr = 3.0).
- Volume profile splits each bar's volume evenly across spanned bins (mirrors S/R histogram).
- Momentum filter compares against the most recent Buy within `thresholds.momentum_filter_lookback_days` (60); skips when none in window.
- `compute_signal_score` RSI band is regime-sourced (was hardcoded 35–65).
- Market-context gate: Bearish index (via `ticker_index_map`) downgrades a single-Buy upgrade to Watch; exempt: strong_technicals and 2+ Buy votes; toggle `market_context.use_index_sentiment`; Buys annotated `[<index> <sentiment>]`.
- `agent_json.py` meta version reads `params.yaml` via `load_params().get("version")` (was hardcoded).

### Trade-outcome & report fixes (2026-10-20/21, `signal_store.py` + `html_report.py` + `daily_run.py`)

- `update_outcomes` closure gate lowered from `run_date <= today-(horizon_days+5)` (~26d) to 2d so SL/TP hits close as soon as observed; its `outcome_cols` merge must include `entry_date` and `entry_price_actual` (store rows feed the report's entry/exit columns).
- `simulate_outcomes` is a first-touch bar-by-bar walk: TP/SL closes on first touch, SL wins only a same-bar tie, outcome stays `pending` until a level is touched (no premature `expired`); transient empty fetch → retryable `no_data`; price downloads batched per ticker to stay within CI timeout.
- Trade Outcomes table in `html_report.py` reads the store outcome first (`tp1_hit`/`tp2_hit`/`tp3_hit`/`stop_loss`/`expired` → TP1/TP2/TP3/SL/TIME badges, exit/entry from store); pending renders mark-to-market OPEN. The old close-only re-simulation (missed intraday first-touches; dropped pending rows with no later store rows) was deleted — do not reintroduce.
- `daily_run.py` calls `update_outcomes()` (try/except) just before `generate_html_report` so the report is fresh in-run; the workflow's separate `update_outcomes.py` step no longer leaves the report a day stale. Regression test: `scripts\test_report_outcomes.py`.
- Report monthly Average row: average the benchmark over the same months as the strategy (mixed month sets produced a negative average); `load_store()` returns `run_date` as object-dtype dates — compare date-to-date in report code.

### ChartScan infrastructure

- Real 52MB YOLOv8 checkpoint `weights\custom_yolov8.pt` (Git LFS); CI loads it with `lfs: true` + a `YOLO()` load check that fails loudly.
- 0%-confidence days for most tickers are expected: the model fires only on trained patterns and detection rate varies with market data. Tuning levers: `chartscan.confidence_threshold` (0.3) in params.yaml and `MIN_BOX_CONF` (0.40) in `src\signals\chartscan.py`.
- Report renders "0%" (not "N/A") when the model ran without a match (falsy-0 display bug fixed in `html_report.py`).

## Important Notes

- Engine is ticker-agnostic: no ticker-specific logic anywhere (ETEL appears only in test fixtures).
- Keep the dual crons (14:00 UTC safety + 16:00 UTC main) and the git-log dedup step in `daily-analysis.yml` — they mitigate GitHub Actions' best-effort cron skips (runbook details live in the egx-signals skill).
- No separate local VCS needed: git (root `D:\Stocks\EGX Project` → `ARa2of/Egx-Signals`) covers code/config history; this file + `memory\history.jsonl` cover session context. Intentionally untracked in git: `skills\`, `prompts\`, `cron\`, `check_today.py` / `check_consensus.py` / `check_zeot.py`, `AGENTS.md`, `HEARTBEAT.md`, `.nanobot\`, and the duplicate `egx-signals\` / `egx-portfolio-manager\` folders.
