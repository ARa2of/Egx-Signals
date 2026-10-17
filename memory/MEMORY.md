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

### ChartScan infrastructure

- Real 52MB YOLOv8 checkpoint `weights\custom_yolov8.pt` (Git LFS); CI loads it with `lfs: true` + a `YOLO()` load check that fails loudly.
- 0%-confidence days for most tickers are expected: the model fires only on trained patterns and detection rate varies with market data. Tuning levers: `chartscan.confidence_threshold` (0.3) in params.yaml and `MIN_BOX_CONF` (0.40) in `src\signals\chartscan.py`.
- Report renders "0%" (not "N/A") when the model ran without a match (falsy-0 display bug fixed in `html_report.py`).

## Important Notes

- Engine is ticker-agnostic: no ticker-specific logic anywhere (ETEL appears only in test fixtures).
- `verify_etel_fix.py` live-replay FAIL (ETEL → Avoid, score 57.84) is pre-existing data drift (replay has no intraday data + a partial last bar), identical on clean HEAD. The old "ETEL must end Buy" gate was **waived by the user** — not a blocker.
- `scripts\test_consensus_cases.py` contains a local mirror of the consensus function — update the mirror whenever consensus rules change.
- Git commit-graph rename/write warnings on this machine are harmless (Windows file locking); pushes succeed regardless.
- No separate local VCS needed: git (root `D:\Stocks\EGX Project` → `ARa2of/Egx-Signals`) covers code/config history; this file + `memory\history.jsonl` cover session context. Intentionally untracked in git: `skills\`, `prompts\`, `cron\`, `check_today.py` / `check_consensus.py` / `check_zeot.py`, `AGENTS.md`, `HEARTBEAT.md`, `.nanobot\`, and the duplicate `egx-signals\` / `egx-portfolio-manager\` folders.
- GitHub Actions cron is best-effort: scheduled runs can be skipped entirely under high load; typical queue delay is 3–4 h past the scheduled time. The dual crons (14:00 UTC safety + 16:00 UTC main) and the git-log dedup step (skips the pipeline when a "Daily signals" commit already exists that day) mitigate this — keep them in `daily-analysis.yml`.
