# EGX Signals V2 — Session Handoff

**For:** next agent taking over this workspace  
**Repo:** https://github.com/ARa2of/Egx-Signals  
**Local root:** `D:\Stocks\EGX Project`  
**Branch:** `main`  
**Config version:** `v2.2` (`config/params.yaml`)  
**Last major work:** uptrend scoring fixes, ML conviction v3, ChartScan weights, consensus consistency

---

## 1. What this project is

Daily automated **Egyptian Exchange (EGX)** stock signal engine.

- Runs via **GitHub Actions** (`.github/workflows/daily-analysis.yml`) at ~16:00 UTC Sun–Thu  
- Primary data: **TradingView** batch TA; fallback **yfinance** (`.CA` tickers)  
- Outputs: Excel, `output/latest_signals.csv`, `docs/index.html` (GitHub Pages), `output/agent_data.json`, Parquet signal store  
- Final label is **consensus**, not a single score: Base technical + ML + ChartScan, then override filters

User’s core complaint historically: **strong uptrends (e.g. ETEL) were labeled Avoid/Watch** because scoring, ML, and consensus were mean-reversion biased. That was addressed in v2.2 — do not regress it.

---

## 2. Critical paths

### Pipeline entry
| Path | Role |
|------|------|
| `scripts/daily_run.py` | **Main pipeline.** TV + yf, enrich, ML, intraday, base score, consensus, overrides, exports |
| `config/params.yaml` | Weights, thresholds, RSI zones, ML params, trade plan (**v2.2**) |
| `config/enhancement.yaml` | Weekly/monthly tuning config |
| `config/tickers.xlsx` | Input list (`Selected_Stocks` sheet) |
| `config/EGX_Stock_Groups.xlsx` | Sector peers + INDEX membership for ML |

### Signals / ML
| Path | Role |
|------|------|
| `src/signals/scoring.py` | 9-category base score; `resolve_rsi_regime()`; trend-aware VWAP/VP |
| `src/signals/technical.py` | Indicators + `enrich()`; **`detect_trend_regime()`** (bullish bias in v2.2) |
| `src/signals/ml.py` | XGBoost+LightGBM quantile ensemble, direction classifiers, **conviction v3**, model cache |
| `src/signals/chartscan.py` | YOLOv8 candlestick detector; vote sanitization in caller |

### Data
| Path | Role |
|------|------|
| `src/data/loader.py` | TV batch scan, yfinance download, FX, volume, ADL/MFI, fundamentals |
| `src/data/intraday_analysis.py` | 5-min analysis; **bull-trend aware** volume skew/momentum/VWAP |
| `src/data/intraday.py` | tvDatafeed hourly summary |
| `data/cache/` | yfinance OHLCV cache |
| `data/model_cache/` | Per-ticker ML pickle cache (14-day TTL) |
| `data/intraday_cache/` | 5-min parquet cache (4h) |
| `data/signals.parquet` | Append-only history (**Git LFS**) |

### Trade / output
| Path | Role |
|------|------|
| `src/trade/planner.py` | ATR entries/stops/targets; fallback = **CHASE — NO CLEAN PULLBACK** (not direction AVOID) |
| `src/trade/holding_engine.py` | Personalized hold windows |
| `src/output/excel.py`, `html_report.py`, `agent_json.py`, `alerts.py` | Reports + Telegram |
| `src/store/signal_store.py` | Parquet append + `latest_signals.csv` |
| `src/config.py` | Loads YAML (module-level cache — **reload after params edits**) |

### ChartScan weights
| Path | Role |
|------|------|
| `weights/custom_yolov8.pt` | **Git LFS** (~52MB). Classes: `{0: Buy, 1: Sell}`. Source: [Omar-Karimov/ChartScanAI](https://github.com/Omar-Karimov/ChartScanAI) |
| `.gitattributes` | LFS track for weights + parquet |
| `.gitignore` | `weights/*.pt` ignored **except** `!weights/custom_yolov8.pt` |

### Tests / diagnostics (run these after any scoring/consensus change)
| Path | Role |
|------|------|
| `scripts/verify_etel_fix.py` | Live ETEL replay → expect **Buy** |
| `scripts/test_scoring_fixes.py` | Unit checks: regime map, VWAP/VP, weights sum=100 |
| `scripts/test_ml_v3.py` | Conviction v3, direction classifiers, ChartScan load |
| `scripts/test_consensus_cases.py` | ETEL/EFIH/MBSC/ARCC consensus expectations |
| `check_today.py`, `check_consensus.py` | Quick store diagnostics |

### Docs / HTML
| Path | Role |
|------|------|
| `README.md` | Architecture (updated v2.2 weights) |
| `docs/index.html` | GitHub Pages report (regenerated each run) |
| `_sources/` | Research briefs (untracked) |

### Untracked / side projects (not part of daily pipeline)
`egx-signals/`, `egx-portfolio-manager/`, `egx_distribution_analysis/`, `check_*.py` — separate copies/experiments. **Prefer editing root `src/` + `scripts/`, not `egx-signals/`.**

---

## 3. How a signal is actually decided

```
yfinance bars ──(merge TV session)──► enrich() ──► regime, EMAs, VWAP, VP, patterns
                                                      │
                         Base score (weights sum 100) │
                         + preliminary → intraday      │
                         (Buy/Watch only, bull-trend)  │
                                                      ▼
                         ML ensemble forecast + conviction v3
                                                      │
                         ChartScan YOLO (sanitized votes)
                                                      ▼
                         CONSUS (see below)
                                                      ▼
                         Overrides: RSI hard-cap (ADX-aware),
                         momentum filter, entry quality (not chase)
```

### Consensus (in `scripts/daily_run.py`)
- **ML tiers:** Buy ≥ 35, Watch ≥ 20, else Avoid (same numbers for `ML Signal` display)  
- **ChartScan:** abstain if model missing / conf=0; Buy needs conf≥0.40; Avoid needs Sell conf≥0.50; Sell + bullish candle → Neutral  
- **Strong technicals path:** Base=Buy + golden + diamond + ADX≥30 + MACD bullish + no death cross → **Buy even if ML Avoid**, unless ChartScan Avoid  
- **Votes:** 2 active Buys → Buy; 1 Buy + 0 Avoid → Buy; etc.  
- **Entry override:** chase/wait/fallback only annotates basis; **POOR SETUP** can downgrade Watch if not strong_technicals  

### Scoring pitfalls that caused the original bug (v2.1 → v2.2)
| Old behavior | v2.2 behavior |
|--------------|---------------|
| Regime `"trending"` fell back to ranging RSI bands | `resolve_rsi_regime()` → `trending_up` |
| Regime detector required DI+ > DI− strictly | Bullish EMA + price>SMA50 + ADX≥25 → trending_up |
| Intraday weight 30, volume-at-highs = distribution | Weight **15**; bull_trend dampens skew |
| VWAP >5% above = mean-reversion penalty | Up to 8% OK in strong trend |
| ChartScan Neutral always voted Watch | Abstains when no real signal |
| Entry “AVOID (no clean setup)” vetoed Buy | Chase/wait ≠ direction Avoid |
| ML cached path used `(1+r)` on log returns | **`exp(r)`** |

---

## 4. ML details (what “enhanced” means)

- **Models:** 3 quantiles (0.10 / 0.50 / 0.90) × XGBoost + LightGBM; ensemble 50/50  
- **Horizon:** 10 sessions (`ml.horizon_days`)  
- **Features:** lags, SMA/EMA distance, RSI, BB, ATR, MACD, ADX, ROC, stoch, hist vol, GK vol, OBV slopes, vol ratio, VWAP dist, peer returns, sector breadth, rel strength, calendar + **v2.2 trend features** (`above_ema50`, `days_above_ema50`, `up_days_10`, `ret_sign_consistency`, …)  
- **Direction:** proper `XGBClassifier` / `LGBMClassifier`, P(up) fed into conviction  
- **Cache:** `data/model_cache/{TICKER}_models.pkl`; feature-schema mismatch forces retrain; max age 14 days  
- **Conviction v3:** upside, cone, asymmetry, downside protection, P(up), direction accuracy, ensemble agreement, **bull_trend adjustments**; MAE penalty only if `|upside|≥1.5%`  
- **Known weakness:** models still mean-revert after big runs (features look “overbought”). Consensus strong_technicals path is the safety net — keep it.

---

## 5. Runbook

```bash
# Full local day (needs network; slow)
python scripts/daily_run.py config/tickers.xlsx -o output

# After editing params.yaml — config module caches YAML:
python -c "from src.config import reload_configs; reload_configs()"

# Verification suite (run after scoring/consensus/ML edits)
python scripts/test_scoring_fixes.py
python scripts/test_ml_v3.py
python scripts/test_consensus_cases.py
python scripts/verify_etel_fix.py   # ETEL must end Buy

# Store diagnostics
python check_today.py
python check_consensus.py
```

**Push:** user typically wants changes on `main` then **Actions → Daily EGX Analysis → Run workflow**.  
`gh` CLI may be unavailable locally; use the GitHub UI.

**Git notes:**
- LFS required for `weights/custom_yolov8.pt` and `data/signals.parquet`  
- If `git pull` blocked by untracked `output/agent_data.json`, move it aside then pull  
- Do not force-delete signal store history  

---

## 6. Current expected behavior (post v2.2 consistency fix)

| Ticker | Score | ML display | ChartScan | Final |
|--------|------:|------------|-----------|-------|
| ETEL | ~64–66 Buy band | Watch/Avoid allowed | Buy (low conf OK) | **Buy** via strong_technicals; basis notes chase entry |
| EFIH | high | **Watch** at conf 30 (not Avoid) | Neutral | **Buy** if 1 Buy + 0 Avoid |
| MBSC | high | Buy | Sell + bullish candle → **Neutral** | **Buy** if Base+ML |
| ARCC | ~80 | Buy | Neutral | Buy unless **momentum filter** (price drop since first Buy) |

If ETEL returns to Watch, check in order:
1. `regime` still `trending_up`?  
2. Base score ≥ 60?  
3. `strong_technicals` true (golden+diamond+ADX≥30+MACD)?  
4. Entry override still vetoing? (should not for chase)  
5. Momentum filter / RSI hard-cap firing?

---

## 7. Open issues / next agent backlog

1. **ML mean-reversion** on extended names — longer horizon or trend-conditional targets; more weight on direction model  
2. **ChartScan quality** — YOLO still noisy; may need chart style / crop / multi-candle context; consider downweighting in consensus further  
3. **Trade planner R/R** on chase entries often &lt; min_rr — entry quality flag exists; UI could separate “direction Buy” vs “entry quality”  
4. **ARCC-style momentum filter** — first Buy close can be very old; consider rolling window not all-time first Buy  
5. **README / docs drift** — keep in sync with params.yaml  
6. **Side folders** (`egx-signals/` etc.) — decide keep/archive so agents don’t edit the wrong tree  
7. **Workflow** — ML model cache in Actions helps; first run after feature changes retrains everything (slow but OK)  
8. **agent_data.json** — was untracked/local; workflow commits it on CI  

---

## 8. Conventions

- **Do not** assume TA-Lib or heavy new deps without adding to `requirements.txt`  
- **Do not** change consensus thresholds casually — they were tuned after user complaint about zero Buys  
- **Do** run `test_consensus_cases.py` + `verify_etel_fix.py` before push  
- **Do** bump `config/params.yaml` `version` + `updated` when changing weights/thresholds  
- Prefer root `src/` + `scripts/` over `egx-signals/` duplicate  

---

## 9. One-paragraph context for a fresh agent

EGX Signals is a daily Egypt-market Buy/Watch/Avoid engine. In Oct 2026 the user correctly identified that **uptrending names like ETEL were mislabeled Avoid** because (a) regime labels didn’t match RSI config keys, (b) scoring punished extension/VWAP/intraday distribution the way mean-reversion tools do, (c) ML conviction collapsed on overbought features, (d) consensus let ML/ChartScan/planner entry-veto block technical Buys, and (e) ChartScan YOLO had no weights and later produced contradictory Sell signals. **v2.2 fixed regime mapping, weight balance, trend-aware VWAP/VP/intraday, strong-technical consensus path, ML exp() cache bug + conviction v3 + direction classifiers, committed YOLO weights via LFS, and stopped “no clean pullback” from vetoing direction.** ETEL-class setups should now show **Buy** even if ML is cautious, with chase risk noted on the trade plan. Keep that invariant when changing anything in scoring, consensus, ML, or the planner.
