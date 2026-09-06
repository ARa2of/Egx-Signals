# EGX Signal Generator
Daily automated signal generation for Egyptian Exchange (EGX) stocks.

## Overview
- **Runs daily at 5:00 PM UK time** via GitHub Actions
- **TradingView batch scanning** (primary) + yfinance fallback
- **6-category weighted scoring** (Trend, MACD, RSI, Volume, ADI, Support)
- **ML price forecasting** (XGBoost quantile regression, 5-day horizon)
- **ATR-based trade planning** (entry zones, stops, targets, position sizing)
- **Signal history** stored in Parquet for backtesting and enhancement
- **Telegram alerts** for signal changes and high-conviction setups

## Architecture
```
.github/workflows/
├── daily-analysis.yml      # Daily 16:00 UTC (Mon-Fri)
├── weekly-enhance.yml      # Weekly quality monitoring
└── monthly-tune.yml        # Monthly grid search tuning

src/
├── data/loader.py          # TV batch + yfinance + fundamentals
├── signals/
│   ├── technical.py        # All technical indicators
│   ├── ml.py               # XGBoost quantile forecasting
│   └── scoring.py          # 6-category base score
├── trade/planner.py        # ATR-based entry/stop/targets
├── store/signal_store.py   # Parquet append-only store
├── enhancement/param_tuner.py  # Fast grid search
└── output/
    ├── excel.py            # Detailed Excel workbook
    └── alerts.py           # Telegram notifications

scripts/daily_run.py        # Main CLI entry point
config/
├── params.yaml             # All tunable parameters
├── enhancement.yaml        # Enhancement engine config
└── tickers.xlsx            # Input: Selected_Stocks + INDEX column
```

## Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/YOUR_USER/egx-signals.git
cd egx-signals
pip install -r requirements.txt
```

### 2. Configure Input
Edit `config/tickers.xlsx` with your stock list:
- Sheet: `Selected_Stocks`
- Column A: Ticker (e.g., `OBRI`, `TMGH`, `PHDC`)
- Column B (optional): `INDEX` - one of `EGX30`, `EGX70`, `EGX33`, `UNINDEX`

### 3. Run Locally
```bash
python scripts/daily_run.py config/tickers.xlsx
```
Output: `output/EGX_Signals_YYYYMMDD.xlsx` + `output/latest_signals.csv`

### 4. Deploy to GitHub Actions
1. Push to GitHub
2. Add secrets in repo Settings → Secrets → Actions:
   - `TELEGRAM_BOT_TOKEN` - BotFather token
   - `TELEGRAM_CHAT_ID` - Your chat ID (get from @userinfobot)
   - `EODHD_API_KEY` - Optional, for better EGX coverage
3. Enable GitHub Actions
4. Workflows run automatically on schedule

## Configuration

### Key Parameters (`config/params.yaml`)
```yaml
weights:
  trend: 30.0
  macd: 15.0
  rsi: 15.0
  volume: 15.0
  adi: 12.5
  support: 12.5

thresholds:
  buy: 60.0
  watch: 50.0

trade:
  stop_loss_atr: 1.5
  target_atr: 3.0
  min_rr: 1.2
```

### Enhancement (`config/enhancement.yaml`)
```yaml
param_tuning:
  enabled: true
  frequency: monthly
  param_grid:
    rsi_oversold: [28, 30, 32, 34, 36]
    volume_spike: [1.5, 1.8, 2.0, 2.2]
    stop_loss_atr: [1.2, 1.5, 1.8, 2.0]
```

## Outputs

### Excel Workbook (`output/EGX_Signals_YYYYMMDD.xlsx`)
- **Stock_Analysis**: Full detail per ticker (80+ columns)
- **Indices**: EGX30/70/33 snapshots

### Signal Store (`data/signals.parquet`)
Append-only Parquet with full history for:
- Backtesting parameter changes
- Score calibration tracking
- Regime performance analysis
- ML model retraining

### Telegram Alerts
- Daily summary with top buys
- Signal changes vs previous day
- High-conviction setups (score ≥ 80, R/R ≥ 3, ML conf ≥ 70%)

## Local Development

### Run Weekly Enhancement
```bash
python -c "from src.enhancement.param_tuner import run_weekly_enhancement; run_weekly_enhancement()"
```

### Run Monthly Grid Search
```bash
python -c "
from src.store.signal_store import load_store
from src.enhancement.param_tuner import run_grid_search
store = load_store()
result = run_grid_search(store, test_window_days=60)
print(result)
"
```

### View Signal History
```python
import pandas as pd
df = pd.read_parquet("data/signals.parquet")
print(df[df["ticker"] == "OBRI"][["run_date", "recommendation", "score", "entry_price"]])
```

## Requirements
- Python 3.11+
- TradingView TA library (handles batch scanning)
- yfinance for fallback and fundamentals
- XGBoost, scikit-learn for ML
- TA-Lib (system package: `libta-lib-dev`)

## Notes
- **EGX trading hours**: Sun-Thu, 10:00-2:30 PM local (close ~3:30 PM)
- **UK time**: 5 PM = 16:00 UTC (winter) / 17:00 UTC (summer)
- **Signal store uses Git LFS** - ensure `git lfs install` before cloning
- **YOLOv8 ChartScanAI** - Place model at `weights/custom_yolov8.pt` to enable

## License
MIT