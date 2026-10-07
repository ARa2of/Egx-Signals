# EGX Signals V2 — Methodology & Math Reference

**Audience:** any engineer or AI agent working on this codebase.
**Purpose:** explain exactly how a ticker becomes **Buy / Watch / Avoid**, the math behind every stage, and the invariants that must not regress.
**Config version:** v2.3 (`config/params.yaml`). Bump `version` + `updated` when weights or thresholds change.

This document describes the code **as implemented**. If this doc and the code disagree, the code wins — then fix this doc.

---

## 1. System overview

```
config/tickers.xlsx  (universe)
        │
        ▼
DATA LAYER
  TradingView batch TA + yfinance OHLCV (.CA, 10y) + FX (EGP=X)
  5-min intraday (tvDatafeed) for Buy/Watch tickers only
        │
        ▼
SIGNAL LAYERS (independent)
  1. Base technical score   0–100   src/signals/scoring.py
  2. ML forecast + conviction 0–100 src/signals/ml.py
  3. ChartScan YOLO votes           src/signals/chartscan.py
        │
        ▼
CONSENSUS (scripts/daily_run.py ~742–902)
  strong_technicals override → vote rules → override filters
        │
        ▼
Trade plan (ATR entries/stops/targets) + holding window
        │
        ▼
Outputs: Excel, latest_signals.csv, docs/index.html,
         agent_data.json, data/signals.parquet (append-only)
```

The final label is a **consensus of three independent layers**, never a single score. The pipeline runs daily via GitHub Actions (~16:00 UTC, Sun–Thu) and locally via:

```bash
python scripts/daily_run.py config/tickers.xlsx -o output
```

---

## 2. Data layer

| Source | Role | Fallback |
|---|---|---|
| TradingView screener (`egypt`) | Real-time session TA, close, P/E, EPS | — |
| yfinance (`.CA` suffix) | 10y daily OHLCV for indicators/ML | — |
| `EGP=X` | FX conversion | — |
| tvDatafeed | 5-min bars (~400 bars ≈ 5 sessions), 4h cache | skipped → intraday neutral |

TradingView close overrides yfinance's last close when the EGX session is live; the session is appended to chart history with `volume=0`. Stale-data warning: last bar older than 4 sessions.

---

## 3. Indicator math (`src/signals/technical.py`)

All on daily bars via `enrich()`:

| Indicator | Formula as implemented |
|---|---|
| SMA_n | `rolling(n).mean()` |
| EMA_n | `ewm(span=n, adjust=False).mean()` |
| RSI_14 | `RS = SMA(gains,14)/SMA(losses,14)`; `RSI = 100 − 100/(1+RS)` |
| MACD | `EMA12 − EMA26`; signal `= EMA9(MACD)`; hist `= MACD − signal` |
| Bollinger | `SMA20 ± 2·std`; `%B = (C − lower)/(upper − lower)` |
| ATR_14 | EMA(span=14) of `max(H−L, |H−C_prev|, |L−C_prev|)` |
| ADX/DI | DM± smoothed EMA(14); `DI± = 100·EMA(DM±)/ATR`; `DX = 100·|DI+−DI−|/(DI++DI−)`; `ADX = EMA(DX,14)` |
| VWAP(20) | `Σ((H+L+C)/3 · Vol)/Σ(Vol)` rolling 20 bars; `dist = (C−VWAP)/VWAP` |
| Fibonacci | Retracements 38.2/50/61.8% of full-range swing high/low |

### 3.1 Volume profile (fixed-range)

20 price bins over the last 21 sessions:

- Bin volume = each bar's volume split evenly across the price bins its range spans (v2.3 — a wide bar used to count in full in every overlapped bin, skewing POC/VA)
- **POC** = bin with max volume (price = bin center)
- **Value area** = bins covering ≥70% of volume, selected highest-volume-first
- `above_poc = last_close > poc`

### 3.2 Support / resistance (cluster method)

1. **Swing points**: local highs/lows at orders 2, 3, 5
2. **Volume peaks**: local maxima of a 20–50-bin volume histogram (volume split evenly across a bar's spanned bins)
3. Candidates filtered to ±25% of last price, then clustered within 1.5%
4. Per-cluster score: `touches·2 + recency + cluster_size·0.5`, recency weighting recent touches `1/(1 + bars_ago/10)`
5. Strength: **Strong** (≥4 touches or score ≥6) · **Medium** (≥2 touches or score ≥3) · else **Weak**

### 3.3 Candlestick patterns

Rule-based on the last 5 bars: Doji, Hammer, Inverted Hammer, Shooting Star, Hanging Man, Strong Bull/Bear candles, Engulfing (±12 pts), Harami (±5), Three White Soldiers / Black Crows (±15). Net `score_delta` feeds the trade planner (capped ±15 there); `latest_signal` ∈ bullish/bearish/neutral.

### 3.4 Trend regime (`detect_trend_regime`)

Let `A = ADX`, `B = EMA50 > EMA200`, `P = close > SMA50`:

```
direction = bullish  if DI+ > DI− and P
            bullish  if B and P and DI+ ≥ 0.75·DI−      (EMA structure overrides DI)
            bullish  if B and P and DI+ ≥ 0.5·DI−       (even when DI− > DI+)
            bearish  if DI− > DI+ and close < SMA50
            bearish  if not B and close < SMA50 and A ≥ 25
            neutral  otherwise

regime = trending_up     if A ≥ 25 and direction = bullish
       = trending_down   if A ≥ 25 and direction = bearish
       = trending        if A ≥ 25, neutral, but P and B   (mapped to trending_up downstream)
       = transitioning   if 20 ≤ A < 25
       = ranging          if A < 20
```

**v2.2 fix (do not regress):** `resolve_rsi_regime()` maps `trending`→`trending_up`, `transitioning`→`ranging`. Before this, strong-ADX uptrends fell into ranging RSI bands and RSI 70–80 was punished as overbought.

---

## 4. Base technical score (`scoring.py`) — 0–100

Nine categories, each a sub-score in **[0,1]** × weight. Weights sum to **100** (enforced by `test_scoring_fixes.py`):

```
trend 26 · macd 9 · rsi 9 · volume 10 · adi 7 · support 7
vwap 8 · volume_profile 9 · intraday 15
```

`raw_score = Σ subscore_i · weight_i`, clamped [0,100].
**Bands:** Buy ≥ 60 · Watch ≥ 50 · else Avoid.
Missing data: category → 0 pts, except intraday → **0.5 (neutral)**.

Shared flag: `strong_trend = (ADX ≥ 25) or (EMA50 > EMA200)`.

### 4.1 Trend (26)

```
+0.55  price > EMA50 and EMA50 > EMA200      (full bullish alignment)
+0.35  price > EMA200 and EMA50 > EMA200
+0.15  EMA50 > EMA200, price < EMA50
+0.20  EMA20 > EMA50                          (diamond cross)
+min(0.25, max(0, (price−EMA200)/EMA200))     (distance bonus)
→ clamp [0,1]
```

### 4.2 MACD (9)

```
MACD > signal:  strength = min(1, |MACD−signal| / |signal|)
                score = 0.4 + 0.6·strength;  MACD > 0 → min(1, score+0.1)
else:           0
```

### 4.3 RSI (9) — regime zones from params.yaml

```
trending_up:   oversold 35 · healthy 45–72 · overbought 80
ranging:       oversold 28 · healthy 40–68 · overbought 70
trending_down: oversold 25 · healthy 35–55 · overbought 65

healthy_low ≤ RSI ≤ healthy_high   → 1.0
oversold ≤ RSI < healthy_low       → 0.6
RSI < oversold                     → 0.8 (reversal candidate)
healthy_high < RSI ≤ overbought    → 0.5 if ADX ≥ 25 else 0.25
RSI > overbought                   → 0.35 if ADX ≥ 25 else 0.1
```

Elevated RSI inside a confirmed trend is **strength**, not a sell.

### 4.4 Volume (10)

Two sub-scores summed then clamped [0,1]:

```
vol_multiplier (vol / avg20):
  ≥3.0 → 1.0 · ≥2.0 → 0.6+0.4(m−2) · ≥1.5 → 0.3+0.3(m−1.5)/0.5
  ≥1.0 → 0.1(m−1)/0.5 · else 0
buy_vol_multiplier: same shape, reaches 1.0 at 3.0x
```

### 4.5 Accumulation / ADI (7)

```
ADL slope (20d):  >0 → +0.5 · <0 → −0.5
MFI:              ≤20 → +0.3 · ≥80 → −0.3 · 40–60 → +0.1
→ clamp [0,1]
```

### 4.6 Support (7)

Only within 2% of a support level:

```
proximity = max(0.3, 1 − (distance/0.02)·0.7)
volume-confirmed bounce: min(1, proximity + 0.2)
```

### 4.7 VWAP (8) — trend-aware

```
pct = dist_VWAP·100
pct>0:  ≤2 → 0.6 · ≤5 → 0.8 · ≤8 → 0.65 (strong_trend) else 0.3
        >8 → 0.45 (strong_trend) else 0.3
pct≤0:  |pct|≤2 → 0.4 · ≤5 → 0.2 · else 0.1
```

Premium to VWAP = participation in a trend. The pre-v2.2 ≤5% mean-reversion penalty suppressed uptrending names.

### 4.8 Volume profile (9) — trend-aware

```
above POC:  within VA (price ≤ VA_high) → 0.8
            above VA_high → 0.7 (strong_trend) else 0.5
            no VA info → 0.5
below POC:  price ≥ VA_low → 0.6 (value-area low support) · else 0.2
```

### 4.9 Intraday (15) — 5-min bars, Buy/Watch tickers only

From `src/data/intraday_analysis.py`:

```
volume_skew = (vol_below_POC − vol_above_POC) / (vol_above + vol_below)   ∈ [−1, +1]
    bull_trend: negative skew (volume at highs) dampened ×0.3

momentum_score = RSI_recent·0.40 + (50 + RSI_accel·5)·0.35
                 + min(100, up_sessions·20)·0.25
    bull_trend: floor at RSI_recent when RSI ≥ 55;
                "decelerating" re-labelled stable if RSI ≥ 60

vwap_score = above_ratio·60 + (50 + avg_dist·5)·0.40
    bull_trend: floor 60 when above_ratio ≥ 0.6

ad_score ∈ [−1,+1]: AD-line slope (+0.4), vol-at-lows vs highs (+0.4 / −0.4,
    distribution penalty halved to −0.2 in bull trend), MFI >60 +0.2 / <40 −0.2

intraday_score = (50 + skew·50)·0.30 + momentum·0.30 + vwap·0.25
                 + (50 + ad_score·50)·0.15
    bull_trend floor: max(score, 55)   ← intraday noise cannot veto trend quality
```

Unavailable → 0.5 neutral in the base score.

---

## 5. ML layer (`src/signals/ml.py`)

**Task:** predict the **10-session forward log return**:

```
y = clip( log(C_{t+10} / C_t), −0.5, +0.5 )
direction label: 1 if C_{t+10} > C_t else 0
```

**Ensemble:** 3 quantiles (0.10 / 0.50 / 0.90) × {XGBoost `reg:quantileerror`, LightGBM `quantile`} + direction classifiers {XGB `binary:logistic`, LGBM `binary`}. Weights: XGB 0.5 / LGBM 0.5 per quantile and for P(up). XGBoost: 300 trees, depth 4, lr 0.05, subsample 0.8, min_child_weight 5; LightGBM mirrors it. Seeds fixed at 42.

**Data:** up to 10y daily; rows with any NaN feature dropped; `RobustScaler`; min 120 rows. Production model retrains on **all** valid rows; walk-forward (5 splits over the trailing 20%) only measures MAE. Forecast price: **`P_q = last_price · exp(r̂_q)`** — models predict log returns, so `exp()` is correct. An earlier `(1+r)` cache bug is fixed; do not reintroduce.

**Features (~40):** return lags 1/2/3/5/10 (clip ±0.5); SMA/EMA distance; RSI + OB/OS flags; BB%; ATR%; MACD cross & hist; ADX + strong flag; ROC 5/10/20 (clip ±0.5); stoch %K(14); historical vol 10/20; Garman–Klass vol; OBV slopes 10/20 (OLS via polyfit); volume ratio vs 20d MA (clip 0–5); VWAP distance; **v2.2 trend features** `above_ema20/50/200`, `days_above_ema50` (10d fraction), `sma50_gt_sma200`, `up_days_10`, `ret_sign_consistency` (10d mean of sign(log return)), `roc5_above_ema50`; peer returns 1/5/10d; sector 5d return, breadth (fraction of peers > SMA20), relative strength, dispersion; calendar day/month/month-end.

**Peers:** sector map from `config/EGX_Stock_Groups.xlsx`, max 4 peers.

**Cache:** `data/model_cache/{TICKER}_models.pkl`, TTL 14 days; a **feature-schema change forces retrain** (stored `feature_names` compared).

### 5.1 Conviction v3 (0–100)

```
upside%   = (P_med − P)/P·100     downside% = (P − P_low)/P·100
cone%     = (P_high − P_low)/P·100  asymmetry = (P_high − P)/(P − P_low)

Upside magnitude    ≥5→35 · ≥3→28 · ≥1.5→20 · ≥0.5→10 · >0→3
Cone tightness      ≤3→25 · ≤6→18 · ≤10→10 · ≤15→5
Asymmetry           ≥3→20 · ≥2→14 · ≥1.5→8 · ≥1→3
Downside protection down≤2 & high-up≥1 →12 · ≤4&≥2→8 · ≤6&≥3→5
P(up)               ≥0.60→+10 · ≥0.55→+6 · ≥0.52→+3 · <0.40→−4
MAE penalty         only if |upside| ≥1.5%: ratio>3→−10 · >2→−6 · >1.5→−3
Path-to-upside      high-up≥0.5 & down≤10 →+4 · ≥0.2 & ≤6 →+2
Dir. accuracy       ≥0.60→+20 · ≥0.55→+12 · ≥0.52→+6
Ensemble agreement  XGB>0 and LGBM>0 →+10 · signs disagree →−5

Bull-trend adjustment:
    upside ≥0.5 → +8 · upside ≥−1.0 → +6 · upside ≥−2.5 → +2
    floor: bull_trend and score<25 and upside≥−2.0 and downside≤8 → score = 25
```

`clamp [0,100]`. Labels: **≥70 High [GREEN]** · **≥45 Moderate [YELLOW]** · **≥25 Low [RED]** (threshold 20 in bull trend) · else **No Conviction [BLOCKED]**.

ML display tiers (used as the ML consensus vote): **Buy ≥ 35 · Watch ≥ 20 · else Avoid**.

**Known weakness:** tree ensembles still mean-revert after large runs (features look "overbought"; median forecast goes flat/negative). The bull-trend conviction floor and the consensus `strong_technicals` path are the designed safety net.

---

## 6. ChartScan (YOLOv8)

`weights/custom_yolov8.pt` (Git LFS; classes `{0: Buy, 1: Sell}`, source ChartScanAI), 180-candle lookback, confidence threshold 0.3. Vote sanitization in `daily_run.py`:

```
Buy  signal, conf ≥ 0.40         → vote = Buy
Sell signal, conf ≥ 0.50         → vote = Avoid
any other real signal (conf > 0) → vote = Watch
conf = 0 / model missing         → abstain (excluded from voting)
Sell + bullish candle context    → Neutral / abstain
```

Abstention prevents a noisy YOLO Sell from vetoing a technical Buy.

---

## 7. Consensus (`scripts/daily_run.py` ~742–902)

Active votes: **Base**, **ML**, **ChartScan** (abstentions removed). `B/W/A` = Buy/Watch/Avoid counts among active votes.

### Step 1 — strong-technical override (checked first)

```
strong_technicals = Base==Buy and death_cross≠Yes and golden_cross==Yes
                    and diamond_cross==Yes and ADX ≥ 30 and MACD bullish

if strong_technicals and ChartScan ≠ Avoid:
    final = Buy
```

**v2.2 invariant:** a confirmed uptrend structure stays **Buy** even if ML and ChartScan are cautious. Only a solid ChartScan Avoid (conf ≥ 0.50) can block it.

### Step 2 — vote rules (in order)

```
all active votes Buy (≥2)  → Strong Buy
B ≥ 2                      → Buy
B = 1 and A = 0            → Buy      (one Buy, nobody objects)
B = 1 and A ≥ 1            → Watch
W ≥ 1 and A = 0            → Watch
A ≥ 1 and B = 0            → Avoid
otherwise                  → Watch    (insufficient signals)
```

### Step 3 — override filters

```
Market-context gate (v2.3):
    final = Buy and buy_votes = 1 and avoid_votes = 0 and not strong_technicals
        and ticker's index sentiment = Bearish
        → downgrade to Watch (index Bearish annotation in basis)
    Exempt: strong technicals and 2+ Buy votes — independent evidence.
    Toggles: market_context.use_index_sentiment (params.yaml).
    Bullish/Neutral index → basis annotated `[<index> <sentiment>]` on Buys.

RSI hard cap:
    RSI > regime overbought threshold and final ∈ {Buy, Strong Buy}:
        tolerate if trend_exception = (ADX ≥ 30 and EMA50>EMA200 and price > EMA50)
        else downgrade → Watch

Momentum filter:
    reference = close of the most recent Buy in the signal store within the
                rolling window (thresholds.momentum_filter_lookback_days, default 60)
    price_drop = (current − reference) / reference
    price_drop < −5% and final ∈ {Buy, Strong Buy}  →  Watch
    no Buy within the window → filter skipped
    (v2.3: replaces the all-time first Buy, which pinned tickers at Watch forever)

Entry-quality gate (planner):
    entry_quality == "poor" (candidate score < 20) and not strong_technicals
        → downgrade Buy to Watch
    entry_quality == "chase"/"wait" ("CHASE — NO CLEAN PULLBACK")
        → direction kept; basis annotated with chase/wait risk
```

**Design rule:** entry quality ≠ direction. A missing clean pullback level does not make a trend unbuyable. This was the main consensus bug fixed in v2.2.

---

## 8. Trade plan (`src/trade/planner.py`)

**ATR context:** `stop_loss_atr = 1.8` · `max_stop_atr = 3.0` · `max_wait_atr = 3.5` · `min_rr = 3.0` · entry zone half-width `0.30·ATR` · max entry distance 8%.

**Entry candidates:** support levels, EMA20/SMA50 below price, Fibonacci levels below price, BB lower (if RSI < 45), VWAP below price, VP POC / VA low, breakout candidates above price (admitted if ML **median** or **High quantile** ≥ 1.02× that resistance, or direction P(up) ≥ 0.55 — v2.3; median alone blocked breakouts after big runs), current price. Candidates >8% below current price or >3.5×ATR away are rejected.

**Candidate score (0–100):**

```
R/R quality (35):     min(35, (rr/3)·35);  rr = reward/risk
                      reward = next resistance − entry
ML agreement (30):    min(15, upside%·15/2) + min(15, asymmetry·15/2),
                      × conviction/100 (multiplier floor 0.2)
Momentum (20):        RSI 35–60 → +10 · edge bands → +5
                      MACD > signal → +10 · MACD > 0 → +5
Level confluence (15):support within 0.4·ATR (+8/5/3 by strength), fib +7, MA +5
VWAP/VP (10):         within 0.3·ATR of VWAP +5 or above +2;
                      near POC +5 or inside VA +3
Candlestick:          net pattern points, capped ±15
Proximity penalty:    −6 per ATR below current price (−8 relief if within 0.5%)
```

Candidates with `rr < 3.0` are discarded. Best surviving candidate defines the entry zone `[price ± 0.3·ATR]`.

**Entry action:** distance-based — `BUY NOW` (≤ max(1.5%, 0.5·ATR%)), `WAIT FOR PULLBACK` (≤ max(5%, 1.5·ATR%)), `BUY ON BREAKOUT` (≤3% above), else deeper-wait labels. No surviving candidate → `CHASE — NO CLEAN PULLBACK` (quality flag, not a direction veto).

**Stop:** `max(entry − 1.8·ATR, strongest support below entry within 3.0·ATR × 0.992)`. Urgency: ≤2% → exit on intraday touch; ≤4% → exit on 1 close below; wider → 2 closes below.

**Targets:** fixed TP1/TP2/TP3 from `params.yaml trade.tp1_pct/tp2_pct/tp3_pct` (+1%/+2%/+6% defaults, backtest-tuned; v2.3: no longer hardcoded in planner), plus ML quantile prices and resistance levels within 20% of TP3 if their R/R ≥ 3 (`trade.min_rr`). Deduplicated within 0.5%, top 3 returned.

**Position size:** `shares = capital · (risk%/100) / (entry − stop)`.

**Holding window:** regime base (trending_up 1–4 weeks; ranging 1–3; transitioning 1–2), gated by conviction (≥70 → upper bound; ≥45 → midpoint; low → min) and candle signals; personalized per ticker by `holding_engine.py`. EGX week = 5 sessions (Sun–Thu).

---

## 9. Verification suite

Run after **any** scoring / consensus / ML change, before push:

```bash
python scripts/test_scoring_fixes.py      # regime map, VWAP/VP, weights sum = 100
python scripts/test_ml_v3.py              # conviction v3, direction classifiers
python scripts/test_consensus_cases.py    # ETEL/EFIH/MBSC/ARCC expectations
python scripts/verify_etel_fix.py         # live ETEL replay — must end Buy
```

**Invariant:** ETEL-class strong uptrends must end **Buy** via `strong_technicals`. If any test fails, do not push.

---

## 10. Change-control rules for agents

1. Weights, thresholds, RSI zones, trade constants live only in `config/params.yaml`; bump `version` + `updated` there.
2. `src/config.py` caches YAML at module load — after editing params:
   `python -c "from src.config import reload_configs; reload_configs()"`
3. Consensus thresholds were tuned after real user complaints (zero-Buy days) — change only with test evidence.
4. Do not remove: the `strong_technicals` path, RSI regime mapping, trend-aware VWAP/VP rules, bull-trend conviction floor, intraday bull-trend floor, or the chase/wait ≠ Avoid logic.
5. LFS required for `weights/custom_yolov8.pt` and `data/signals.parquet`; never force-push; never delete signal-store history.
6. Edit root `src/` + `scripts/` + `config/` — never the `egx-signals/` duplicate tree.
