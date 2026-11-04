# Range Forecast Audit — Layer 1 & Layer 2 (pre-edit)

Date: 2026-11-03 · Status: **written BEFORE any code edit** (per verify-first rule)
Scope: `scripts/intraday_prediction/` (egxfc + backtest scripts) · Data on disk through 2026-11-03
Companion: `docs/HOURLY_RANGE_OVERLAY_WORKLOG.md` (phase log) · Probes: `_audit_p1{,b,c}_20261103.py`

Every claim below was checked against code or data, not against
`HOURLY_RANGE_OVERLAY.md`. Probe outputs are reproducible with the commands
listed at the end.

---

## 1. Layer 1 — day-ahead high/low

### 1.1 Predictands
`build_daily_features.py::TARGETS`:
`t_high = high[t+1]/close[t] − 1`, `t_low = low[t+1]/close[t] − 1`
(+ `t_close`, `t_open`). Predictands are **next-session high/low as returns vs
today's close**, known at the EOD run (16:30 Cairo) of day t for session t+1.

**Leakage check — no same-session print enters any feature.** All 41 features
are functions of day ≤ t:
- index block (`egx30/70/100_ret1/ret5`, `breadth`): day-t index return, aligned
  by date to row t. Index closes 14:30 Cairo, EOD runs 16:30 → known at cutoff.
  Not a leak (the prompt's concern would only bite if a t+1 return sneaked in).
- `spx_ret_prev`/`brent_ret_prev`: last US session **strictly before** day t
  (`asof_prev` uses +1 ns shift). At EOD t the US session of day t has just
  opened (16:30 Cairo = 09:30 ET) so t−1 is genuinely the last completed one.
- `fx_ret1/ret5`: `.shift(1)` — previous FX session.
- engine block (`eng_*`): `src.signals` rolling indicators only (look-back
  windows); fibonacci/SR/patterns are computed by `enrich()` but **not** used as
  features.
- calendar/returns/candle/volume: day ≤ t by construction.

### 1.2 Loss
`train.py::skill_score`: **RMSE on return units**, `skill = 1 − rmse/rmse0`.
Not ATR-normalised, not pinball, no quantiles. Model selection = highest skill
on the test month; three candidates (ridge / rf / lgbm) — see 1.4 for the
selection caveat.

### 1.3 True baseline behind "beat no-change" (prompt deliverable 1)
`rmse0` = RMSE of predicting **0 return**, i.e. predicting
`high[t+1] = close[t]`. So the documented hurdle is *"tomorrow's high/low equal
today's **close**"* — weaker than persistence of yesterday's high/low (baseline
1 in the rescore plan) and weaker than typical excursion (baseline 2). The
claim "high and low beat no-change" is therefore true but **against a weak
hurdle**; the P3 walk-forward must re-score vs persistence and typical
excursion before any skill claim survives.

Registry (`models/registry.json`, test_month=2026-10, evaluated):
- `skill_high > 0`: **21/21**, `skill_low > 0`: **21/21**, `skill_close > 0`: **7/21**.
- Doc §1 says "all **20** names" and "16 of 20 failed on close". The excluded
  symbol is **ICFC** (added to the universe 2026-11-03; bundle trained
  2026-11-03T00:04). Reproducible counts differ slightly from the doc: the
  accuracy-table run (20 stocks, no ICFC) had 5/20 close-positive (15/20
  failed); registry has 7/21 (14/21 failed). **The doc's exact "16 of 20"
  could not be reproduced from either source — treat the doc number as
  unsourced; use 7/21 (registry) or 5/20 (accuracy table), both cited.**
- Doc §1 also reports `acc_touch`/`acc_close`: registry `acc_close` (realised
  close inside band) ranges 0.31–0.75, `acc_touch` (both edges reached)
  0.00–0.25 over the 16 October sessions.

### 1.4 Split / holdout hygiene — **two real problems**
1. **October anchors are in-sample.** `_select_and_refit` records the honest
   step-1 skill (fit pre-Oct, score Oct) but then `refit=True` fits the
   persisted bundle on **train + October** (`rows_used = 1247 = 1231 + 16`).
   `_overlay_backtest.py` and `overlay.selftest` predict October rows with that
   persisted bundle → the day-ahead anchors feeding the overlay backtest saw
   October during fitting. The recorded skill is honest; **the backtest's
   anchors are not**.
2. **Two cache layers can go stale.** `_overlay_backtest.py` reads
   `daily_features_{sym}.parquet` (written manually 2026-11-01 13:20) instead
   of `egxfc.features.build_matrix` (whose cache `train_{sym}_daily.parquet` is
   refreshed by the EOD job). On 2026-11-03 the inputs were refreshed (index/FX
   parquets 01:21, stock daily 14:34) and the on-disk features file was left
   behind: **it holds only 16 of October's 20 sessions** — the four Mondays
   2026-10-05/12/19/26 are absent from the file but present in raw bars and
   clean in a rebuild (probe C2: in-memory rebuild = 20 rows, disk = 16).
   egxfc's own matrix already has all 20 with no NaN (probe B4).
   Consequence: **the published backtest ran on 16 sessions, not 20.**

Scaler: `StandardScaler` inside each pipeline, fit on train only during
SELECT — but the persisted (refit) bundle's scaler saw all data incl. October
(problem 1). Features themselves are deterministic rolling transforms — no
feature fitting step exists.

### 1.5 Independent high/low fit & incoherence
High and low are fit as **separate models** (loop over targets). Raw-pair
incoherence over full history with current bundles (probe C2): **1 row out of
~31,000** (ARAB, 0.06%, width −0.001%). `predict.py` reports `band_pos_ok`
but does not clip; `overlay_at` containment makes it moot in practice.
Verdict: independent fit is fine today; a positive-excursion parameterisation
(P5) removes the theoretical risk for free.

### 1.6 Mild look-ahead inside overlay inputs (both in backtest AND live)
- `beta_vs_home` computes cov/var over the **entire daily history passed in**
  (through 2026-11-03) → October replay uses Oct/Nov returns for beta.
- `avg20_volume` = `tail(40).mean()` of that same full history → early-October
  checkpoints use late-Oct/Nov volume. (Also: **doc says "20-day average";
  code uses 40 days** — `overlay.py::avg20_volume`. Doc/code mismatch, like
  FRAC_LEFT.)
- For a strict walk-forward both must be sliced `≤ session date`. Live is
  mildly affected (beta/vol include today's own session in `dly`? — `dly` is
  the daily parquet, which during the session still ends at the previous
  session until EOD refresh, so live is *mostly* clean; the replay is not).

### 1.7 Data hazards (code-level confirmation)
- **Limit-hit days are silently dropped, not censored:** `close_pos` uses
  `(h−l)` with zeros → NaN → `dropna` removes the whole session from training
  AND from any evaluation built on the same matrix. Limit days vanish.
- **TV retro-adjustment:** cannot be verified from code; any bonus/split
  adjustment rewrites history → replay levels may differ from what that morning
  showed. Open item for live-vs-replay comparison (P4).
- **Ramadan 2027 (~Feb) / half-days:** `CKPTS`, `FRAC_LEFT`, and the 18-bar
  grid all assume 10:00–14:30 Cairo (`live.py::SESSION_OPEN/CLOSE`). A changed
  session length invalidates `14/18, 10/18, 6/18` and the checkpoint labels —
  no guard exists. Document, don't fix now (dates unknown until published).

### 1.8 Model-selection caveat
SELECT picks the best of 3 candidates **by skill on the test month**, then
refits it. Choosing a family on the test month and reporting that month's
skill is selection-biased (max of 3 noisy estimates on ≤16 sessions). The
P3 walk-forward (bundle that existed that morning, per month) fixes this for
the rescore; the production cycle keeps it (out of scope to redesign, noted as
inherent optimism in October's recorded skill).

---

## 2. Layer 2 — hourly overlay

### 2.1 Reconstructed equation (from `overlay.py::overlay_at`), units stated
```
frac_el   = 1 − FRAC_LEFT[ck]            # dimensionless, code: 4/18, 8/18, 12/18
pace      = clip(vol_sofar / (avg20vol × frac_el), 0, 5)   # [vol]/([vol]×1) = 1
belief    = min(1, pace/1.5); ×0.4 if pace < 0.8           # dimensionless
w         = sqrt(FRAC_LEFT[ck])          # dimensionless, 0.883 / 0.745 / 0.577
exhausted if pace > 3 and carved_w > 0.8 × pred_w → w × 0.7
adj_edge  = anchor_edge + beta × idx_move × live × belief
            # [EGP] + 1 × 1 × [EGP] × 1 = [EGP]   ✓ resolves to a price
upd_edge  = live + (adj_edge − live) × w # [EGP] + [EGP]×1 = [EGP]   ✓
rem_h     = max(carved_h, upd_h …) with live-containment, then
            clamp inside anchor band ONLY if carve already inside (L119-122)
rem_l     = symmetric
```
**Verdict on the prompt's unit concern:** in the *code* `live` is always a
price and the products resolve cleanly to EGP — no unresolved unit mix in
implementation. The **doc** overloads `live` in prose ("blend toward live"
reads like a regime flag). Real methodological issues remain: (a) `beta` is a
**daily** beta applied to an **intraday** index move (first ablation suspect,
per prompt); (b) `w` decays toward `live` as the session runs, so the anchor's
influence *fades* with time — documented behaviour, not a bug; (c) doc §2's
"72/50/28% left" is wrong — code uses `14/18, 10/18, 6/18` = **78/56/33%**
(consistent with the session math; the doc contradicts itself, its own
"8 of 18 bars by 12:00" ⇒ 56%).

### 2.2 Missing name in the backtest (prompt finding 2) — **solved**
`_overlay_backtest.py` iterates `DEFAULT_UNIVERSE` (21) but reads
`daily_features_{sym}.parquet`; **ICFC has no such file** (it was created via
egxfc's `build_matrix`, which writes `train_ICFC_daily.parquet`). The
`except: continue` at line 22-25 drops it silently → **320 = 20 names × 16
sessions**. Also the doc's "all 20 names" (§1) excludes ICFC.

### 2.3 Session count (prompt finding 2) — **solved**
16 sessions = stale features file (§1.4 problem 2). Rebuild → 20 October
sessions. So the published 955 overlays = 20 names × 16 sessions × 3 ckpts
(315 at 13:00 = 5 name-sessions with no future bar), not 21 × 16.

### 2.4 DST (prompt finding 4) — **confirmed and quantified**
`zoneinfo` (probe, not memory): Europe/London back **2026-10-26** 01:00 UTC
(last Sunday of Oct = Oct 25, transition at 01:00 UTC = Oct 26 00:00 by the
daily sampler's date boundary — the session of **Sun 2026-10-25** already runs
in the mismatch), Africa/Cairo back **2026-10-30** (offset change detected
between 10-29 and 10-30). Resulting Cairo−London gap:
- 2026-10-24/25: 2 h (normal) → sessions 10-25 onward after UK flip: **3 h**
- **sessions 2026-10-25, 26, 27, 28, 29: gap = 3 h** (UK on GMT, EEST still)
- spring mismatch 2026-03-30→04-24: gap = 1 h (probe: 2026-04-05 = 1 h)
Hard-coded feed labels `09:00/10:00/11:00` assume 2 h ⇒ on those sessions the
labelled checkpoint maps to **Cairo 12:00/13:00/14:00** — one hour LATE = the
replay uses an extra hour of bars (look-ahead) and leaves less future to cover.

Quantification (probe C3, harness reproduces the doc's as-is numbers exactly:
95/88/85 · 93/91/84 · 94/94/87 ✓):

| sessions | 11:00 both | 12:00 both | 13:00 both |
|---|---|---|---|
| all, as-is (published) | 85% (n=320) | 84% (n=320) | 87% (n=315) |
| affected 4 sessions, as-is | 89% (n=80) | **96%** (n=80) | **100%** (n=75) |
| affected 4 sessions, corrected labels | 85% (n=80) | **84%** (n=80) | 90% (n=80) |

→ On mismatch sessions the look-ahead inflates both-edge coverage by **up to
+12 points (12:00 ck)**; pooled October effect ≈ **−1 to −3 points** once
corrected. (Affected = 4 of the 16 backtested sessions: 10-26 is one of the
stale-file Mondays, so 25/27/28/29.) Live `compute_all` is NOT affected the
same way — it fires on the Cairo clock (`active_checkpoint` compares Cairo
strings); the hard-coded labels matter for the **feed-bar filter inside
`day_overlay`** (`bars ≤ ck`), which in live on a 3 h day grabs bars through
Cairo 14:00 when asked for "13:00" — i.e. **live can also over-grab near the
close on affected days**, worth confirming in P2's golden test.

### 2.5 Selftest (prompt finding 5) — confirmed
`overlay.selftest` replays cached **TV** 15m bars while production `compute_all`
feeds `spliced_15m` = TV + **beta-tape** overlay (live wins on duplicates).
Containment asserts are true by construction (carve floor/max). No skill
assertion, no volume-definition comparison, no pace-distribution comparison —
those belong to P4.

### 2.6 Coverage tautology (prompt finding 1) — mechanism confirmed
`_overlay_backtest` scores rest-of-day (`bars > ck`) vs `rem_h/rem_l`, and
`overlay_at` floors `rem_h` at the **carved** high. On days whose final extreme
already printed by the checkpoint, `rem_h ≥ carved_h ≥ rest_h` is guaranteed →
"high covered" is automatic. Widths 119/114/107% confirm the containment
exception routinely widens past the anchor (doc §2 "never widens" vs §3b
reality — the exception, not the rule, is doing the work). P4 must split:
final-extreme coverage / new-extreme-after-ck coverage / fraction already-in,
and score widths separately for carve-inside vs carve-outside days.

### 2.7 Constants frozen
pace 0.8/1.5, exhaustion 3 / 80% / 0.7, beta clip [−0.5, 2.0] fallback 0.6,
FRAC_LEFT — all **not** tuned in this pass (invariant). Doc corrections only.

---

## 3. October's status (prompt finding 6) — CONFIRMED burnt
Three independent burns: constants tuned to it (invariant list), candidate
selection on it (§1.8), refit including it (§1.4). October = report only.
Clean out-of-sample begins with the forward log from **2026-11-04** (P6).

## 4. Prompt findings 1–6: verdict summary

| # | Finding | Verdict |
|---|---|---|
| 1 | Overlay widens; containment exception overrides the rule | **Confirmed** (widths >100%; mechanism §2.6) |
| 2 | Coverage rests on few sessions; one name missing (320=20×16) | **Confirmed** — missing name = **ICFC** (§2.2); 16 = stale features file (§2.3); ±15pt CI claim stands, corrected sample will be 21×20 |
| 3 | Fraction-left constants wrong (72/50/28) | **Half-refuted** — code is right (78/56/33 = 14/18,10/18,6/18); **doc wrong**. Linear-time-as-volume-proxy critique stands (P4 ablation) |
| 4 | DST claim wrong, may contaminate October | **Confirmed + quantified** (§2.4): sessions 10-25..29 gap 3h; up to +12pt coverage inflation on those sessions, ≈1–3pt pooled |
| 5 | Selftest proves nothing about skill; TV vs beta-tape | **Confirmed** (§2.5) |
| 6 | October burnt | **Confirmed** (§3) |

New findings not in the prompt: in-sample October anchors (§1.4.1) · stale
features file → 16 sessions (§1.4.2) · `avg20_volume` uses 40 days + doc says
20 (§1.6) · beta/volume walk-forward violations (§1.6) · limit-hit sessions
dropped by `close_pos` NaN (§1.7) · "16 of 20" close claim unsourced (§1.3) ·
SELECT-on-test-month selection bias (§1.8).

## 5. Reproduction commands
```bash
cd "D:\Stocks\EGX Project\scripts\intraday_prediction"
python -m egxfc.overlay --selftest                       # PASS (MBEG/MASR 48 each)
python _audit_p1_20261103.py                             # A: ICFC file gap; B: 16 vs 20; C: incoherence
python _audit_p1b_20261103.py                            # NaN cols per missing date; registry provenance
python _audit_p1c_20261103.py                            # mtimes; rebuild=20 rows; DST as-is vs corrected; refit flags
```
Probe outputs quoted in this document are from 2026-11-03 and describe the
**pre-fix** state (the audit was written before any edit). Fixes landed after
this audit: P2 zoneinfo DST fix (`overlay.feed_cutoff`, golden tests) and P6
atomic write + `range_outcomes.jsonl`. Re-running `_audit_p1c_20261103.py`
now shows its corrected-DST column matching the shipped code; its
"as-is" column reproduces the historical (buggy) behaviour this audit
documented. The other findings (ICFC gap, stale features file, in-sample
October anchors, unsourced "16 of 20", `avg20`=40 days, selection bias)
remain open as recorded — see `HOURLY_RANGE_OVERLAY_WORKLOG.md` for what
each phase did about them.
