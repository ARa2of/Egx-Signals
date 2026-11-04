# EGX Forecast System + Hourly Range Overlay — Architecture & Verification

Date: 2026-11-03 · Author: EGX Assistant (built with the user) · Status: live

This document describes the whole forecasting pipeline end to end, then the
new hourly range overlay in detail: what it does, how it is wired, the test
results, how to re-run every test, and the known gaps. It is written so a
third party can read it and assess the architecture without running anything.

---

## 1. The forecasting system in one page

**Idea.** Predict tomorrow's trading RANGE (high and low) for 21 EGX names.
Not the close, not direction — only the band. The predicted low is the
bid/support zone, the predicted high is the peel/sell zone.

**Why range-only.** On the October 2026 holdout, day-ahead CLOSE predictions
had ~zero skill (registry: only 7 of 21 names' close models had skill > 0 vs
the predict-no-change baseline; the accuracy table shows 15/20 failed), while
day-ahead HIGH and LOW beat the baseline on 21/21 and 21/21 names
(`models/registry.json`, P1 audit `docs/HOURLY_RANGE_AUDIT_20261103.md` §2).
The earlier "16 of 20" figure in this doc was unsourced and has been
replaced. Skill = 1 − rmse/rmse0 where rmse0 predicts a 0 return (next close
equals today's close); note this baseline is weak — P3 re-scores against
persistence / typical-excursion / ATR-box baselines. So the system predicts
the range and refuses to call direction.

**Components.**

| Piece | Where | What it does |
|---|---|---|
| History cache | `data/intraday/*_{daily,1h,15m}.parquet` | TV (tvDatafeed) bars per name + EGX30/EGX70EWI/EGX100EWI indices |
| Feature builder | `egxfc/build_daily_features.py`, `egxfc/build_features.py` | Daily + intraday feature matrices (36 base features; home-index variants) |
| Trainer | `egxfc/train.py`, `egxfc/rolling.py` | Trains once a month, persists bundles to `models/<T>/<leg>.pkl` + `models/registry.json` |
| Predictor | `egxfc/predict.py` (`predict_leg`) | Loads the SAVED bundle, never retrains |
| Daily EOD job | Task Scheduler `EGX-EOD-Daily`, Sun–Thu 16:30 Cairo | Fetch → score → rebuild `forecast.html` dashboard |
| Live loop | Task Scheduler `EGX-Live-Forecast`, Sun–Thu from 09:58 Cairo (`egxfc/live.py`) | Polls the EGX beta tape, aggregates 15m/1h bars, splices them over TV history |
| Dashboard | `egxfc/gui.py` → `output/intraday_prediction/forecast.html` | Plotly, offline (JS inlined), phone-first, served over LAN + Tailscale |
| Forum cards | `Scripts/egx_forecast_live.py` (agent root) | Per-name Discord forum threads: fixed 8-row daily ladder, patched in place at :05 passes, stripped post-close |
| Overlay (NEW) | `egxfc/overlay.py` | Narrows/shifts the day-ahead band at 11:00/12:00/13:00 Cairo |

**Cadence rule.** Daily = fetch/score/forecast only. Monthly = retrain + review.
One session is one row in ~3,000; re-picking models daily flaps the family and
chases yesterday.

**Retired (2026-11-03).** Intraday 1h/15m point predictions are off display —
skill ~0. Models kept on disk, loop still aggregates their bars because the
overlay consumes them.

---

## 2. The hourly range overlay

**Problem.** The day-ahead band is static. By 12:00 the session has already
printed 8 of 18 bars — the morning's carved high/low plus the home index's
move contain real information the static band ignores.

**Rule.** The overlay NARROWS or SHIFTS the band as the session prints. It
never widens it (except to contain what already printed — see §5). If the tape
truly breaks the band, that is the forecast being wrong, and the card should
say so, not redraw the lines around the miss. Rationale: widening on intraday
noise ratchets the band outward until everything sits inside it and it says
nothing about where to bid or peel.

**Checkpoints.** 11:00 / 12:00 / 13:00 Cairo ≈ 78% / 56% / 33% of the session
left (14/18, 10/18, 6/18 of the 18-bar session — an earlier version of this
doc said 72/50/28, which was the prototype's rounded fractions, not the
actual bar counts). The FEED label (Europe/London, tz-naive TV stamps) is
derived **per session date** with `zoneinfo` (`overlay.feed_cutoff`): the
Cairo–London gap is 2h normally, 1h during the spring mismatch
(London 03-29 → Cairo 04-24) and 3h during the autumn mismatch
(London 10-25 → Cairo 10-30). Hard-coded 09:00/10:00/11:00 labels were
wrong on those dates (P1 audit §2.4); fixed in P2 with golden tests.

**Maths per checkpoint** (graduated unchanged from the 2-name prototype
`hourly_range_test2.py`):

1. `beta` = covariance(stock, HOME-index daily returns) / variance(home),
   floored/clipped to [−0.5, 2.0], fallback 0.6. Cached per name.
2. `idx_move` = home-index session-open → checkpoint return.
3. `pace` = volume-so-far / (average daily volume × fraction elapsed),
   clipped to [0, 5]. `belief` = min(1, pace/1.5); thin tape (pace < 0.8)
   gets belief × 0.4. Note: the function is named `avg20_volume` but uses
   `tail(40)` of daily volume (doc/code mismatch flagged in the P1 audit;
   left as-is — changing it would retune a frozen constant).
4. `w` = √(fraction left). Exhaustion guard: pace > 3 while the carved range
   already covers >80% of the predicted width → w × 0.7.
5. Shift each edge: `adj = anchor + beta × idx_move × live × belief`.
   Blend toward live: `upd = live + (adj − live) × w`.
6. Contain: updated high ≥ max(carved high, live price); updated low ≤
   min(carved low, live price). Clamp inside the anchor band wherever the
   carve allows it.

**Home index per name** (`egxfc/enrich.py::home_key`, verified 2026-11-02):
EGX30 → AMOC, ORAS, ETEL, MFPC, EGAL, EFID, ADIB. EGX70EWI → POUL, ARAB,
SWDY, AMER, MASR, ISMQ. EGX100EWI → everything else (MBEG, EGAS, SAUD, CPCI,
AXPH, ADCI, GGCC, ICFC).

**Data flow in session.**

```
EGX beta tape ──► live.py Bars ──► data/intraday/live/*.parquet
                                        │
day-ahead anchors (predict_leg daily) ──┤
TV 15m cache + home-index 15m ──────────┤
                                        ▼
                          overlay.compute_all() at 11/12/13
                                        │
                        ┌───────────────┼───────────────┐
                        ▼               ▼               ▼
              live_range.json    GUI rebuild     forum :05 pass
              (1 file, 21 rows)  (LIVE badge)    (footer line)
```

`live_range.json` shape: `{generated, date, ck, cairo, feed, rows: {SYM: {rem_h,
rem_l, live, carved, anchor, pace, belief, w, exhausted, idx_move, beta, ck,
cairo, feed}}}` (`ck`/`cairo` are the Cairo checkpoint, `feed` the
date-specific Europe/London label — added by the P2 DST fix). Written
atomically (temp + `os.replace`) since P6. All three consumers stale-guard
on the `date` field.

---

## 3. Test results

### 3a. Overlay selftest — PASS (46 checks + replay)

Command: `python -m egxfc.overlay --selftest` (from
`scripts/intraday_prediction/`). Stages:

1. **Clock golden** (31 checks, `--test-clock`): overlay_at maths frozen
   pre-fix (pace-clip/exhaustion/slow-belief), day_overlay on winter+summer
   dates (old == new code), active_checkpoint semantics, a 13-date
   `feed_cutoff` DST table (all 4 transition days + both mismatch windows),
   day_overlay on both mismatch windows.
2. **I/O** (15 checks, `--test-io`): idempotent jsonl append, atomic write
   leaves no `.tmp`, `join_forward` hand-computed forward extremes.
3. **Replay**: replays October 2026 day by day on MBEG + MASR — loads each
   day's daily feature row, predicts the anchors from the saved bundle, runs
   `day_overlay` at all 3 checkpoints off cached 15m bars, and asserts live-price
   and carved-extreme containment on every overlay.

Result (2026-11-03): 46/46 checks PASS · MBEG 48 overlays PASS ·
MASR 48 overlays PASS.

### 3b. Full-board October backtest — corrected 81–85% both-edge coverage

Script: `scripts/intraday_prediction/_overlay_backtest.py`. Same replay as the
selftest but for all 21 names, and instead of asserting containment it scores
the overlay against the REALISED rest-of-day high/low after each checkpoint,
plus the overlay width as a fraction of the anchor width.

**Corrected result (2026-11-03, post-DST-fix, 320 overlays per checkpoint):**

- 11:00 Cairo: n=320 · high covered 95% · low covered 87% · both 84% · width 118% of anchor
- 12:00 Cairo: n=320 · high covered 93% · low covered 87% · both 81% · width 111% of anchor
- 13:00 Cairo: n=320 · high covered 92% · low covered 92% · both 85% · width 107% of anchor

Pre-fix numbers previously published here (85/84/87, n=320/320/315) were
inflated by the DST look-ahead on sessions 10-25..29 — see §4 gap #2.

**These numbers are still on the OLD basis** (16 sessions from a stale
features file, in-sample anchors, ICFC silently missing — 320 = 20 names ×
16 sessions; P1 audit). The tautology-free re-score on honest walk-forward
anchors is P4 (`python -m egxfc.eval_overlay`) and supersedes this table;
its already-in / new-extreme split is the number to trust.

Reading: the misses are breakout days where the rest of the session ran past
even the carve-inclusive band. Widths above 100% occur when the morning carve
itself already broke the anchor — containment of what printed takes precedence
over narrowing, by design.

### 3c. Pipeline verification — PASS (with one pre-existing data note)

- `python -m egxfc.live --selftest`: bar grid, aggregation and splice checks
  pass except one CPCI 15m mismatch (see §4, gap #1).
- `python -m egxfc.gui --bars 5`: dashboard rebuilds, 21 names, ~4.4 MB.
- `python Scripts/egx_forecast_live.py --selftest` (agent root): card
  parse/build/strip PASS.
- `range_line` unit check: synthetic `live_range.json` rendered
  `LIVE 12:00 299.50-309.10` for a covered name and `''` for an uncovered
  one; test file removed afterwards.
- `store_tape` check: wrote a well-formed `live_forecast.json` (tape closes
  only, EGX30 excluded); test file removed afterwards.

---

## 4. Known gaps

1. **CPCI missing its last 15m bar on 2026-11-03** (TV never printed the
   12:15 feed bar; refetch adds +0). Effect: `live --selftest` grid check
   stays FAIL(1) on that date permanently. Cosmetic — the overlay does not
   depend on it. Fix path: none on our side; TV gap.
2. **Daylight-saving re-cut — FIXED (P2, 2026-11-03).** Checkpoint keys are
   now Cairo wall-clock and each session's feed label is derived per date
   with `zoneinfo` (`overlay.feed_cutoff`) — correct on normal (2h),
   spring-mismatch (1h, Mar 30–Apr 24) and autumn-mismatch (3h, Oct 25–29)
   dates including transition days. Golden tests: `python -m egxfc.overlay
   --test-clock` (31 checks, incl. a 13-date DST table). The old
   `CK_CAIRO` constant and winter-only labels are gone; the prototype-era
   `feed+3` comments must still not be trusted.
   Corrected October backtest after the fix: both-coverage 84/81/85
   (was 85/84/87 pre-fix; the 12:00 checkpoint lost 3pt — its look-ahead
   was the largest, as the audit estimated).
3. **Coverage is 81–85%, not 100%** (corrected post-DST-fix figures; the
   tautology-free P4 numbers may differ). Breakout days escape the band. The
   system reports the band; it does not redraw around a miss. No stop,
   ticket or gate decision may use the band as a trigger — forecasts remain
   research output, never overriding Deep Check, gate, or stops.
4. **Exhaustion guard is heuristic** (pace > 3, carve > 80% of anchor width,
   w × 0.7). Sensible, unoptimized; revisit with a full season of data.
5. **Beta is static per name** (computed once from daily closes, cached in
   process). A regime change in a name's index linkage will not show up until
   the cache is cleared (process restart).
6. **Forum delivery depends on the bot token + thread map** (`egx_live_tags`,
   `egx_live_threads` in the agent root). The overlay only supplies the line;
   posting failures there are outside this pipeline.
7. **Forward-outcome logging — DONE (P6, 2026-11-03).** The live loop now
   appends every checkpoint row to `data/intraday/range_outcomes.jsonl`
   (append-only, idempotent on date/ck/sym) and writes `live_range.json`
   atomically (temp + `os.replace`). Realised forward extremes are joined
   lazily at read time (`overlay.join_forward`) — no EOD-job changes.
   First honest accuracy numbers come from the forward log starting
   2026-11-04 (October is burnt: constants, selection and refit all saw it).

---

## 5. Invariants (do not break)

- Predict loads bundles; it never retrains. Training happens monthly.
- The overlay narrows or shifts; it never widens around a miss.
- Consumers stale-guard on file dates; a stale band is never displayed.
- Test artifacts (`live_range.json`, `live_forecast.json` written by hand)
   are deleted after the test — only the live loop writes them in session.
- Forecasts are research, never a trade trigger.
