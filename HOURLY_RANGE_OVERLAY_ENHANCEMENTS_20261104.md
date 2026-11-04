# Hourly Range Overlay — Verify & Harden Enhancements

Date: 2026-11-04 (pass run 2026-11-03) · Status: **FINAL — all phases
complete; P4 ship bar passed, P5 shipped no change** ·
Working dir: `scripts/intraday_prediction`

Companion documents:
- `docs/HOURLY_RANGE_AUDIT_20261103.md` — P1 pre-edit audit (written before any code change)
- `docs/HOURLY_RANGE_OVERLAY_WORKLOG.md` — phase-by-phase log, resume point
- `docs/HOURLY_RANGE_OVERLAY.md` — original handoff doc (corrected in this pass, see §6)

Plan: P0 selftest → P1 audit (read-only, first) → P2 DST fix → P3 Layer-1
walk-forward rescore → P4 tautology-free overlay eval + ablation →
P5 ship-bar-gated changes → P6 forward logging + atomic writes → P7 this doc.
No module was relocated; nothing outside `scripts/intraday_prediction/` +
`docs/` was touched; nothing was committed to git.

---

## 1. Enhancements shipped

### 1.1 P2 — DST-correct checkpoints (bug fix)

The old code hard-coded feed labels (`09:00/10:00/11:00` Europe/London) for
Cairo checkpoints `11:00/12:00/13:00`. That mapping holds only when Cairo and
London are both on standard or both on DST time. During the two annual
mismatch windows the gap is 1h (spring, London 03-29 → Cairo 04-24) or 3h
(autumn, London 10-25 → Cairo 10-30), so the old code read **1 hour of
look-ahead** (autumn) or **found no bars** (spring) at the checkpoint.

- Checkpoint keys are now **Cairo wall-clock** (`CKPTS = 11:00/12:00/13:00`);
  `CK_CAIRO` removed (redundant).
- New `feed_cutoff(day, ck)` derives the Europe/London label **per session
  date** with `zoneinfo` — correct on normal, both mismatch windows, and all
  four transition days.
- `day_overlay` filters stock AND index bars by that date-specific label;
  payload/rows gained a `feed` field (`cairo` kept — GUI badge unchanged).
- Golden test `clock_selftest()` written and run **before** the edit
  (13 checks against then-current code), then the edit, then 31 checks PASS:
  overlay_at maths frozen pre-edit, day_overlay winter+summer (old == new),
  active_checkpoint semantics, 13-date DST table, day_overlay on both
  mismatch windows (autumn: old read 1h ahead; spring: old returned None).

Impact, corrected October backtest (`_overlay_backtest.py`, all post-fix):

| ck (Cairo) | pre-fix both | post-fix both | Δ |
|---|---|---|---|
| 11:00 | 85% | 84% | −1pt |
| 12:00 | 84% | 81% | −3pt |
| 13:00 | 87% | 85% | −2pt |

The audit's estimate (up to +12pt look-ahead at 12:00 ck on affected
sessions, ≈1–3pt pooled) is confirmed.

### 1.2 P3 — Layer-1 walk-forward rescore (new module, read-only)

`egxfc/rescore_l1.py` re-scores the day-ahead band under frozen definitions
(declared in the module docstring before any number existed, selftest-checked
against hand-computable fixtures): ATR(14) units; baselines persistence /
typical-excursion(20) / ATR-box (k=0.5 a priori) plus the legacy no-change;
Winkler @ α=0.2; day-level pooling; day-block bootstrap (block 5, B=1000,
seed 20261103) vs best-of-3 baselines; family fixed from registry (no
per-month selection); common support; guards MIN_TRAIN 300 / MIN_TEST 15.
Read-only w.r.t. bundles and registry.

**Verdict — the model LOSES to strong baselines** (full run, 21 names,
62 scorable months 2021-09..2026-10, 1253 pooled day-rows):

- pooled delta **+0.016 [+0.011, +0.021] ATR → LOSES** (model 0.396 vs
  typ_exc 0.381 band-MAE)
- month verdicts: BEATS 4 · MIXED 32 · LOSES 27; every month 2025-10 →
  2026-08 is LOSES
- **all 21 names** positive mean delta (best MASR +0.017, worst AXPH +0.080)
- 62 not 121 months because `egx100_ret1/5` exist only from 2020-05 →
  common support starts 2020-05-17 → MIN_TRAIN floors first month at 2021-09
  (probed, declared, not a bug)

Reading: the registry's "skill > 0 on 21/21" is vs the weak predict-0
baseline (P1 audit §2). Against persistence / typical excursion / ATR box the
ML layer adds no band-MAE value. The day-ahead band remains usable (the
baselines themselves score ~0.38 ATR), but skill claims must be quoted
against strong baselines from now on.

### 1.3 P6 — forward logging + atomic writes

- `data/intraday/range_outcomes.jsonl` — append-only checkpoint log,
  one line per (date, ck, sym), **idempotent** (replay / process restart
  appends 0 duplicates); written by `compute_all` (best-effort, never raises);
  read back with `load_range_log()`; realised forward extremes joined lazily
  at read time with `join_forward()` (date-specific feed cutoff, three-way
  split flags, coverage) — **no `eod.py` changes**.
- `live_range.json` — written atomically (temp + `os.replace`); GUI and
  other readers can never observe a partial file.
- Clean OOS forward log starts **2026-11-04** (October 2026 is burnt:
  constants, selection, and refit all saw it — report-only).

### 1.4 P4 — tautology-free eval + ablation (DONE)

`egxfc/eval_overlay.py` scores the overlay against **forward** extremes
(bars strictly after the checkpoint) instead of whole-session extremes
(which containment makes trivial once already printed — P1 audit §2.6).
Frozen protocol in the module docstring: months 2025-09..last complete;
selection 2025-09..2026-07, LOCKED 2026-08..; walk-forward anchors by
default (bundle mode = backtest-compat); causal beta/volume (daily history
strictly before the session day); three-way split (already-in fraction
reported, never counted as coverage); primary pooled Winkler @80,
ATR-normalised, day-level pooling + day-block bootstrap; ablation
B0 anchor / A0 contain / A1 +beta / A2 +pace-belief / A3 +exhaustion with
**constants frozen** (pace 0.8/1.5, exhaustion 3/80%/0.7 — terms switched,
never retuned).

Ablation flags are keyword-only params on `overlay_at(shift/belief_on/
exhaust_on, defaults=True)` + `day_overlay(**kw, beta_cache=)` — defaults are
production behaviour and are golden-guarded (31 clock checks still PASS).

**Ship bar (declared before the run):** on LOCKED months A3 must beat BOTH
A0 and B0 on pooled Winkler (ci_hi < 0) to ship as-is; otherwise P5
evaluates the conformal replacement.

**Result (full run, 21 names × 14 months 2025-09..2026-10, walk anchors):**

Tautology quantified: **already-in 77% (high) / 60% (low)** pooled — most of
the session extreme is already printed by the checkpoint, which is exactly
why the old whole-session coverage numbers were inflated. Honest
new-extreme coverage (pooled): A0 0.874/0.830, A1–A3 ≈0.75/0.65, B0 same as
A0 — the shift trades new-extreme coverage for width (A3 width 1.006× anchor
vs A0 1.201×) and wins on the primary metric.

Deltas (day-block bootstrap, pooled / selection / locked):

| comparison | pooled | selection | locked |
|---|---|---|---|
| A3 − A0 | −0.140 BEATS | −0.149 BEATS | **−0.106 [−0.123, −0.088] BEATS** |
| A3 − B0 | −0.205 BEATS | −0.200 BEATS | **−0.220 [−0.271, −0.178] BEATS** |
| A1 − A0 (beta term) | −0.129 BEATS | −0.138 BEATS | −0.098 BEATS |
| A2 − A1 (pace/belief) | −0.012 BEATS | −0.011 BEATS | −0.012 BEATS |
| A3 − A2 (exhaustion) | +0.001 MIXED | −0.000 MIXED | +0.004 MIXED |

**Ship bar: SHIP as-is** (LOCKED: A3 beats A0 and B0, ci_hi < 0 on both).

Term reads: **beta shift carries the overlay** (−0.129 of the −0.140 total);
pace/belief is small but significant (−0.012); **exhaustion is neutral**
(MIXED, delta ≈ 0) — it does fire naturally (n_exh 1012 pooled / 214 locked,
instrumented after the smoke run showed 0), it just doesn't help or hurt.
By-checkpoint: A3 wins at every ck; already-in rises with the day
(67%/78%/87% at 11/12/13).

### 1.5 P5 — ship-bar-gated changes (DONE — no change shipped)

> **P5 OUTCOME: no change shipped (documented positive).** The ship bar
> passed, so the production overlay stands as-is; no conformal replacement
> evaluated. The only ablation candidate it surfaced — dropping the
> neutral exhaustion term — is optional (terms may be dropped, constants
> never changed) and is recorded as an open item, not done here: A3 ≈ A2 is
> MIXED, so removing it buys nothing measurable and would invalidate
> comparability with the frozen-prototype golden tests for no gain.

---

## 2. Files and functions

| File | Change |
|---|---|
| `egxfc/overlay.py` | P2: Cairo ck keys, `feed_cutoff()`, date-specific bar filters, `feed` field, `CK_CAIRO` removed. P4: `overlay_at` ablation flags (kw-only, default = production), `day_overlay(**kw, beta_cache=)`, `beta_vs_home(cache=)`. P6: `_atomic_write_text()`, `RANGE_LOG`, `log_range_rows()`, `load_range_log()`, `join_forward()`, `io_selftest()`; `compute_all` writes atomically + logs. Tests: `clock_selftest()` (31), `io_selftest()` (15), flags `--test-clock` / `--test-io`. |
| `egxfc/rescore_l1.py` | **new (P3)** — frozen-definition walk-forward rescore; `selftest()` (18 checks incl. leakage guard); writes `rescore_l1.{csv,md}`, `rescore_l1_delta.csv`, `rescore_l1_by_name.csv`. |
| `egxfc/eval_overlay.py` | **new (P4)** — forward-extreme eval + ablation; `selftest()` (8 checks); writes `eval_overlay_*` under `output/intraday_prediction/`. |
| `egxfc/live.py` | P2: checkpoint log line prints Cairo + feed label. (P6 log fires via `compute_all` — no other change.) |
| `_overlay_backtest.py` | P2: print uses Cairo ck; its own future-bar filter switched to `feed_cutoff(day, ck)` (the first re-run caught this: 13:00 got n=0). |
| `docs/HOURLY_RANGE_OVERLAY.md` | P7: corrected frac-left, "16 of 20", avg20=40 days, DST gap → FIXED, logging gap → DONE, §3a/§3b numbers, `live_range.json` shape. |
| `docs/HOURLY_RANGE_AUDIT_20261103.md` | P1 (pre-edit) + P7 §5 note (probe outputs are pre-fix). |
| `docs/HOURLY_RANGE_OVERLAY_WORKLOG.md` | running log + resume point. |
| one-off probes | `_audit_p1{,b,c}_20261103.py`, `_p2_golden_gen_20261103.py` (read-only / print-only). |

Untouched: `src/` engine, gate, Deep Check, stops, tickets, sizing, close
models, 1h/15m point models, `eod.py`, `enrich.home_key`, registry,
bundles, all frozen constants.

---

## 3. Verification commands and results

```bash
cd "D:\Stocks\EGX Project\scripts\intraday_prediction"
python -m egxfc.overlay --test-clock        # 31 checks PASS (golden frozen pre-edit + DST table)
python -m egxfc.overlay --test-io           # 15 checks PASS (idempotent log, atomic write, join)
python -m egxfc.overlay --selftest          # PASS (31 + 15 + MBEG/MASR 48 overlays each)
python -m egxfc.rescore_l1 --selftest       # 18 checks PASS (frozen definitions + leakage guard)
python -m egxfc.eval_overlay --selftest     # 8 checks PASS (flags default==production, split, winkler, pooling)
python -m egxfc.rescore_l1                  # full: 21 names × 62 months → LOSES pooled (+0.016 [+0.011,+0.021])
python -m egxfc.eval_overlay                # full: 21 names × 14 months → ship bar PASSED (SHIP as-is)
python -m py_compile egxfc\live.py egxfc\overlay.py egxfc\eval_overlay.py egxfc\rescore_l1.py   # COMPILE OK
```

Outputs: `output/intraday_prediction/rescore_l1.{csv,md}`,
`rescore_l1_delta.csv`, `rescore_l1_by_name.csv`, `eval_overlay_walk.md`,
`eval_overlay_{split,delta,month}_walk.csv`, `eval_overlay_rows_walk.parquet`.

Corrected October backtest (post-P2, old basis): 84/81/85 both-coverage,
widths 118/111/107% of anchor — see §1.1.

---

## 4. Invariants preserved (audited)

- `predict_leg` never retrains; day-ahead band immutable; overlay narrows or
  shifts, never widens past the anchor except to contain printed carve;
  consumers stale-guard on `date`; clocks via `zoneinfo`; no CPCI
  special-case; no `enrich.home_key` edits; October untouched as a tuning
  target (pace 0.8/1.5, exhaustion 3/80%/0.7 frozen — P4 switches terms
  only); no gate/stops/sizing/close-model/1h-15m-point-model changes.
- Consensus invariant in `AGENTS.md` untouched (`src/` never edited).

---

## 5. Open items

1. **Forward log accrues from 2026-11-04** — first honest Layer-2 accuracy
   numbers after ~1 live week; `join_forward` + `load_range_log` are the
   read path. Watch: new-extreme coverage drift vs the eval's 0.75/0.65.
2. **Neutral exhaustion term** — P4 ablation: A3 ≈ A2 (MIXED, Δ ≈ 0) but it
   does fire (n_exh 1012 pooled). Dropping it is legal (terms may be
   dropped, constants never changed) but buys nothing measurable and would
   break comparability with the frozen-prototype golden tests — left in.
3. **Layer-1 anchor source** — P3 shows the ML band ≈ typ_exc on band-MAE;
   P4's B0 (frozen ML anchor) loses to every overlay variant, but the
   overlay's job is to improve on the anchor, which it does (A3 − B0 =
   −0.220 locked). Swapping the anchor source (e.g. typ_exc) is a *model
   change* — out of scope here, needs its own gated pass.
4. **Registry skill claims** — high/low 21/21 vs rmse0 only; docs now say
   so, but `models/registry.json` semantics stay as-is (not ours to edit
   in this pass).
5. **`avg20_volume` uses `tail(40)`** — documented; changing it would
   retune a frozen constant (deliberately not done).
6. **October 2026 = report-only** — any table quoting October is in-sample
   or burnt; clean confirmation is the Nov-forward log.
7. **ICFC has no `daily_features_ICFC.parquet`** (P1) — the backtest's
   `except: continue` drops it silently (320 = 20×16). egxfc's own matrix
   includes ICFC fine; only the legacy replay scripts are affected.
