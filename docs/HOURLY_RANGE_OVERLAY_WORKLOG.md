# WORKLOG — Range Overlay Verify & Harden pass

Started: 2026-11-03 · Status: **DONE (P0–P7 all complete; ship bar passed, no change shipped)** · Working dir: `scripts/intraday_prediction`
Plan approved by user: **full P1→P7** (audit-first, gated).
Final deliverable (at the end): `docs/HOURLY_RANGE_OVERLAY_ENHANCEMENTS_20261104.md`

**How to resume if this pass breaks:** read "Phase status" below, find the first
`NOT DONE` item, re-run its verification command, continue from there. Every phase
lists its own verify command. Nothing outside `scripts/intraday_prediction/` +
`docs/` is touched; no module is relocated.

---

## Phase status

| Phase | What | Status | Verify command |
|---|---|---|---|
| P0 | Selftest + doc read + claim spot-checks | **DONE** | `python -m egxfc.overlay --selftest` → PASS |
| P1 | Audit both layers (read-only, written first) | **DONE** | `python _audit_p1c_20261103.py` → doc `docs/HOURLY_RANGE_AUDIT_20261103.md` |
| P2 | DST fix (zoneinfo-driven ckpts) + golden test | **DONE** | `python -m egxfc.overlay --test-clock` → 31 PASS + `--selftest` → PASS |
| P3 | Layer 1 walk-forward rescore (ATR units, no model change) | **DONE** — verdict: model LOSES pooled | `python -m egxfc.rescore_l1 --selftest` → PASS; outputs `output/intraday_prediction/rescore_l1.{csv,md}` + `_delta` + `_by_name` |
| P4 | Layer 2 tautology-free eval + ablation | **DONE** — ship bar PASSED | `python -m egxfc.eval_overlay --selftest` → PASS; full outputs `output/intraday_prediction/eval_overlay_*` |
| P5 | Ship-bar-gated model changes (may end in negatives) | **DONE** — no change shipped (bar passed) | §1.5 of enhancements doc |
| P6 | Logging/I-O: `range_outcomes.jsonl` + atomic `live_range.json` | **DONE** | `python -m egxfc.overlay --test-io` → 15 PASS; `--selftest` → PASS (clock+io+replay) |
| P7 | `HOURLY_RANGE_OVERLAY_ENHANCEMENTS_20261104.md` + doc corrections | **DONE** (doc FINAL; both handoff docs corrected) | doc review |

Invariants (never break): `predict_leg` never retrains · day-ahead band immutable,
overlay never overwrites · overlay never widens past the anchor except to contain
a printed carve · consumers stale-guard on date · clocks via `zoneinfo` · no CPCI
special-case · no `home_key` edits · no tuning to October (pace 0.8/1.5,
exhaustion 3/80%/0.7 frozen) · no gate/stops/sizing/close-model changes.

---

## P0 — DONE (2026-11-03)

- `python -m egxfc.overlay --selftest` → **PASS** (MBEG 48 overlays, MASR 48).
- `docs/HOURLY_RANGE_OVERLAY.md` read end-to-end. NOTE: doc has only §1–§5;
  "7 known gaps" = §4, invariants = §5. No §6/§7 exist (instruction referenced
  sections 5–7 — whole doc read instead, coverage complete).
- Claim spot-checks (read-only):

| # | Prompt claim | Verdict | Evidence |
|---|---|---|---|
| 1 | Overlay widens in practice | CONFIRMED | doc §3b widths 119/114/107% > 100%; carve-containment in `overlay_at()` L119-122 overrides "never widen" |
| 2 | 320 @11:00 ⇒ 20 names not 21 | CONFIRMED anomaly | `config.DEFAULT_UNIVERSE` = 21 (incl. ICFC); `_overlay_backtest.py` L20 loops all 21; 320 = 20×16 → one name drops out (ICFC suspect); doc says "all 20 names" vs 21 universe |
| 3 | Fraction-left constants wrong | HALF-REFUTED | `overlay.py` L34 `FRAC_LEFT = 14/18, 10/18, 6/18` = 78/56/33% (correct). **Doc §2 wrong** (says 72/50/28, contradicts its own "8 of 18 bars by 12:00"). Linear-time-as-volume-proxy critique still open |
| 4 | DST claim wrong, may contaminate Oct | CONFIRMED | zoneinfo: London back **2026-10-25** 01:00 UTC, Cairo back **2026-10-30** (offset change detected between 10-29 and 10-30 UTC) → Cairo−London gap = **3h** for sessions Oct 26–29 (gap(10-25)=2h at midnight UTC but UK flipped 01:00 UTC that day; gap(10-26/28/29)=3h). Hard-coded feed labels `09:00/10:00/11:00` (`overlay.py` L32-33) assume 2h ⇒ checkpoints fire ~1h LATE on those sessions = **look-ahead**, inflating October coverage. Spring window: gap=1h on 2026-04-05 (London 03-30, Cairo 04-24) ⇒ fires early = conservative. Doc gap #2 only warns about a FUTURE re-cut; it misses that the mismatch already sat inside the Oct backtest |
| 5 | Selftest proves containment only | CONFIRMED | `overlay.selftest()` L195-196 asserts containment (true by construction); replays TV 15m bars, production uses beta tape |
| 6 | October burnt as holdout | CONFIRMED | constants per invariant list; only clean OOS = forward log from 2026-11-04 |

- Layer 1 code read: `egxfc/train.py`, `egxfc/features.py`, `egxfc/predict.py`
  (+ earlier: root `build_daily_features.py` builder, `egxfc/overlay.py`,
  `_overlay_backtest.py`).

## P1 — DONE (2026-11-03)

Audit written **before any code edit**: `docs/HOURLY_RANGE_AUDIT_20261103.md`.
Probes: `_audit_p1_20261103.py`, `_audit_p1b_20261103.py`, `_audit_p1c_20261103.py`
(all read-only). Key findings (full detail in the audit doc):

- **Layer 1 predictands/leakage clean**: `t_high/t_low` = next-session H/L vs
  today's close; all 41 features are day ≤ t (index day-t known at 16:30 EOD,
  FX shifted +1, S&P/Brent = last completed US session). No same-session leak.
- **Baseline is weak**: skill = 1 − rmse/rmse0, rmse0 predicts 0 return
  (= tomorrow equals today's close). Weaker than persistence / typical
  excursion → P3 must re-score vs stronger baselines before skill survives.
- **Registry**: skill_high>0 21/21, skill_low>0 21/21, skill_close>0 7/21.
  Doc's "16 of 20" close claim **unsourced** (registry says 14/21 failed,
  accuracy table 15/20 failed).
- **Excluded name = ICFC**: no `daily_features_ICFC.parquet` → backtest's
  `except: continue` drops it → **320 = 20 names × 16 sessions**.
- **16 vs 20 sessions**: stale features file (written 11-01) missing Mondays
  10-05/12/19/26; in-memory rebuild gives 20, egxfc matrix already has 20
  clean rows. Published 955 overlays ran on 16 sessions.
- **October anchors in-sample**: persisted bundles `refit=True` fitted on
  train+October (`rows_used` includes Oct) → backtest anchors saw October.
  Recorded step-1 skill is honest; backtest anchors are not.
- **DST quantified** (probe C3, harness reproduces doc's 95/88/85 exactly):
  sessions 10-25..29 have 3h Cairo−London gap → look-ahead inflates
  both-coverage up to **+12pt at 12:00 ck**, ≈**1–3pt pooled** over October.
  Affected 4 backtested sessions as-is 89/96/100 vs corrected 85/84/90.
- **Layer-2 units resolve in code** (`live` = price throughout, products in
  EGP); the doc's prose overloads `live`. Daily-beta-on-intraday-move and
  time-decaying `w` flagged as ablation items (P4), not bugs.
- **Incoherence** (separate high/low fits): 1 row / ~31,000 (ARAB) — harmless
  today.
- **`avg20_volume` uses `tail(40)`** — doc says 20 (doc/code mismatch like
  FRAC_LEFT).
- **beta/volume walk-forward violations** in replay (full-history cov/var and
  tail(40) include future data); limit-hit sessions dropped via `close_pos`
  NaN; SELECT-on-test-month selection bias noted; Ramadan/18-bar grid unguarded.
- **October = burnt** (3 independent burns): constants, selection, refit.
  Clean OOS starts with forward log 2026-11-04 (P6).

## P2 — DONE (2026-11-03)

DST fix + golden freeze. **Golden test written and run BEFORE the edit**
(13 checks PASS against then-current code), then the clock change, then the
full suite (31 checks PASS).

What changed:
- `egxfc/overlay.py`: checkpoint keys are now **Cairo wall-clock**
  (`CKPTS = 11:00/12:00/13:00`, `FRAC_LEFT` re-keyed; `CK_CAIRO` removed as
  redundant). New `feed_cutoff(day, ck)` converts each checkpoint to its
  Europe/London label **per session date** via `zoneinfo` — correct on normal
  (2h), spring-mismatch (1h, Mar 30–Apr 24) and autumn-mismatch (3h,
  Oct 25–29) dates incl. transition days. `day_overlay` filters stock AND
  index bars by that date-specific label; payload/rows gained a `feed` field
  (`cairo` field kept — GUI badge `LIVE 11:00` unchanged).
- `egxfc/live.py`: log line now prints `Cairo {ck} (feed {label})`.
- `_overlay_backtest.py`: its OWN future-bar filter (`fut > ck`) also had to
  switch to `feed_cutoff` (raw Cairo label against feed-stamped bars would
  zero out 13:00 — caught on first re-run, patched).

Tests (`python -m egxfc.overlay --test-clock`, also first stage of
`--selftest`): A = overlay_at golden (pace-clip/exhaustion/slow-belief,
values frozen pre-edit), B = day_overlay golden on winter+summer dates
(old == new code, must be byte-identical), C = active_checkpoint semantics,
D = feed_cutoff DST table (13 dates: winter, summer, all 4 transition days,
both mismatch windows, today), E = day_overlay on both mismatch windows
(autumn: old code read 1h ahead; spring: old code returned None at 11:00 ck).

Corrected October backtest (`python _overlay_backtest.py`, post-fix):

| ck (Cairo) | published (pre-fix) both | corrected both | Δ |
|---|---|---|---|
| 11:00 | 85% (n=320) | 84% (n=320) | −1pt |
| 12:00 | 84% (n=320) | 81% (n=320) | −3pt |
| 13:00 | 87% (n=315) | 85% (n=320) | −2pt |

n at 13:00 rose 315→320 (earlier cutoff ⇒ future bars exist for all
name-sessions). Widths still 118/111/107% of anchor (widening finding
stands). Caveat: still the OLD basis (16 sessions, stale file, in-sample
anchors, ICFC missing) — P3/P4 redo this properly.

## P3 — RUNNING (2026-11-03, background job)

New module `egxfc/rescore_l1.py` (read-only w.r.t. bundles/registry).
Frozen definitions declared in the module docstring BEFORE any number:
ATR(14) units · baselines no_change / persistence / typical-excursion(20) /
ATR-box (k=0.5 a priori) · metrics MAE(ATR), signed bias, coverage per edge
and both, Winkler @ α=0.2, miss tail mean/p95 · common support (one row set
for all methods) · day-level pooling · day-block bootstrap (block 5,
B=1000, seed=20261103) vs best-of-3-baselines per month · family fixed per
name/target from registry (NO per-month selection → no Oct selection bias) ·
guards MIN_TRAIN 300 / MIN_TEST 15 · months 2016-10..2026-10 (complete).

- `python -m egxfc.rescore_l1 --selftest` → **PASS (18 checks)** incl.
  walk-forward leakage guard (train strictly < scored month).
- Smoke (2 names × 3 months) → CSV/MD written; verdicts MIXED/LOSES/BEATS
  on the tiny sample (model ≈ baselines at parity — full table pending).
- Full run launched in background: `python -u -m egxfc.rescore_l1`
  (≈21 names × ~117 months × 2 targets ≈ 4700 fits, est. 30–90 min).
  Outputs: `output/intraday_prediction/rescore_l1.{csv,md}` +
  `rescore_l1_delta.csv` + `rescore_l1_by_name.csv`.
- Known caveat declared in module: last train row's label may fall on the
  test month's first session (same boundary convention as production
  `_slice`, 1 row) — reported, not changed.

## P3 — DONE (2026-11-03): model LOSES to strong baselines

Full run: 21 names × 62 scorable months (2021-09..2026-10), 1253 pooled
day-rows. Outputs: `output/intraday_prediction/rescore_l1.{csv,md}`,
`rescore_l1_delta.csv`, `rescore_l1_by_name.csv`.

**Why 62 months, not 121:** common-support dropna needs all features
finite; `egx100_ret1/ret5` exist only from 2020-05-11/17 (index history
limit) → first valid row 2020-05-17 → with MIN_TRAIN=300 the first
scorable month is 2021-09. Probed and confirmed (not a bug — declared
guard). 1298 name-months skipped, all for train/test < floor.

**Headline (pooled `__ALL__`, day-block bootstrap block=5 B=1000 seed=20261103):**

- model MAE 0.396 ATR vs best baseline **typ_exc 0.381**, delta **+0.016
  [+0.011, +0.021] → LOSES** (ci_lo > 0). atr_box was the runner-up
  baseline in several months.
- Month verdicts: **BEATS 4 · MIXED 32 · LOSES 27** of 63 rows.
  BEATS months are old (2023-03/04/10, 2024-05); **every month from
  2025-10 through 2026-08 is LOSES** (2026-09 MIXED, 2026-10 MIXED at
  −0.002) — the model has been consistently worse than the trivial
  excursion baseline for a year.
- **All 21 names have positive mean delta** (model worse than that
  month's best baseline): best MASR +0.017, worst AXPH +0.080, ADCI
  +0.068. Even the "best" names lose on average.

**Interpretation (for the enhancements doc):** the P1 audit's worry was
real — the registry's "skill > 0 on 21/21" is vs the weak rmse0
(predict-0) baseline. Against persistence / typical-excursion / ATR-box,
the walk-forward model adds **no** band-MAE value and is now consistently
worse. The day-ahead band itself can still be useful (baselines are
decent: ~0.38 ATR MAE), but the ML layer is not what makes it good.
Consequence for P4: Layer-2 anchors come from this model — the eval must
report the anchor quality honestly; any P5 change would be about the
anchor source (e.g. typ_exc as anchor), which is a model change and
therefore **gated/out-of-scope unless the ship bar demands it** (P5 rule:
P4 ship bar first, conformal replacement only if A3 fails).

**Caveats declared:** family fixed from registry (no per-month selection);
same `_slice` boundary convention as production (1 row); best-of-3 baseline
per month is chosen on the same month (optimistic for the baseline side —
makes the model's LOSES verdict conservative, not inflated).

## P4 — CODE DONE, full run queued (2026-11-03)

New module `egxfc/eval_overlay.py` + ablation flags in
`egxfc/overlay.py::overlay_at` (keyword-only `shift / belief_on /
exhaust_on`, defaults = production, golden-guarded) and
`day_overlay(**kw pass-through, `beta_cache=False` support), plus
`rescore_l1` helpers reused (prep/interval_score/block_boot_ci/SUPPORT).

Frozen protocol (docstring, before numbers): months 2025-09..last complete
(15m history) · selection 2025-09..2026-07, LOCKED 2026-08.. · anchors
walk (default, honest pre-month fit) or bundle (backtest-compat, in-sample) ·
causal beta/vol (daily history strictly BEFORE the session day — mirrors
live) · zoneinfo feed cutoffs · forward extremes only · three-way split
(already-in fraction reported, never counted as coverage) · primary pooled
Winkler @80 ATR-normalised day-level + day-block bootstrap · ablation
B0 frozen anchor / A0 contain-only / A1 +beta / A2 +pace-belief /
A3 +exhaustion (constants FROZEN — terms only) · ship bar on LOCKED:
A3 must beat BOTH A0 and B0 (ci_hi<0) or P5 evaluates conformal replacement.

- `python -m egxfc.eval_overlay --selftest` → **PASS (8 checks)**.
- Smoke (2 names × 2026-10, walk anchors) → full path works. Early reads
  (NOT final): already-in 68%/57% at ck1 (tautology quantified) ·
  A3 beats A0 (−0.128) and B0 (−0.416) on Winkler · beta term helps most
  (−0.111), belief small (−0.017), **exhaustion delta exactly 0.000 —
  term may never fire naturally** (now instrumented: `n_exh` column) ·
  A1-A3 trade new-extreme coverage (0.56–0.87 vs A0's 1.00) for narrower
  width — exactly the honest trade the metric is meant to expose.
- Full run **launched 2026-11-03 after P3 finished** (bg `python -u -m
  egxfc.eval_overlay`, walk anchors, 21 names × 2025-09..2026-10).

## P4 — DONE (2026-11-03): ship bar PASSED

Full run: 21 names × 14 months (2025-09..2026-10), walk anchors, all 14
months scored per name. Outputs: `eval_overlay_walk.md`,
`eval_overlay_{split,delta,month}_walk.csv`, `eval_overlay_rows_walk.parquet`.

- **Tautology quantified**: already-in 77% (high) / 60% (low) pooled —
  most of the session extreme is printed by the checkpoint, which is why
  the old whole-session coverage (84–87%) was inflated.
- **Primary (pooled Winkler @80, ATR, day-level, day-block bootstrap):**
  A3 1.357 vs A0 1.496 vs B0 1.561. Deltas: A3−A0 −0.140
  [−0.150, −0.131] BEATS · A3−B0 −0.205 [−0.224, −0.186] BEATS.
- **LOCKED (2026-08..10, the ship-bar scope):**
  A3−A0 **−0.106 [−0.123, −0.088] BEATS** ·
  A3−B0 **−0.220 [−0.271, −0.178] BEATS** → **SHIP as-is**.
- **Ablation reads**: beta term carries it (A1−A0 −0.129 pooled,
  −0.098 locked, BEATS); pace/belief small but significant (−0.012);
  **exhaustion neutral** (A3−A2 MIXED, Δ ≈ 0) — and it DOES fire
  (n_exh 1012 pooled / 214 locked; the earlier smoke's 0 was a
  2-name-1-month artifact). A3 wins at every checkpoint; already-in
  rises through the day (67/78/87% at 11/12/13).
- Honest trade exposed: A1–A3 new-extreme coverage 0.75/0.65 vs A0's
  0.87/0.83, bought with width (A3 1.006× anchor vs A0 1.201×) — Winkler
  says the trade is worth it.

## P5 — DONE (2026-11-03): no change shipped

Ship bar passed ⇒ production overlay stands **as-is**; no conformal
replacement evaluated (that path was gated on the bar failing). The only
candidate surfaced — dropping the neutral exhaustion term — deliberately
NOT done: A3 ≈ A2 is MIXED (no measurable gain); "terms may be dropped"
allows it, but removing it would break comparability with the
frozen-prototype golden tests for zero benefit. Recorded as open item
(enhancements doc §5.2).

## P6 — DONE (2026-11-03)

Forward-log + crash-safe I/O, all in `egxfc/overlay.py` (no new module):

- **`range_outcomes.jsonl`** — append-only checkpoint log. `log_range_rows(payload)`
  writes one line per (date, ck, sym), **idempotent**: replaying a cached payload
  or a mid-session process restart appends 0 duplicates. Best-effort — returns
  rows appended, never raises. Single-writer assumption = the live loop.
  `load_range_log()` reads it back to a DataFrame; `join_forward(df, bars_by_sym)`
  does the lazy next-EOD join of realised forward extremes (feed-label cutoff,
  three-way split flags, coverage) — **no `eod.py` changes**; NaN while the
  session has no forward bars yet.
- **Atomic `live_range.json`** — `_atomic_write_text()` = temp file +
  `os.replace`, so GUI/consumers never observe a partial JSON.
  `compute_all` now writes atomically + logs (both best-effort, never raises).
- **Tests**: `python -m egxfc.overlay --test-io` → **15 PASS** (idempotent
  append: first=2 / replay=0 / partial=1 / new-ck=1; log keys unique; atomic
  content + no `.tmp` left; `join_forward` hand-computed extremes on synthetic
  bars incl. wide-band coverage, ck12 vs ck11 cutoffs, no-bars → NaN).
  Full `--selftest` → PASS (31 clock + 15 io + MBEG/MASR replay).
  Selftest replays `day_overlay` directly and io tests use a temp dir —
  neither touches the production log.
- **Consumers**: live loop already fires `compute_all` once per ck (guarded by
  `fired`) → log fires there; restart-safe via idempotency.

---

## Change log (append after every edit)

- 2026-11-03 (P1, doc-only): created `docs/HOURLY_RANGE_AUDIT_20261103.md`
  (pre-edit audit). No source files touched.
- 2026-11-03 (P1, probes, read-only): created
  `scripts/intraday_prediction/_audit_p1_20261103.py`,
  `_audit_p1b_20261103.py`, `_audit_p1c_20261103.py` (one-off probes; they
  print, never write data files).
- 2026-11-03 (worklog): created/updated this file.
- 2026-11-03 (P2, `egxfc/overlay.py`): canonical Cairo ck keys + `feed_cutoff`
  via zoneinfo; `day_overlay` date-specific bar filters; `feed` field in
  payload/rows; `CK_CAIRO` removed; module docstring updated; added
  `clock_selftest()` + `--test-clock` flag; `selftest()` now runs clock
  tests first.
- 2026-11-03 (P2, `egxfc/live.py`): checkpoint log line shows Cairo + feed.
- 2026-11-03 (P2, `_overlay_backtest.py`): print uses Cairo ck; future-bar
  filter switched to `O.feed_cutoff(day, ck)`.
- 2026-11-03 (P2, probe): created `_p2_golden_gen_20261103.py` (one-off
  golden generator, run pre-edit).
- 2026-11-03 (P3): created `egxfc/rescore_l1.py` (frozen-definition
  walk-forward rescore; selftest + smoke pass; full run in background).
- 2026-11-03 (P4, `egxfc/overlay.py`): `overlay_at` gained keyword-only
  ablation flags `shift / belief_on / exhaust_on` (defaults = production,
  golden-tested); `day_overlay` passes them through + `beta_cache=False`
  option; `beta_vs_home(..., cache=...)`.
- 2026-11-03 (P4): created `egxfc/eval_overlay.py` (tautology-free forward
  eval + ablation; selftest + smoke pass).
- 2026-11-03 (P3 support): `rescore_l1.py` `SUPPORT` list hoisted to module
  level (shared with eval_overlay); `load_bframe` lives in eval_overlay
  (rescore untouched mid-run).
- 2026-11-03 (P6, `egxfc/overlay.py`): `_atomic_write_text()` (temp+replace),
  `RANGE_LOG = range_outcomes.jsonl`, `log_range_rows()` (idempotent
  append-only), `load_range_log()`, `join_forward()` (lazy next-EOD join of
  forward extremes, no eod.py changes); `compute_all` writes atomically +
  logs (best-effort); `io_selftest()` (15 checks, temp dir) wired into
  `selftest()` + new `--test-io` flag.
- 2026-11-03 (P7 partial, `docs/HOURLY_RANGE_OVERLAY.md`): corrected
  frac-left 72/50/28 → 78/56/33 (14/18, 10/18, 6/18); "16 of 20" → sourced
  registry numbers (close 7/21, high/low 21/21) + skill definition;
  avg20_volume → notes tail(40); checkpoints section → per-date
  `feed_cutoff`/zoneinfo; §3a selftest → 46 checks + replay; §3b →
  corrected post-fix numbers (84/81/85) with old-basis caveat + P4
  supersedes; §4 gap #2 DST → FIXED (P2); gap #7 → DONE (P6); gap #3
  coverage range updated; `live_range.json` shape → +`feed`, atomic write.
- 2026-11-03 (P7 partial, `docs/HOURLY_RANGE_AUDIT_20261103.md`): §5 note —
  probe outputs are pre-fix; p1c as-is vs corrected columns explained.
- 2026-11-03 (P3 complete): full rescore run finished — verdict LOSES
  pooled (+0.016 [+0.011, +0.021]); 62 scorable months 2021-09..2026-10
  (egx100_ret* features start 2020-05 → MIN_TRAIN floors at 2021-09);
  results in `output/intraday_prediction/rescore_l1.*`; worklog P3 section
  written. P4 full run launched immediately after (CPU free).
- 2026-11-03 (P7): created `docs/HOURLY_RANGE_OVERLAY_ENHANCEMENTS_20261104.md`
  — full draft with P0–P3/P6 results; §1.4 P4 result and §1.5 P5 marked
  PENDING until their runs finish.
- 2026-11-03 (P4 complete): full eval run — ship bar PASSED on LOCKED
  (A3−A0 −0.106 [−0.123,−0.088], A3−B0 −0.220 [−0.271,−0.178], both
  BEATS); already-in 77/60%; exhaustion fires (n_exh 1012) but is neutral;
  outputs `output/intraday_prediction/eval_overlay_walk.*`.
- 2026-11-03 (P5): no change shipped (bar passed; conformal path not
  triggered). Exhaustion-drop candidate documented, deliberately not done.
- 2026-11-03 (P7 final): enhancements doc §1.4/§1.5 filled with P4/P5
  results, status FINAL; §3 verification table completed; §5 open items
  rewritten (forward-log watch, exhaustion note, anchor-source gate,
  ICFC features gap). Worklog phase table all-DONE.
