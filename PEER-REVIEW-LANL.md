# LANL Self-Review Ledger

LANL / `altgan` self-review notes. This file was reconstructed after a local
root-doc clobber replaced the prior long ledger with a placeholder; the detailed
race ledger now lives in [altgan/RESULTS.md](/Users/darrell/Zarathustra/altgan/RESULTS.md)
and [VERSIONS-LANL.md](/Users/darrell/Zarathustra/VERSIONS-LANL.md).

---

## Round 1 (2026-04-30) — Doc And Code Clobber Is A Live Race Risk

### Finding

Multiple LANL-owned files were replaced by placeholders or older copies during
the current loop: `VERSIONS-LANL.md`, `RESPONSE-LANL.md`,
`PEER-REVIEW-LANL.md`, `PEER-REVIEW-LLNL.md`, and local `altgan/*.py` files.
This can silently invalidate runs if the remote copy is overwritten with stale
local code.

### Required Control

Before launching or promoting runs, verify the local and remote code both show:
`ThreadPoolExecutor`, `--jobs`, `set_conditions`, `def observe`,
`--numeric-loss-weight`, `--categorical-loss-weight`, and
`--snapshot-epochs`. Do not sync placeholders over durable remote docs.

---

## Round 2 (2026-04-30) — Current Tencent Promotion Bar

### Finding

The promoted LANL Tencent object process remains strict-holdout PhaseAtlas with
forced phase, `transition_blend=0.55`, `local_prob_power=0.8`, and late rank
scales `1.0,1.0,1.1,1.1`. The current mark checkpoint is
`/tiamat/zarathustra/checkpoints/altgan/tencent_phaseatlas_marks_e20_128files_h128_catw025.pkl.gz`,
with reservoir numeric marks and neural categoricals as the last confirmed
four-seed row.

### Required Control

Any new runtime setting must preserve HRC/reuse/stack metrics and beat the
four-seed mark score, not only a single seed. Seed-42 wins require seeds 43-45
confirmation before promotion.

---

## Round 3 (2026-04-30) — Restored-Code Numeric Blend Confirmation Complete

### Finding

The code-clobber control was necessary. After restoring the emitted-history
rollout contract, size-only numeric work split into two branches: emitted
size-only log blend `0.018` confirmed at mean mark score `0.02725443`, and
feedback-only size log blend `0.018` improved the four-seed mean further to
`0.02710896`.

### Required Control

Use the restored emitted-history rollout contract (`mark_runtime.observe`) for
all numeric-feedback evaluations. The current LANL promotion is feedback-only
size log blend `0.018`, not emitted numeric blending.

---

## Round 4 (2026-04-30) — Raw Output Size Correction Closed Negative

### Finding

Adding emitted raw-size blend `0.02` on top of feedback-size `0.018` produced a
tiny seed-42 improvement but worsened the four-seed mean: `0.02711492` versus
feedback-only `0.02710896`.

### Required Control

Keep feedback-only size `0.018` as the promoted runtime. Do not promote
single-seed emitted-output corrections unless they beat the paired four-seed
panel.

---

## Round 5 (2026-04-30) — Fresh-Seed Feedback Robustness Supports Promotion

### Finding

Seeds `46-49` confirm the feedback-only size `0.018` direction outside the
original confirmation seeds. The no-feedback control averaged `0.02729909`;
feedback averaged `0.02722472`, with unchanged HRC/reuse/stack metrics. One
seed moved slightly negative, but the paired four-seed mean improved.

### Required Control

Use the new `sweep_mark_hybrids` feedback-grid path for future numeric sweeps,
and keep subprocess math threads capped during parallel evals to avoid
oversubscribing `vinge.local`.

---

## Round 6 (2026-04-30) — Feedback Strength Promotion Moved To 0.080

### Finding

The initial feedback-size `0.018` promotion was conservative. A wider paired
sweep found feedback-size `0.080` better on both the original `42-45` panel and
fresh `46-49` seeds, with eight-seed mean mark score `0.02677934` and unchanged
object metrics. Higher checks `0.100` and `0.120` missed on seeds `42-45`, so
the high side is at least locally bracketed.

### Required Control

Promote `mark_feedback_numeric_blend=0.080` only for feedback state, not emitted
numeric marks. Seeds `50-53` also improved as a paired panel, bringing the
promotion to 12 fake seeds, but continue to require paired panels because
individual seeds move non-monotonically.

---

## Round 7 (2026-04-30) — Object Micro-Sweep Needs More Confirmation Before Promotion

### Finding

The seeds `50-53` object micro-sweep under the promoted feedback-size `0.080`
runtime found a real HRC opportunity but not yet a promotion. `tb=0.525`,
`lp=0.825` won HRC at `0.00873275`, while `tb=0.575`, `lp=0.75` was almost tied
on HRC at `0.00876875` and better on mean mark score (`0.02629462`). The old
`tb=0.55`, `lp=0.8` row scored HRC `0.00925137` and mark `0.02670647` on the
same seeds.

### Required Control

Do not promote a new object cell from seeds `50-53` alone. The seeds `42-45`
confirmation grid must include the old row and both candidate rows before LANL
changes the advertised object recipe.

### Follow-up

The seeds `42-45` confirmation did not give a single clean winner. Combined
with seeds `50-53`, `tb=0.575/lp=0.75` has the best HRC (`0.00862656`) and
better stack p90, but `tb=0.55/lp=0.75` has the better mark score
(`0.02651714`) while still improving HRC over `tb=0.55/lp=0.8`. A seeds
`54-57` exact-pair panel is running before any object promotion.

### Follow-up 2

Seeds `54-57` confirmed the split. Across all twelve evaluated seeds,
`tb=0.575/lp=0.75` is now the HRC-leading object candidate and beats the old
row on mark as well (`0.02683659` vs `0.02701257`). `tb=0.55/lp=0.75` remains
the mark-leading compromise at `0.02650085`. A seeds `58-61` local-power refine
around `tb=0.575` is running before changing the advertised object recipe.

### Follow-up 3

The seeds `58-61` local-power refine made `tb=0.55/lp=0.75` the best shared
sixteen-seed row among fully measured candidates: HRC `0.00871025`, mark
`0.02671515`, versus old `tb=0.55/lp=0.8` at HRC `0.00890641`, mark
`0.02704201`. `tb=0.575/lp=0.70` is promising on four seeds only, so it is now
being confirmed on seeds `42-57` before promotion.

### Follow-up 4

`tb=0.575/lp=0.70` confirmed as the HRC-leading object candidate: 20-seed mean
HRC `0.00858638`, mark `0.02694865`, reuse `0.61350`, stack p90 `169.8`. On
the same sixteen seeds as `0.55/0.75` and old `0.55/0.8`, it beats the old row
on HRC and mark, but `0.55/0.75` remains the mark leader. A remote clobber of
`evaluate_neural_atlas.py` was repaired by restoring the full LANL `altgan/`
tree and resuming with `--skip-existing`; output metadata confirms feedback
blend `0.08`.

### Follow-up 5

The seeds `62-65` interpolation panel kept `tb=0.575/lp=0.70` as the current
HRC-leading shared candidate: fair-set HRC `0.00854965`, mark `0.02680073`.
`tb=0.55/lp=0.75` remains nearly tied on mark (`0.02678954`) but worse on HRC
(`0.00871607`). The only plausible new challenger was `tb=0.5625/lp=0.70`,
so it was confirmed over seeds `42-61` alongside `0.55/0.70`.

### Follow-up 6

The lower-transition confirmation did not dislodge `tb=0.575/lp=0.70`.
Over seeds `42-61`, `tb=0.55/lp=0.70` scored HRC `0.00856863`, mark
`0.02690313`, reuse `0.61286`, p90 `168.35`; `tb=0.5625/lp=0.70` scored HRC
`0.00863293`, mark `0.02686441`, reuse `0.61362`, p90 `169.5`. On the fair
principal set `42-45,50-65`, `tb=0.575/lp=0.70` remains the HRC leader at
`0.00854965`; `0.55/0.75` remains the mark hair-split but gives back HRC.

### Follow-up 7

Promote `tb=0.575/lp=0.70` for the 1M smoke, but compare it directly to the
old `0.55/0.8` row on the same deterministic real manifest before advertising
a long-trace claim. That control is now complete: promoted scores HRC
`0.0589916`, old scores HRC `0.05981515`, so the promoted row remains slightly
better. Both fail the 1M tail similarly: real reuse `0.72841` vs fake
`0.61286/0.61385`, and real p90 `29150` vs fake p90 `170/169`.

### Follow-up 8

The new tail-rank controls are functional but not sufficient. With
`stack_rank_tail_pivot=84` and `stack_rank_tail_scale=340`, fake p90 moved to
`24224`, close to real `29150`, while the median stayed `54`; however HRC
worsened to `0.08607485` because total fake reuse remained `0.61286`. The next
control should add a measured new-to-reuse conversion path aimed at long ranks.
