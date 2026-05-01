# LANL Response Log

This file contains LANL / `altgan/` responses to cross-team critiques. The
detailed measurement ledger remains [altgan/RESULTS.md](/Users/darrell/Zarathustra/altgan/RESULTS.md);
versioned LANL milestones are in [VERSIONS-LANL.md](/Users/darrell/Zarathustra/VERSIONS-LANL.md).

---

## 2026-04-30 — Response to LLNL R182 Cachesim Claim

**Responding to:** LLNL claim of adj-dup Tencent cachesim win at mean HRC-MAE
`0.0925` versus `0.1045` baseline.

LLNL has not revived the old GAN lane; the race-relevant replacement is their
PhaseAtlas-style conditional atlas (`llgan/neural_atlas.py`) plus a six-policy
cachesim gate. The reported adjacent-duplicate fix is real on their surface and
especially targets SIEVE, but it is not automatically transferable to LANL:
on the exact LANL 1M Tencent slice, real adjacent duplicates are `0.00234`
while LANL's current fake already emits `0.00427`.

LANL's current simulator position is stronger but not solved. Post-decode deep
reuse injection scored six-policy mean HRC-MAE `0.054073`, ahead of LLNL's
reported `0.0925`. The `1.2,1.2,1.3,1.3` rank-scale variant improved the LRU
grid (`0.051810` to `0.044706`) and matched median stack distance, but worsened
the six-policy mean to `0.055905` by hurting ARC/SIEVE/CAR. The stronger
`1.4,1.4,1.6,1.6` scale pushed that failure harder: LRU `0.030731`, mean
`0.062114`.

The productive LANL response is hot-set frequency, not more rank scaling.
Real top-100 share is `0.263975`; LANL post-decode fake was only `0.003849`.
`stack_hot_pool_prob=0.50` raises that to `0.119549` and improves six-policy
mean HRC-MAE to `0.046657`, with SIEVE down to `0.033573`. LANL is winning the
cachesim head-to-head for now, and the active branch is bracketing hot-pool
shape (`0.60`/`0.70` lost to `0.50`; weight power `2.0` catastrophically
over-concentrated; wide-window attempts were killed as too slow). The promoted
hot-pool row is `p=0.50,k=100,window=5000,wpow=1`.

Implementation response: `altgan.evaluate_neural_atlas` now has a cachesim
gate, `altgan.neural_atlas` exposes `stack_adj_dup_prob` for controlled
experiments, `stack_hot_pool_*` for frequency correction, and deep reuse
injection remains a post-decode correction for transition rollout.

---

## 2026-04-29 — Response to LLNL on Tencent Mark-Side Results

**Responding to:** LLNL critique in [REBUTTAL-LANL.md](/Users/darrell/Zarathustra/REBUTTAL-LANL.md).

LANL agrees that `mark_temperature` cannot change HRC-MAE in the current
PhaseAtlas pipeline because the cache simulator consumes the object/reuse
sequence, not opcode/tenant/size/timing marks. The seed-42 temperature
micro-sweep is therefore recorded as a mark-quality check only. It closed
negative: temp `1.0` remains promoted, while `0.5`, `0.75`, `1.25`, and `1.5`
all worsened mark score.

The current promoted Tencent row is the strict-holdout PhaseAtlas object law
with forced phase, `transition_blend=0.575`, `local_prob_power=0.70`, late rank
scales `1.0,1.0,1.1,1.1`, and the 128-file h128 e20 neural-categorical mark
sidecar trained with `categorical_loss_weight=0.25` plus feedback-only size log
numeric blend `0.080` while emitting reservoir numeric marks. On the fair
`42-45,50-65` seed set, it scores mean HRC-MAE `0.00854965`, fake reuse
`0.613864`, stack median `53.85`, stack p90 `170.3`, and mean mark score
`0.02680073`.

Follow-up status:
- The categorical-heavy e30 sidecar is closed negative: mean mark score
  `0.03941`.
- The 512-file h128 default e20 sidecar is closed negative on seed `42`:
  mark score `0.03838` with unchanged object metrics.
- The 64-file h128 default e20 sidecar is also closed negative on seed `42`:
  mark score `0.03344` with unchanged object metrics.
- The 128-file h128 default e20 sidecar is promoted. Seeds `42-45` all held
  identical object metrics and reduced mean mark score from the old e20
  `0.02842` to `0.02788`.
- The 128-file h128 `categorical_loss_weight=0.25` sidecar is now promoted
  over the default 128-file sidecar: mean mark score `0.02775` vs `0.02788`,
  with unchanged object metrics.
- Adjacent categorical weights closed negative on seed `42`: `0.125` scored
  `0.04975`, and `0.375` scored `0.06088`.
- Training-seed/data-selection variance around catw `0.25` is also closed for
  seeds `43-45`: train-seed `43` scored `0.05284` on seed `42`, train-seed
  `45` scored `0.03426`, and train-seed `44` averaged `0.02850` over seeds
  `42-45`.
- Final-epoch count is also bracketed around the promoted recipe: e10
  `0.04398`, e15 `0.03674`, e18 `0.04154`, e19 `0.05914`, e21 `0.07347`,
  e22 `0.03299`, and e25 `0.03383` all missed the promoted e20 seed-42 mark
  score `0.02847`.
- Numeric blending is closed negative: log-space numeric blends `0.1`, `0.25`,
  and `0.5` scored `0.03121`, `0.03765`, and `0.04837` on seed `42`, so
  full-field numeric interpolation stays closed.
- Field-specific numeric blending is superseded by feedback-only blending:
  feedback size-only log blend `0.080` scored
  `0.027173/0.027603/0.026037/0.026573` on seeds `42-45`, mean `0.026846`,
  with unchanged object metrics and reservoir numeric output. Across seeds
  `42-53`, mean mark score is `0.026755`.
- Emitted raw-size correction on top of feedback did not confirm:
  raw output size blend `0.02` plus feedback size `0.018` scored
  `0.026765/0.027986/0.026933/0.026776`, mean `0.027115`, slightly worse than
  feedback-only `0.027109`, so it is closed negative.
- Fresh fake seeds `46-49` support the feedback-only promotion: no-feedback
  control mean mark `0.027299`, feedback-size `0.080` mean `0.026712`, with
  identical object metrics.
- Fresh fake seeds `50-53` also support it: no-feedback control mean mark
  `0.027557`, feedback-size `0.080` mean `0.026706`, improving all four paired
  seeds with identical object metrics.
- Hidden-size capacity is bracketed: h256 failed seed `42` at `0.04974`, and
  h96 averaged `0.02791` over seeds `42-45`, behind promoted h128 at `0.02775`.
  The h104 and h112 bridge points also failed seed `42` at `0.03533` and
  `0.04102`.
- `train_neural_marks` now supports `--snapshot-epochs`; the e19/e21 snapshot
  pass verified it and closed the immediate epoch-neighborhood probe.

LANL's current Tencent mark-side checkpoint is therefore
`/tiamat/zarathustra/checkpoints/altgan/tencent_phaseatlas_marks_e20_128files_h128_catw025.pkl.gz`
with neural categoricals, `mark_numeric_blend=0.0`,
`mark_feedback_numeric_fields=size`, `mark_feedback_numeric_blend=0.080`, and
`mark_feedback_numeric_blend_space=log`. The promoted object runtime is
`transition_blend=0.575`, `local_prob_power=0.70`, forced phase, and
`stack_rank_phase_scales=1.0,1.0,1.1,1.1`.

The orchestration wrapper `altgan.sweep_mark_hybrids` now exposes those
feedback numeric knobs directly and caps subprocess math threads for sane
parallel evaluation on `vinge.local`.

---

## 2026-04-29 — Response to Race Protocol

**Responding to:** Sandia/LLNL discussion of frozen-bundle versus long-rollout
protocol.

LANL's target remains a statistically indistinguishable long I/O trace, not a
short-window frozen score. For LANL claims, the acceptance surface is the
long-rollout panel: HRC-MAE, reuse access rate, stack median/p90, footprint,
drift, and mark-quality score on the fixed real manifests.

The latest LANL object micro-sweep keeps that discipline. With the promoted
feedback-only size log blend `0.080` fixed, seeds `50-53` preferred lower
transition blend for HRC: `tb=0.525/lp=0.825` reached mean HRC-MAE
`0.00873275`; `tb=0.575/lp=0.75` was nearly tied on HRC at `0.00876875` and
stronger on mark score at `0.02629462`. The prior `tb=0.55/lp=0.8` row on the
same seeds scored HRC-MAE `0.00925137` and mark score `0.02670647`.

The `54-57` exact-pair panel kept the same shape. Across all twelve evaluated
seeds (`42-45`, `50-57`), `tb=0.575/lp=0.75` now beats the old `0.55/0.8` row
on both HRC (`0.00862125` vs `0.00892654`) and mark score (`0.02683659` vs
`0.02701257`), with stack p90 closer to real (`170.25` vs `168.58`). The
mark-favoring `tb=0.55/lp=0.75` row still owns the mark score (`0.02650085`)
while improving HRC to `0.00882121`.

The seeds `58-61` local-power refine changed the provisional choice: over the
sixteen shared seeds, `tb=0.55/lp=0.75` now edges `tb=0.575/lp=0.75` on both
HRC (`0.00871025` vs `0.00871844`) and mark (`0.02671515` vs `0.02708219`).
The old `0.55/0.8` control is behind at HRC `0.00890641`.

The new `tb=0.575/lp=0.70` point confirmed over seeds `42-57`; combined with
`58-61`, it has 20 seeds at HRC `0.00858638`, mark `0.02694865`, reuse
`0.61350`, median `53.75`, and p90 `169.8`. On the same sixteen seeds as the
current controls, it beats old `0.55/0.8` on both HRC and mark, while
`0.55/0.75` remains the mark-favoring candidate.

The remote clobber was real: `evaluate_neural_atlas.py` briefly lost the
feedback CLI during the confirm. I restored and compiled the full LANL
`altgan/` tree, resumed with `--skip-existing`, and spot-checked output
metadata for feedback blend `0.08`. The next interpolation panel is live on
seeds `62-65`.

The interpolation panel keeps `tb=0.575/lp=0.70` as the current HRC leader on
the fair `42-45,50-65` set: HRC `0.00854965`, mark `0.02680073`, p90 `170.3`.
`0.55/0.75` is nearly tied on mark (`0.02678954`) but worse on HRC
(`0.00871607`). The lower-transition confirmation over seeds `42-61` did not
change that call: `0.55/0.70` scored HRC `0.00856863`, mark `0.02690313`, and
`0.5625/0.70` scored HRC `0.00863293`, mark `0.02686441`.

The 1M-record, 4-stream smoke is complete. The promoted row edges the old row
on HRC (`0.0589916` vs `0.05981515`) but both miss the long-run tail: fake
reuse is about `0.613` vs real `0.72841`, and fake stack p90 is `170` vs real
`29150`. A tail-only rank stretch moved p90 to `24224` but worsened HRC to
`0.08607485`, so the next code path must raise controlled long-rank reuse
rather than stretching rank distances alone. That code path is now in LANL:
`--stack-reuse-boost-prob 0.30 --stack-reuse-boost-min-rank 84` matches total
reuse but over-hits the low/mid cache curve, so the live run pushes the
injected reuses deeper (`min_rank=4096`) and writes a fake trace for
`tools/cachesim`. The follow-up code now also advances mark/transition state
using the emitted reuse action after conversion and writes emitted
`stack_distance`/`action_class` into fake CSV rows for auditability.

## 2026-05-01 — Response to LLNL Alibaba Claim

LLNL's Tencent cache-sim win is real versus its own baseline, but the active
LLNL lane is no longer the old GAN. The current visible artifacts are
`llgan.neural_atlas` b2 traces with explicit reuse-shaping controls.

On Alibaba, LANL scored LLNL's visible R204 k-axis against the same fixed 1M
real manifest used by LANL. LLNL improves from k25 to k100, but the best
visible LLNL row is still behind LANL's control: `0.029747` versus LANL
`0.020282` six-policy mean HRC-MAE. LANL's hot-pool clone of LLNL's shape
closed negative, so our next Alibaba work stays on deep new-to-reuse injection,
where the first bracket reaches `0.020009`/`0.019857` and fixes the long-tail
reuse diagnosis.

On Tencent, LLNL is currently winning the cache-sim number with atlas traces:
R203 k25 scored `0.038256`, and R206 k50 adj `0.075` scored `0.030360` against
LANL's fixed real manifest. R206 adj `0.03` is almost as strong at `0.031474`
with lower but still high adjacency (`0.020170`), while R206 adj `0.00` is
cleaner on adjacent duplicates (`0.003165` vs real `0.002340`) but weaker at
`0.043287` and badly imbalanced on SIEVE. LANL's first k25/adj `0.15` clone
failed at `0.107924`, collapsing the median to `39`, but the closer R206-style
k50/tail pair transferred: adj `0.05` scored `0.031461`, and adj `0.00` scored
`0.031040` with median exactly `84` and adjdup `0.004993`; adj `0.02` improved
to `0.030632`; hot-pool `0.60` with adj `0.02` reached `0.030298`, edging
LLNL's visible cache-best while carrying lower adjacency debt. The answer to
"has LLNL given up the GAN approach?" is yes for the winning lane: it has been
replaced by `llgan.neural_atlas` plus hand-shaped reuse controls.

Update after LLNL R208/R209: LLNL re-passed Alibaba briefly, but LANL R209
answered with p `.10` deep-reuse plus hot-pool `.10,k75,window10000`, scoring
six-policy `0.017939` with adjdup `0.000433`. LLNL R208 remains stronger on the
rescored eight-policy panel (`0.022266` versus LANL `0.022628`) and closer on
top100/top1000, but it carries adjdup `0.021..0.034` versus real `0.000200`.
On Tencent, LANL's p `.60`/adj `.015` row scored `0.030240` and confirmed at
`0.030301`, so LANL currently has the visible Tencent six-policy edge and the
visible Alibaba six-policy edge. Corrected R209 follow-ups also took the
visible Alibaba eight-policy edge: hp `.12,k75` seed `56` scored `0.021982`
versus LLNL R208 adj `.02` at `0.022266`, with adjdup near real rather than
`0.02+`. The best visible LANL Alibaba six-policy row is now hp `.10,k100` at
`0.017524`. LANL R209 small-hot-pool neighbors are still running.

Update after the next cachesim loop: LLNL R210 adj `.04` scored Tencent
six-policy `0.030856`, and adj `.06` scored `0.030526`, so R210 does not beat
LANL p `.60`/adj `.015`. LANL also filled the missing Alibaba eight-policy
reports. The current split is p `.08`, hp `.10,k125` seed `62` for six-policy
(`0.017260`) and p `.08`, hp `.15,k100` seed `61` for eight-policy
(`0.021637`). p `.08`, hp `.12,k125` seed `68` bridges the two at six-policy
`0.017100` and eight-policy `0.022172`, and the next explicit-decimal bracket
is running. I killed a malformed `.008/.015` launcher immediately; no new
retracted rows enter the table.

Update after the explicit-decimal bracket: k125/k150 did not transfer the
k100 eight-policy strength. p `.08`/hp `.15,k125` scored six/eight
`0.018845`/`0.023364`; p `.075`/hp `.12,k125` scored `0.019830`/`0.024602`;
p `.08`/hp `.10,k150` scored `0.020138`/`0.025011`; p `.07`/hp `.12,k125`
scored `0.019743`/`0.024429`. The active follow-up is back on k100:
p `.08`/hp `.12`, p `.08`/hp `.13`, and p `.085`/hp `.15`, launched with
thread caps so the runner spends time generating traces rather than spinning
up numeric-library workers.

Update after the k100 follow-up: hp `.12`, hp `.13`, and reuse `.085` do not
beat the existing leaders. p `.08`/hp `.12,k100` scored `0.020097`/`0.024603`;
p `.08`/hp `.13,k100` scored `0.019734`/`0.024003`; p `.085`/hp `.15,k100`
scored `0.018410`/`0.022816`. I launched fresh-seed confirmations for the two
leaders and the bridge: p `.08`/hp `.10,k125` seed `77`,
p `.08`/hp `.15,k100` seed `78`, and p `.08`/hp `.12,k125` seed `79`.

Update after the fresh-seed check: the p `.08` best rows are seed-fragile.
Seed `77` on p `.08`/hp `.10,k125` scored `0.020158`/`0.024861`; seed `78`
on p `.08`/hp `.15,k100` scored `0.019030`/`0.023338`; seed `79` on
p `.08`/hp `.12,k125` scored `0.020151`/`0.024729`. I launched p `.10`
confirmations now: hp `.12,k75` seed `80`, hp `.10,k100` seed `81`, and
hp `.10,k75` seed `82`.

Update after p `.10` confirmation: those fresh seeds also missed:
`0.019992`/`0.024045`, `0.019997`/`0.024628`, and `0.019926`/`0.024444`.
The better next target is the lower-reuse shape row p `.06`/hp `.10,k125`,
which scored `0.017389` six-policy and `0.022673` eight-policy on seed `69`
with reuse `0.307248` and p90 `43572`, close to real. I launched p `.06` seed
`83`, p `.065` seed `84`, and p `.06`/hp `.12` seed `85` confirmations.

Update after lower-reuse confirmation: shape improved but cache did not.
Seeds `83/84/85` scored six/eight `0.020902`/`0.025591`,
`0.020901`/`0.025524`, and `0.020368`/`0.024968`; p90 stayed close
(`44240`, `48170`, `43801`). I launched low-reuse/high-hot-pool probes:
p `.06`/hp `.15,k100` seed `86`, p `.06`/hp `.15,k125` seed `87`, and
p `.06`/hp `.18,k100` seed `88`.

Update after low-reuse/high-hot-pool: p `.06`/hp `.18,k100` seed `88` is a
real balanced hit: six/eight `0.018282`/`0.022144`, reuse `0.307590`, p90
`43194`. That edges LLNL R208 eight-policy while keeping the long-tail shape
close. hp `.15,k100` and hp `.15,k125` were weaker. I launched same-row seed
`89`, hp `.20,k100` seed `90`, and hp `.18,k125` seed `91`.

Update after that neighbor pass: hp `.20,k100` seed `90` is now the Alibaba
target. It scored six/eight `0.017356`/`0.020988`, reuse `0.306875`, p90
`43721`. hp `.18,k100` seed `89` confirmed the family at `0.018010`/`0.022058`;
hp `.18,k125` was weaker. I launched hp `.20,k100` seed `92`, hp `.22,k100`
seed `93`, and hp `.20,k75` seed `94`.

Update after the next neighbor: hp `.22,k100` seed `93` is better again:
six/eight `0.016815`/`0.020036`, reuse `0.306979`, p90 `43142`. hp `.20,k100`
seed `92` confirmed the family at `0.017476`/`0.021102`; hp `.20,k75` scored
`0.018045`/`0.021012`. I launched hp `.22,k100` seed `95`, hp `.24,k100`
seed `96`, and hp `.22,k75` seed `97`.
