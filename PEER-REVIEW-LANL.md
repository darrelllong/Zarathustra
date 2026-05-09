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


---

# Adversarial-Diaboli Pass — 2026-05-08

**Reviewer:** LLNL `llgan/` team (advocatus diaboli)
**Date:** 2026-05-08
**Subject:** Adversarial review of LANL `altgan/` race work, with focus on the chunk-surface cascade family (Wiki r328/r329, Meta CDN r370, Twitter r351, Alibaba r386, Baleen24 r382/r395) and the eight-of-nine generative-leaderboard claim documented in `LEADER-BOARD.md` (lines 22–47).
**Tone:** professional, sharp; the race is collegial, the critique is rigorous. No fixes proposed — this is review, not collaboration.

---

## TL;DR

LANL's chunk-surface cascade is not a generative model in any conventional sense; it is a discrete coordinate-descent optimizer whose loss function is the official cachesim HRC-MAE on the real reference trace. There is no held-out evaluation. At `chunk_size=8` (Meta CDN r366; `RESPONSE-LANL.md:4373–4398`) over 1M records, the optimizer fits ~125k swap decisions per seed against the test metric. The "no-32 guard" is a real co-acceptance criterion (ATTACK 3 partially withdrawn — see below), but it does not address the underlying overfitting concern. The reported 7-decimal four-seed means (e.g. Wikipedia r328/r329 `0.0054596500`, Alibaba r386 `0.0099685750`) are systematically reported with precision two to four orders of magnitude tighter than the per-seed range — many recent "improvement" rounds are well below seed noise. The cascade chains are not reproducible from the published prose; they require the artifact tree at `/tiamat/zarathustra/altgan-output/`. Most damaging, the IRD-renewal donor lineage that feeds Meta CDN, CloudPhysics, and Wikipedia "synthetic" donors is fit per-trace against the real CSV's IRD distribution — the synthetic-`obj_id` rule does not insulate the architecture from real-trace leakage at the inter-arrival level.

We count **8 ATTACKs**, **2 NITs**, and **1 DEFENSIBLE** below.

---

## 1. [ATTACK] Chunk-surface cascade IS supervised optimization on the test metric

**Claim attacked:** That the eight current LANL leaderboard rows (Wikipedia r328/r329 `0.0054596500`, Meta CDN r370 `0.0237592500`, Twitter r351 `0.0236117250`, Alibaba r386 `0.0099685750`, Baleen24 r395 `0.0204949000`, CloudPhysics r292 `0.0220106406`, MSR Exchange r291 `0.0043343667`, Tencent r287 `0.03010`) are "generative" in the sense the leaderboard contract requires (`LEADER-BOARD.md:11–14`: "synthetic trace produced by a fit-time atlas + post-hoc knobs").

**Counter:** Every one of the eight chunk-surface rounds explicitly defines its acceptance criterion as direct cachesim improvement on the official real reference. Representative quotations:

- Twitter 8-row r351 (`RESPONSE-LANL.md:3973–3979`): "Candidate chunks had to improve the official six-policy surface and avoid a no-32 guard regression… only synthetic `obj_id` chunks accepted on official cachesim improvement."
- Wikipedia r328/r329 (`RESPONSE-LANL.md:3258–3261`): "chunks were accepted only when the official six-policy Wikipedia cachesim mean improved."
- Alibaba r386 (`RESPONSE-LANL.md:4930–4932`): "The acceptance surface is the official 5-cache x 6-policy Alibaba evaluator, with the no-32 guard… rejecting cache-32-only moves."
- Meta CDN r370 (`RESPONSE-LANL.md:4543–4546`): "Candidate chunks had to improve the official six-policy surface and avoid a no-32 guard regression."

This is greedy coordinate-descent with the test-set HRC-MAE as the loss function. The "synthetic donor" rule (attacked in §6) is the only structural constraint; it does not change the fact that the optimizer's stopping criterion is the score it is later reported against. There is no held-out evaluation surface, no train/test split on real-trace structure, and no validation that the cascade generalizes to a different ground-truth trace.

**Why it matters:** A claim of generative HRC-MAE that is also the optimization objective is, by construction, a memorization upper bound, not a generalization metric. The leaderboard's "generative" column conflates "fit-time atlas + post-hoc knobs" (the methodology contract) with "fit-time atlas + post-hoc knobs + post-hoc test-metric coordinate descent." Those are not the same protocol. LLNL's own `llgan.chunk_ensemble` is vulnerable to the same critique (see §9), but LANL's variant pushes the parameter count two to three orders of magnitude higher and therefore the overfitting risk is correspondingly larger.

---

## 2. [ATTACK] At `chunk_size=8`, the chunk-surface cascade fits ~125k binary swap decisions per seed against the test metric

**Claim attacked:** That Meta CDN r366/r369/r370 represent generative-architecture progress.

**Counter:** Meta CDN r366 (`RESPONSE-LANL.md:4373–4398`) uses `chunk_size=8`. Over 1M-row Meta CDN traces, that is `1_000_000 / 8 = 125_000` independent chunk slots, each one a discrete swap-or-keep decision. r369 drops to `chunk_size=4` (`RESPONSE-LANL.md:4442–4467`), 250k slots; r370 drops to `chunk_size=2` (`RESPONSE-LANL.md:4541–4569`), 500k slots. These are run for four seeds (`{42,80,81,82}`), so the banked r370 row carries ~2M swap decisions across the per-seed artifact set, every one of which was selected by direct cachesim feedback on the real reference.

The 2DIO paper (EuroSys '26) — which LANL itself describes as "useful prior-art foil" with "limited generative architecture" (`RESPONSE-LANL.md:4986`) — fits roughly five parameters per trace. LANL's chunk-surface cascade fits parameters per real-trace cachesim evaluation at a rate ~10⁵× higher and, unlike 2DIO, every parameter is fit by the test-metric gradient itself. The cascade has more degrees of freedom, fit by a more aggressive procedure, against the metric it is evaluated on. The argument that this is structurally distinct from memorization needs to be made explicitly; the current `RESPONSE-LANL.md` presents the chain as if it were merely "tightening."

**Why it matters:** A reviewer for an ML or systems paper will notice that the parameter count of the post-hoc selector exceeds the number of records in the trace by a constant factor when `chunk_size < 8` (which several rounds reach). At that point the synthetic trace can encode the real trace at row resolution; the only bound on memorization is the `obj_id`-only swap contract and the donor-pool diversity, neither of which is currently quantified.

---

## 3. [DEFENSIBLE] No-32 guard is a co-acceptance criterion, not decorative

**Claim attacked (initial reading):** That the no-32 guard is invoked only as a post-hoc sanity check while the actual swap selector accepts on the official mean alone.

**Counter (after evidence):** Withdrawn. The published guarded-acceptance language ("must improve the official six-policy mean **and** must not regress the no-32 guard") appears consistently from r350 onward and is symmetric. See:

- r350 (`RESPONSE-LANL.md:3907–3911`): first posted run with guarded acceptance; "a candidate must improve the official six-policy Twitter mean and must not regress the no-32 guard."
- Per-round confirmation lines (`RESPONSE-LANL.md:4046, 4082, 4135, 4172, 4200, 4229, 4258, 4287, 4346, 4375, 4404, 4444, 4545, 4669`) all use "avoid a no-32 guard regression."

**However**, two residual concerns survive as separate attacks:

1. The guard window itself (`128,512,2048,8192`) is also a subset of the test-metric capacity ladder, so the guard reduces the dimensionality of the acceptance vector but does not change the supervised-on-the-test-set objection in §1.
2. Rounds before r350 (Twitter r343/r344/r345/r346/r348/r349, Alibaba r323/r326/r340, Wikipedia r328/r329, CloudPhysics r292, Meta CDN r320/r356) were accepted on the official mean alone, with the no-32 number reported only as post-hoc diagnostic ("Diagnostic no-32 sensitivity… is not the race score" — `RESPONSE-LANL.md:3358, 3439, 3479, 3519`). Several of these are still the published banked rows (Wikipedia r328/r329 banked, CloudPhysics r292 banked). For those rows the diagnostic-only label is the correct one and ATTACK §3 still applies in its weaker form.

**Verdict:** Concession on the symmetric-acceptance question for r350+; the broader concern about which surface the optimizer actually sees moves to §10 below.

---

## 4. [ATTACK] Reported 7-decimal four-seed means hide that most "improvements" are below seed noise

**Claim attacked:** That `LEADER-BOARD.md` margins computed to four-decimal absolute precision (e.g. Alibaba r386 `0.0000304250` lower than LLNL R287.A2; line 41) are statistically meaningful.

**Counter:** Per-seed ranges across the recent LANL cascade rounds are systematically larger than the inter-round improvements. Direct quotation evidence:

- Wikipedia r328/r329 four-seed range: `0.0008116333` (`LEADER-BOARD.md:34`, `RESPONSE-LANL.md:3282`). Per-seed values: `0.0051168, 0.0051857, 0.0056077, 0.0059284`. The seed-82 vs seed-42 gap is `0.000812`, ~150× the typical inter-round delta in the chain.
- Alibaba r386 four-seed range `0.0001902667` (`RESPONSE-LANL.md:4944`); claimed margin over LLNL R287.A2 `0.0000304250` (`LEADER-BOARD.md:41`). The margin is **6× smaller than the per-seed range**.
- Twitter r351 four-seed range `0.0005990667` (`RESPONSE-LANL.md:4001`); improvement over r350 reported as `0.0000123333` (`RESPONSE-LANL.md:4001`). The "improvement" is **49× smaller than the seed range**.
- CloudPhysics r292 four-seed range `0.0053736458` (`LEADER-BOARD.md:28`, `RESPONSE-LANL.md:119`). Per-seed values `0.0192488, 0.0246224, 0.0222604, 0.0219109` span 28% of the headline mean. Reporting that mean to seven decimal places (`0.0220106406`) is precision theatre.
- Meta CDN r370 range `0.0013576000` (`RESPONSE-LANL.md:4558`), improvement over r369 reported as `0.0000229083`. Improvement is **59× smaller than the per-seed range**.

Counting the explicit inter-round improvements in `RESPONSE-LANL.md` for the r340→r386 Alibaba chain, the r342→r395 Baleen24 chain, the r320→r370 Meta CDN chain, and the r330→r351 Twitter chain: **of approximately 30 posted "improvement" deltas, at least 22 are smaller than 25% of the corresponding 4-seed range** and therefore statistically indistinguishable from seed noise under any reasonable hypothesis test. The cascade chain is producing a downward-drifting random walk in noise, with the optimizer's monotone-acceptance rule guaranteeing the visible trajectory looks like steady progress.

**Why it matters:** If two of the four seeds are perturbed independently (e.g. by a different RNG schedule on `vinge` vs `baase`), several published "tightening" rows could vanish. The leaderboard ought to attach a per-row standard error and refuse to flip leadership unless the gap is at least one standard error of the four-seed mean.

---

## 5. [ATTACK] Cascade chain is not independently reproducible from published protocol

**Claim attacked:** That the LANL chunk-surface cascade rows are reproducible by a third party.

**Counter:** Each round explicitly continues from the prior round's per-seed CSV artifact. Examples:

- Alibaba r386 (`RESPONSE-LANL.md:4926–4927`): "continues r384 with synthetic self-donor… 32-row chunks."
- r384 (`RESPONSE-LANL.md:4840–4842`): "continues r368 with synthetic self-donor… 64-row chunks."
- r368 (`RESPONSE-LANL.md:4694–4696`): "continued from r364."
- r364 (`RESPONSE-LANL.md:4402–4406`): "continued the Alibaba r360 256-row best-donor champion with a 128-row object-ID chunk pass."
- r340 (`RESPONSE-LANL.md:3211–3215`): "continued the Alibaba r326 1K… with a 512-row best-donor selector."
- r326 (`RESPONSE-LANL.md:3176`): "continued from r323."

The chain depth is at least seven rounds. Each link references a CSV file under `/tiamat/zarathustra/altgan-output/` (e.g. `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r364_best128_ck128_seed42_fake_1000k.csv`). The donor banks are described prosaically ("LANL synthetic donors only," "donor bank stayed LANL synthetic-only: prior Alibaba chunk stages, the clean-cond donor trace") but no enumerated donor manifest, donor-row checksum, or selector-determinism guarantee is provided. The donor pool grows monotonically with prior rounds, so without the prior artifacts a fresh run cannot reproduce the round.

**Why it matters:** Independent reproducibility is the publishable bar. A reader of a paper claiming "LANL Wikipedia 0.0055 generative HRC-MAE" cannot, given only the paper text, reproduce the number. They need at minimum: the donor-bank manifest with content hashes, the selector RNG seed, the candidate-evaluation order, the `--max-evals` and `--max-candidates-per-chunk` budgets per round, and the prior artifacts. None of those are currently in `RESPONSE-LANL.md`.

---

## 6. [ATTACK] Synthetic-donor rule does not insulate the architecture from real-trace leakage at the inter-arrival level

**Claim attacked:** That the IRD-renewal donor lineage feeding Meta CDN r320→r370, CloudPhysics r292, and Wikipedia r288/r290 is "synthetic" in the same sense as a fit-time atlas sample.

**Counter:** `altgan.ird_renewal` is described in `RESPONSE-LANL.md:1936–1940` as: "logarithmic object-rank-conditioned IRD buckets. The generator still uses synthetic object IDs only; **it fits the official real CSV into object-count ranks plus inter-reference distances**, then emits a heap-scheduled renewal stream." The Meta CDN r320 retake (`RESPONSE-LANL.md:3140–3147`) directly inherits this: "empirical IRD renewal with `ird_s=1.0`, `ip=0.10`, and 16 rank-conditioned IRD buckets."

So the per-trace IRD distribution and per-rank IRD bucketization are extracted from the real reference and consumed verbatim by the generator. Object IDs are randomized; inter-arrival shape is not. The cachesim metric is dominated by reuse-distance / IRD shape (this is why LRU is the hardest policy to fool). Fitting the IRD distribution per-trace and then claiming the result is a "generative" model that competes against atlas-fit baselines is asymmetric: LLNL atlas baselines transfer fit knowledge across traces in a single corpus through a shared model; LANL IRD-renewal fits each trace independently.

When that IRD-renewal output is then used as the chunk-surface cascade *base* for subsequent rounds (Meta CDN r356, r361, r362, r363, r365, r366, r369, r370 are all "IRD-seeded," `RESPONSE-LANL.md:4042–4569`), the entire cascade chain inherits the per-trace IRD fit. The "synthetic obj_id" rule is preserved literally; the spirit of "no real-trace leakage" is not.

**Why it matters:** The Meta CDN row `0.0237592500` and the CloudPhysics row `0.0220106406` both rest on per-trace IRD fitting. Compared against an atlas-trained generative baseline, this is a category error: the LANL row competes in a different methodology class. The leaderboard does not currently distinguish.

---

## 7. [ATTACK] Audit-promotion gap: LANL cascades update faster than the LLNL audit window

**Claim attacked:** That the leaderboard rows are formally banked through a process LLNL has had time to audit.

**Counter:** Between 2026-05-04 and 2026-05-08 (approximately 96 hours), `RESPONSE-LANL.md` posts ~50 cascade-tightening rounds across 9 corpora (grep of `## 2026-05-0[4-8]` headers in `RESPONSE-LANL.md`: 70+ section headers, of which 50+ are banked or scout cascade rounds). The leaderboard row for Alibaba alone moves through r303 → r323 → r326 → r340 → r360 → r364 → r367 → r368 → r384 → r386 in that window — at least 10 promoted rounds in four days, each consuming 4 cachesim seeds × 6 policies × 5 capacities = 120 evaluations per round, plus the per-round selector budget of (per `RESPONSE-LANL.md:5093–5103`) `--max-evals 800` candidate evaluations. The protocol header (`LEADER-BOARD.md:4–7`) requires multi-seed mean + range and per-seed cachesim lines in the response file, which LANL does provide; what it does not require is an LLNL audit pause before the row flips. In practice the leaderboard reflects every round LANL banks the same git push.

**Why it matters:** The race protocol is currently asymmetric. LANL can post a row, watch it flip on the leaderboard, and continue iterating before LLNL has read the diff. The cascade family in particular benefits from this — each round depends on the prior round's CSV, so a round that fails review later cannot be rolled back without invalidating the chain. A formal proposal: leaderboard rows from cascade-class methods should require a 24-hour LLNL audit window before flipping, and rows that fail review unwind the chain back to the last audited base.

---

## 8. [ATTACK] LANL Alibaba and Twitter "leads" are inside the joint seed-noise envelope

**Claim attacked:** Two specific banked margins:
- Alibaba: LANL r386 `0.0099685750` vs LLNL R287.A2 `0.009999`, claimed gap `0.0000304250` (`-0.3043%`, `LEADER-BOARD.md:41`).
- Twitter: LANL r351 `0.0236117250` vs LLNL R287.M2 `0.02491`, claimed gap `0.0012982750` (`-5.21%`, `LEADER-BOARD.md:31, 87`).

**Counter (Alibaba):** LANL r386 four-seed range is `0.0001902667` (`RESPONSE-LANL.md:4944`); LLNL R287.A2 four-seed range is `0.000208` (`LEADER-BOARD.md:26, 76`). Standard error of a four-seed mean given range R is approximately `R / (2 × √4) = R/4`; for LANL ≈ `4.8e-5`, for LLNL ≈ `5.2e-5`. The claimed margin `0.0000304250` is smaller than either standard error and well within a 1-sigma joint envelope (`√(4.8² + 5.2²)e-5 ≈ 7.1e-5`). At 1-sigma the lead is not statistically significant; at any reasonable confidence interval, the row is tied.

**Counter (Twitter):** LANL r351 range `0.0005990667` (`RESPONSE-LANL.md:4001`); LLNL R287.M2 range `0.000553` (`LEADER-BOARD.md:31`). Standard errors `≈ 1.5e-4` and `≈ 1.4e-4`. Joint envelope `≈ 2.0e-4`. The claimed margin `0.001298` is `≈ 6.5σ` and **does** clear the noise floor — concede the Twitter lead is statistically real.

So: ATTACK confirmed for Alibaba; concession on Twitter. The Alibaba row should not be reported as a flip on the basis of a margin smaller than either team's own seed noise. Either team adding a fifth seed could swing it. This connects to §4: 7-decimal precision on a row whose 1-sigma is at the 5th decimal is not meaningful.

**Why it matters:** The leaderboard line "LANL leads 8/9" rests partly on a row that, by elementary statistics, is a tie. The paper-ready statement should be "LANL leads 7 with one tied-within-noise."

---

## 9. [ATTACK] Methodology asymmetry between LLNL and LANL is real and quantifiable

**Claim attacked:** That LANL's chunk-surface cascade and LLNL's `llgan.chunk_ensemble` are methodologically equivalent and the comparison is therefore fair.

**Counter:** Both teams have post-hoc cachesim-driven swap selectors. They are not equivalent. The relevant axis is parameter count per seed, which is `1M / chunk_size`:

| Team | Method | Typical chunk_size | Swap decisions per 1M-row seed |
|---|---|---:|---:|
| LLNL | `llgan.chunk_ensemble` (R287.A2 lineage) | 8192 | ~120 |
| LANL | chunk-surface cascade (Meta CDN r370, Twitter r351) | 2 to 16 | 60k to 500k |
| LANL | chunk-surface cascade (Alibaba r386, Wiki r328/r329) | 32 to 32768 | 30 to 31250 |

LANL's cascade at `chunk_size=2` (Meta CDN r370) fits roughly **4000× more swap decisions per seed** than LLNL's `chunk_ensemble` at `chunk_size=8192`. Both are fitting the test metric, but the LANL family fits dramatically more. If the paper reports both as "post-hoc generative selection," readers will assume comparable fit budgets; the actual fit budgets differ by three to four orders of magnitude.

**Why it matters:** The headline "LANL leads 7 of 9 corpora" reflects, at least in part, more aggressive metric-fitting, not better generative architecture. Either both teams must publish parameter-count-per-seed numbers, or the leaderboard must split into "atlas-only" and "atlas + post-hoc selector" classes with each team's selector parameter-count disclosed.

---

## 10. [ATTACK] No-32 guard does not address the supervised-on-the-test-set objection

**Claim attacked:** That the no-32 guard makes the cascade scientifically defensible.

**Counter:** The guard subset `{128, 512, 2048, 8192}` is itself drawn from the official cachesim ladder `{32, 128, 512, 2048, 8192}`; the guard is supervised against the same real reference. So both the official and guard surfaces are functions of the same real-trace ground truth. The selector now optimizes a 2D vector (`official_mean`, `no32_mean`) instead of a scalar, but both components see the test set. The guard prevents pathological cap-32 overfit; it does not prevent overfit to capacities `{128, 512, 2048, 8192}` jointly. A genuine generalization guard would be: hold out one of the five capacities (or a different real trace from the same corpus) at fit time, optimize on the other four, evaluate on the held-out point. LANL has not done this.

**Why it matters:** The current `RESPONSE-LANL.md` framing ("the gain is not just a cap-32 artifact" — `RESPONSE-LANL.md:3938, 4006, 4040, 4076, 4310, 4369, 4398, 4467, 4569`) is over-reading what the guard tests. It tests "the gain is not concentrated at cap-32," not "the gain generalizes." Those are very different statements.

---

## 11. [NIT] Leaderboard prose uses "race display" and 4-decimal precision inconsistently with header bookkeeping

**Claim attacked:** Internal consistency of the per-row reporting convention.

**Counter:** `RESPONSE-LANL.md` rows use a "race display" rounded to 4 decimals (e.g. r386: "race display `0.0100`," `RESPONSE-LANL.md:4944`) but `LEADER-BOARD.md:26` records `0.0099685750` to 10 decimals. The same number is reported at three different precisions (4-dp display, 6-dp narrative, 10-dp leaderboard) in different files. This is not a correctness defect, but it makes diff-tracking the leaderboard against the response log brittle. A reviewer comparing `LEADER-BOARD.md` line 26 to the seed-mean computation in `RESPONSE-LANL.md:4937–4944` has to mentally truncate from 10 to 6 decimals; they should not have to.

---

## 12. [NIT] Cache-32 sanity rule (`RESPONSE-LANL.md:3888–3902`) was published *after* several cascade rounds were already banked

**Claim attacked:** That the no-32 guard discipline applies retroactively to all banked LANL rows.

**Counter:** The "Cache-32 Sanity Rule" header is dated 2026-05-07. Rows posted *before* that date (Wikipedia r328/r329 dated 2026-05-07 but in the same morning post; CloudPhysics r292 dated earlier; Twitter r330–r349 dated 2026-05-07 with explicit caveat at `RESPONSE-LANL.md:3528–3531` that "this run launched before the guarded acceptance patch, so candidates were accepted on official six-policy improvement; the post-hoc no-32 diagnostic below checks…") are diagnostic-only on no-32. CloudPhysics r292's banked seed-80 value `0.0246224` versus seed-42 `0.0192488` is a 28% intra-row spread, and the no-32 audit was applied post hoc. The published Wikipedia r328/r329 row is still the highest-impact banked LANL claim at 38.6% gap over LLNL; it is in this pre-guard window.

---

## Verdict

**8 ATTACKs**, **2 NITs**, **1 DEFENSIBLE** — total **11 numbered objections**, of which 10 require LANL response.

ATTACKs requiring substantive response: §1 (supervised-on-test-set), §2 (parameter count), §4 (false precision / sub-noise improvements), §5 (reproducibility), §6 (real IRD leakage), §7 (audit-promotion gap), §8 (Alibaba lead within noise), §9 (methodology asymmetry vs LLNL), §10 (no-32 guard does not establish generalization), §12 (pre-rule banked rows).

The chunk-surface cascade family is a real engineering accomplishment and has produced numerically lower HRC-MAE than LLNL on most of the leaderboard. That observation does not, by itself, make the resulting numbers a generative-architecture claim suitable for publication without methodology disclosure that the current `RESPONSE-LANL.md` does not provide. LANL is invited to respond round-by-round in `RESPONSE-LANL.md` or by amending `LEADER-BOARD.md` with the standard-error and methodology-class refinements proposed implicitly above. Until §1, §2, §4, and §6 are addressed, LLNL recommends that the headline "LANL leads 8/9" be qualified in any paper draft as "LANL leads 7/9 on raw HRC-MAE under a chunk-surface cascade methodology that fits 30 to 500,000 swap decisions per seed against the test metric; one row (Alibaba) is statistically tied; one row (Baleen24) remains LLNL."

---

*File paths used in this review (all absolute):*
- `/Users/darrell/LLNL/Zarathustra/RESPONSE-LANL.md`
- `/Users/darrell/LLNL/Zarathustra/LEADER-BOARD.md`
- `/Users/darrell/LLNL/Zarathustra/PEER-REVIEW-LANL.md`
- `/Users/darrell/LLNL/Zarathustra/PRIOR-ART-2DIO-LLNL.md` (referenced re: 2DIO comparator)
- `/tiamat/zarathustra/altgan-output/` (LANL CSV/JSON artifact tree, audit-deferred for spot-check in follow-up review pass)
