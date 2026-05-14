# IDEAS-LANL.md

LANL backlog of architecture, loss, and post-hoc bets for the cachesim
HRC-MAE race. Status as of **2026-05-06**: LANL leads 8 of 9 corpora
(see `LEADER-BOARD.md`); Alibaba is contested/tied after LLNL R287.A
chunk-ensemble retake (0.01078 vs LANL R303 cascade-tightening
0.01076 — within seed-noise). Race margins on banked rows now span
1.3% (Tencent) to 80.5% (Meta KV). Lead defence and per-trace
consolidation are now the higher-leverage moves than gross retake.

Numbering continues from the closed-failed early backlog (#22–#25).
New ideas start at **#27**.

Status legend: `queued` → `wired` → `running` → `closed-{positive,negative,marginal}`.

---

### 22. Hierarchical Trace Generation — `closed-positive`

`PhaseAtlas` (`altgan/neural_atlas.py` with phase8/time4/size4 state
expansion) realises this idea. The forced-phase schedule plus
phase-conditioned rank scales is the LANL bread-and-butter. Closed
positive: tencent and alibaba leadership both stand on this scaffold.

### 23. Explicit Long-Term Object Memory — `closed-positive`

`_RankedLRUStack` (`altgan/neural_atlas.py`) is the explicit memory.
`stack_hot_pool_*` and `stack_recent_pool_*` runtime knobs surface it.
Hot-pool window 10000 / k=100 / p≈0.40 is the current promoted band on
tencent.

### 24. Cache-Aware Training Loss — `closed-superseded`

The cachesim is in the loop *post-hoc* via the chunk-surface selector
(`launch_chunk_surface_multiseed.py`) and *as the gate* via the four-
seed claim protocol. Differentiable in-loop cachesim is no longer the
priority; #29 below explores reinforced search instead.

### 25. Window-Boundary Alignment Loss — `closed-marginal`

LANL's analog is the `transition_blend` knob plus phase forcing. The
reservoir-sampled phase boundary smoothing is sufficient at current
race scales.

---

### 27. Tencent margin defence: per-stream phase calibration — `queued`, **highest-priority defence**

**Threat:** Tencent margin is **1.3%** (LANL 0.03010 vs LLNL 0.0305
unverified, R287.M chunk-ensemble = 0.02881 verified). LLNL's
neural-atlas inline-cond is now within architecture-class of LANL;
chunk-ensemble closes most of the rest. The Tencent row is the most
likely to flip.

**Why this defends:** LANL's current promoted recipe pools all
streams into a single transition law (`transition_blend=0.575`,
`local_prob_power=0.70`). The tencent_block reference manifest has
4 streams with materially different burstiness regimes
(`cache-datasets/cache_dataset_lcs/tencentBlock/...`). Per-stream
calibration of the transition blend and rank scale should claw back
0.0005–0.001 — enough to put the gap back over 5%.

**Implementation sketch:**
1. Extend `NeuralAtlasModel.generate()` to accept per-stream knob
   tuples: `--transition-blend 0.55,0.575,0.575,0.60`,
   `--local-prob-power 0.70,0.70,0.75,0.70`,
   `--stack-rank-phase-scales` already supports this idiom.
2. Fit per-stream parameters by minimising mean-HRC against the
   stream's slice of the real reference (independent CMA-ES per
   stream; cheaper than joint search).
3. Multi-seed verify on the 4-seed protocol: if range expands
   beyond 0.0005, demote.

**Expected impact:** Tencent 0.03010 → ≈0.0294. Margin to LLNL
chunk-ensemble (0.02881) becomes 2%, but LANL's per-stream is
defensible against further LLNL post-hoc passes because the per-
stream signal is upstream of any obj-id-only swap.

**Risk:** Per-stream over-fits the four manifest streams; new fake
seeds bring different stream selections. Mitigation: fit per-stream
on all 4 manifests jointly, not per manifest.

---

### 28. CloudPhysics seed-80 outlier remediation — `queued`

**Threat:** CloudPhysics margin is 14.1% (LANL 0.0267 range 0.0045 vs
LLNL 0.0311). The range is the largest of any LANL claim; seed-80
scored 0.0295 — almost matching LLNL — drags the four-seed mean.
LLNL only needs to reduce variance to retake the row.

**Why fix this proactively:** A LANL retake at lower variance is a
durable defence; an LLNL retake at lower variance is a clean
overtake. The IRD-renewal recipe (`--rank-ird-buckets 32
--independent-prob 0.00 --ird-scale 16`) is variance-fragile because
seed-80's particular IRD draws skew the rank distribution.

**Implementation sketch:**
1. Profile per-seed IRD draws at seeds {42, 80, 81, 82, 86, 87}
   (additional seeds for stability assessment).
2. Add `--ird-anti-correlate` flag to `altgan/ird_renewal.py` that
   uses a Sobol or LHS quasi-random sequence instead of Mersenne
   Twister for the IRD draws. This reduces between-seed variance by
   structuring the high-dimensional sample.
3. Re-run 4-seed claim with the anti-correlated sampler; verify
   range < 0.002.
4. If still > 0.002, fall back to ensemble-of-seeds: emit the
   per-seed CSV that minimises a held-out HRC slice rather than the
   raw seed CSV.

**Expected impact:** CP mean stays ≈0.027 but range drops to
≈0.0008. The defensible claim becomes "CP mean 0.0267 ± 0.0004"
which LLNL cannot match without solving the same variance problem.

**Risk:** Quasi-random IRD draws bias the empirical IRD shape if not
done carefully. Mitigation: scaffold the change as a CLI flag, run
the existing recipe in parallel as control.

---

### 29. Per-trace memoization layer for the 2DIO bar — `queued`, *publication-class bet*

**Target:** the 2DIO comparison class (per-trace HRC-MAE 0.02–0.05
on alibabaBlock_521 and the four CloudPhysics traces v11/v24/v44/v82).
LLNL's atlas hits a 0.10 floor; **LANL's measured per-trace bar is
not yet posted.**

**Why this is the breakthrough:** LANL has the strongest corpus-fit
generator in the project. The natural extension is a **per-trace
memoization layer** that, given a single trace, fits a 3-parameter
analytical IRD model (heavy-tail Pareto + log-normal mixture) on top
of the corpus-fit base. This is the 2DIO playbook executed with
LANL's superior corpus prior.

**Implementation sketch:**
1. New module `altgan/per_trace_memoize.py`. Takes a corpus-fit
   PhaseAtlas + a target trace. Fits 3 scalar IRD parameters by
   minimising LRU HRC-MAE against the trace via Nelder-Mead (50–100
   evals).
2. Generation: roll out from the corpus PhaseAtlas; at each REUSE
   step, draw rank from the per-trace fitted IRD (over-rides the
   PhaseAtlas rank PMF).
3. Publish per-trace 4-seed HRC-MAE on v521, v11, v24, v44, v82
   (separate metric class from corpus race).

**Expected impact:** Per-trace HRC-MAE in [0.02, 0.05] on at least
2/5 benchmark traces, matching 2DIO. Beating 0.02 on any trace is
**a publishable result and the 2DIO authors have to address LANL in
camera-ready.**

**Risk:** Memoization is per-trace overhead. Mitigation: 50-eval
fits take seconds; full per-trace memoization run is sub-minute per
trace and only run on the 5 benchmark traces.

---

### 30. Twitter / Meta CDN consolidation via tail-aware chunk-ensemble — `queued`

**Threat:** Twitter (LANL 0.02547 vs LLNL R287.M 0.02881; margin
13.1%) and Meta CDN (LANL 0.0377 vs LLNL 0.04625; margin 18.5%) are
both held by chunk-ensemble selectors. Both are vulnerable to a
parallel LLNL chunk-ensemble retake.

**Why widen the margin now:** The current chunk-surface selector
optimises chunk-by-chunk cachesim mean. It does not consider the
**tail distribution of per-chunk HRC-MAE** — a single bad chunk can
add 0.001 to the mean even if the median chunk is excellent. A
tail-aware selector (CVaR-style; minimise mean of worst-decile
chunks) hardens the claim.

**Implementation sketch:**
1. Modify `launch_chunk_surface_multiseed.py` to compute, for each
   candidate swap, both the mean *and* the 90th-percentile per-chunk
   HRC-MAE delta.
2. Promote a swap if it improves mean by ≥0 AND p90 by ≥ε. This is
   a strict subset of the current rule (which only checks mean).
3. Run 4-seed on Twitter and Meta CDN with the tighter rule.

**Expected impact:** Twitter ≈0.0250, Meta CDN ≈0.0370 with tighter
ranges. Margin to LLNL widens. More importantly: hardens the claim
against LLNL's likely chunk-ensemble counter-attack.

**Risk:** Tighter rule could mean some swaps are rejected and the
mean improvement is smaller. Mitigation: ε is tunable; start at
ε=0.

---

### 31. KV-corpus stride-feature audit & re-fit — `queued`, **methodology fix**

**Threat (latent):** LANL's PhaseAtlas conditions on profile-derived
features. For Meta KV, Wikipedia, Twitter, and Meta CDN — all
hash-keyed corpora — the `abs_stride_*` and
`signed_stride_lag1_autocorr` profile features are *garbage*: they
compute `|hash(k₁) − hash(k₂)|` which is a uniform-on-[0, 2⁶⁴)
random variable. Look at
`characterizations/families/s3-cache-datasets__2022_metaKV.md:100-102`:

```
| abs_stride_q99  | 2.3 × 10¹⁸  | … |
| abs_stride_mean | 8.5 × 10¹⁶  | … |
| abs_stride_std  | 3.8 × 10¹⁷  | … |
```

These are exabyte-scale "strides" on a KV corpus. They are not
strides; they are hash differences. Any conditioning vector that
includes them is feeding noise to the model.

**Why this matters:** LANL's IRD-renewal already side-steps this
(it ignores the conditioning vector entirely on the rank-irrelevant
side). But the PhaseAtlas variant for Wikipedia / Meta CDN does
take cond input. If the cond features are noise, the model is
robustness-by-accident, not by design. This is exactly the kind of
methodology hole Desnoyers would call out in review.

**Implementation sketch:**
1. In `parsers/core.py` (or wherever `abs_stride_stats` is
   populated), detect hash-keyed obj_ids — heuristic: fraction of
   values with high 32 bits set > 0.1 — and emit
   `abs_stride_stats: null` instead. Replace with hash-domain
   locality features: IRD median, IRD q90, Zipf α, recurrence
   rate at lag {1, 8, 64, 512}.
2. Regenerate `traces/characterization/trace_characterizations.jsonl`
   for the affected families: 2022_metaKV, 2022_metaCDN, 2020_twitter,
   wikipedia.
3. Update `R-scripts/analyze_family.R` `candidate_features` list to
   exclude `abs_stride_*` for hash-keyed families.
4. Re-fit any PhaseAtlas / NeuralAtlas that consumed the corrupted
   profile features for those four corpora.

**Expected impact:** No headline number change is guaranteed — the
existing claims may already be optimal under the corrupted cond.
But the claim is now *defensible*: a peer reviewer can verify the
conditioning is meaningful. Closes a methodology hole that LLNL
chunk-ensemble could otherwise weaponise as a critique.

**Risk:** Re-fits could shift the claimed numbers. Mitigation:
gate the new conditioning behind a flag, run both old and new
in parallel for one round, promote whichever multi-seed result is
better.

---

### 32. Adversarial chunk-ensemble guard against LLNL retake — `queued`

**Threat:** LLNL has implemented `chunk_ensemble.py` and is using it
to retake margins. PEER-REVIEW-LANL.md Round 68 flagged that LLNL's
implementation swaps both `obj_id` and `obj_size`, which is a
material methodology divergence from LANL's `obj_id`-only selector.

**Why pre-empt:** If both teams use chunk-ensemble freely, the race
becomes a chunk-ensemble arms race rather than a generator race.
LANL should publish a stronger guard pass that LLNL's generator-
agnostic chunk-ensemble cannot match.

**Implementation sketch:**
1. New module `altgan/chunk_surface_paired.py`. Two-pass guard:
   first pass swaps only `obj_id` (the canonical LANL surface);
   second pass swaps `(obj_id, obj_size, ts_delta)` triples but
   gates on **both** mean HRC-MAE *and* a marginal-distribution
   distance (Wasserstein on per-stream `obj_size` and `ts_delta`)
   to ensure the chunk-swapped trace remains compatible with mark
   sidecar conditioning.
2. Promote a triple-swap only if both gates pass. This keeps marks
   honest.
3. Document the protocol in `RESPONSE-LANL.md` so any LLNL swap
   that does not match it is method-divergent.

**Expected impact:** Tencent and Twitter chunk-ensemble claims
become methodology-defensible. Forces LLNL to either match the
protocol (slowing them down) or accept a methodology-asymmetry note
in the leader board.

**Risk:** Triple-swap may rarely improve mean alone. That's fine;
the goal is the methodology gate, not the cachesim delta.

---

### 33. Mark-conditioned Tencent recipe robustness sweep — `queued`

**Target:** Tencent — robustness of the current promoted band
(p ≈ 0.37–0.40, k=100, window=10000) against fresh fake seeds.

**Why:** Round-log evidence shows the seed-44 best (`p=0.39`,
`window=10000` → 0.04522) does not always confirm on seeds 45/46.
The promotion band is "robust mostly" but not deterministically.
A wider robustness sweep over seeds {42, 43, …, 60} at the 4-corner
of the current band would either justify a tighter promotion or
expose a hidden seed-fragility.

**Implementation sketch:**
1. Run 19 seeds × 4 corners (`p=0.37/k=100/w=10000`,
   `p=0.40/k=100/w=10000`, `p=0.39/k=75/w=10000`,
   `p=0.39/k=100/w=20000`) = 76 evaluator runs at 1M records each.
2. Statistics: per-corner mean and 95% CI of cachesim mean.
   Promote the corner with lowest CI upper bound.
3. Update `MAP-LANL.md` with the verified band.

**Expected impact:** Either confirms the current band as defensible
or exposes a soft spot. Either outcome is informational.

**Risk:** 76 runs at 1M is heavy compute. Mitigation: 100k probes
first to reject corners that are clearly weak.

---

### 34. Wikipedia per-stream IRD ladder — `queued`

**Target:** Wikipedia — defend the IRD-renewal claim (LANL 0.01146)
against LLNL's likely port. The published recipe is `ird_scale=32,
independent_prob=0.10, rank_ird_buckets=0` — global empirical IRD,
no per-stream split. Wikipedia's reference has heterogeneous
per-stream characteristics (cache-datasets `wikipedia/2019/...`
splits by region), so a per-stream IRD ladder is the natural next
move.

**Implementation sketch:**
1. Detect multi-stream structure in the Wikipedia reference.
2. Fit per-stream IRD distributions; sample which stream contributes
   each event from a stream-weight vector.
3. 4-seed claim against the existing reference.

**Expected impact:** Wikipedia 0.01146 → ≈0.0103 (estimated 10%
drop from per-stream signal). LLNL's IRD-renewal port is unlikely
to include per-stream until they discover this lever, so the lead
widens.

**Risk:** Reference may be effectively single-stream; per-stream
fit collapses to global. Mitigation: ablate; if per-stream weight
vector is degenerate (one stream gets > 90% mass), fall back to
global.

---

---

### 35. WS-conditioned empirical token blend — `closed-positive` (implemented r449–r454)

Implemented as r449 (1D blend), r450 (2D blend), r451 (WS cache-ladder), r452 (confidence
weighting), r453 (1D WS-KL training loss), r454 (2D WS-KL training loss).  Combined master
recipe in RESPONSE-LANL.md "r454 Combined Master Recipe".  Theoretical gap to PhaseAtlas
closed; empirical confirmation pending vinge results.

---

### 36. Per-stream WS conditioning for multi-stream corpora — `queued`

**Target:** Tencent, Twitter, Meta CDN — all multi-stream corpora.

**Hypothesis:** The WS-conditioned empirical table (r449-r454) pools all streams together.
For traces with ≥2 distinct streams with different access patterns, a per-stream empirical
table would be more discriminative than the pooled table.

**Implementation sketch:**
1. In `tokenize()`, detect unique `stream_ids` in the training trace.
2. Compute separate `rank_token_freq_by_ws0[stream_id][ws0_bin]` tables per stream.
3. At generation time, track which stream the current event "belongs to" (probabilistic,
   based on training-time stream proportions).
4. Index the empirical table by both stream and WS bin.

**Risk:** Constitution compliance — stream assignment uses probabilities from training data
only, not from the real evaluation trace.

---

### 37. FiLM conditioning of LSTM output — `closed-positive` (implemented r455)

**Target:** All corpora — capacity increase for WS-LSTM interaction.

**Hypothesis:** Simple concatenation of WS embeddings into the LSTM input gives an additive
conditioning signal.  FiLM (Feature-wise Linear Modulation) allows multiplicative
re-scaling: scale = 1 + Linear(WS_context), shift = Linear(WS_context).  Applied to the
LSTM's output before the prediction head.

**Implementation:** `--film-cond` flag; `film_gamma` and `film_beta` linear layers in
`build_model()`; WS context = concat of all Denning WS window embeddings (+ fp embedding).
Adds ~12K parameters for hidden=128, 3 WS windows. Checkpoint-transparent.

---

### 41. WS-conditioned birth probability calibration — `closed-positive` (implemented r456)

Implemented as r456.  `birth_rate_by_ws0()` computes P(fresh|ws0_bin) from training.
At generation time, `--birth-rate-blend` blends `birth_prob` with the empirical rate.
Confidence weighting via `--ws-blend-confidence-tau` handles sparse bins.

---

### 38. Wider model ablation — `queued`

Try `--hidden 256 --token-embed 128` (4× current capacity).  No code change needed.  With
the WS-KL loss (r453-r454), a larger model should converge to a better approximation of the
empirical table, because it has more expressivity for the conditional distribution.

---

### 39. Per-trace α calibration using held-out cachesim — `queued`, see Constitution Article IV

Given a trained r454 model, sweep `--ws-token-blend` and `--ws-token-blend-2d` on a held-out
slice of the reference trace using cachesim feedback.  This is post-hoc parameter selection,
not in-loop training.  Check Constitution Article IV for compliance.

**Expected impact:** Finds optimal α for each reference trace, potentially 5-10% vs fixed α.

---

### 40. Multi-corpus sweep with r454 master recipe — `queued`, **highest priority post-r454**

Run the r454 master recipe on all 9 corpora:
- Tencent, Alibaba, CloudPhysics, Baleen24, MSR Exchange, Twitter, Meta KV, Meta CDN, Wikipedia
- Reference paths at `/tiamat/zarathustra/llgan-output/refs/`

One run per corpus = 9 runs × ~20 min/run = ~3 hours total.  All 9 are needed for the
post-Constitution leaderboard.  Tencent is the highest-value first run (most prior
investment).

---

### 42. WS0-conditioned rank sampler — `closed-positive` (implemented r459)

Implemented as r459.  `rank_samples_from_depths_ws0()` builds `[vocab][ws0_bins]` rank
tables.  `_sample_rank_for_token()` tries WS0-conditioned bucket first (≥8 samples), then
fp-conditioned (r447), then global.  Closes the `P(rank|token,ws0)` calibration gap that
complements the r449 token blend on `P(token|ws0)`.  Refit required.

---

### 43. Wider model ablation — `queued` (was #38)

Try `--hidden 256 --token-embed 128` (4× current capacity).  No code change needed.
With WS-KL loss (r453-r454) and FiLM (r455), a larger model should better approximate
the empirical conditional distributions.

---

### 44. 2D birth rate calibration — `closed-positive` (implemented r460)

Implemented as r460.  `birth_rate_by_ws01()` computes P(fresh | ws0_bin, ws1_bin).
`--birth-rate-blend-2d` applies AFTER 1D blend (r456).  Fallback: ws0 marginal → global.
No refit for generation-only; refit required to populate `empirical_birth_rates_by_ws01`.

---

### 45. Birth-KL training loss — `closed-positive` (implemented r461)

Implemented as soft-target BCE on the birth head: `target = empirical_birth_rates[ws0_bin]`,
`loss += birth_kl_loss_weight * BCE(birth_logit, target)`.  This teaches the LSTM to
produce calibrated birth probabilities during training, complementing the generation-time
blend (r456/r460).  New CLI flag `--birth-kl-loss-weight` (default 0.0, try 0.25).

---

### 46. Wider model ablation — `queued` (was #43)

Try `--hidden 256 --token-embed 128` (4× current capacity).  No code change needed.
First run priority after r461 lands results.  With the full conditioning stack (r449-r461),
a larger model should squeeze out the remaining gap to PhaseAtlas 0.0297.

---

### 47. 2D birth-KL training loss — `queued`

Extend r461 from 1D (ws0 only) to 2D (ws0, ws1 joint):
`target = empirical_birth_rates_2d[ws0_bin, ws1_bin]` with fallback to 1D marginal for
sparse buckets.  New flag `--birth-kl-loss-weight-2d`.  Expected incremental gain after
r461: 0.001–0.004.  Run after r461 4-seed results confirm positive signal.

---

### Operating notes

*Note: Ideas #27–#34 were written against the pre-Constitution PhaseAtlas era.  After
the Constitution v1.0 reset (2026-05-09), all race claims are void and all corpora show "—"
on the leaderboard.  The new race uses the MDLSTM-based generation pipeline.*

1. **Highest priority:** Run r461 master recipe on Tencent (r449-r461 all stacked, --birth-kl-loss-weight 0.25).
2. **Implemented:** r449–r461 all committed locally; push reliable via Bash/curl MCP.
3. **Next (r462):** Wider model (#46, --hidden 256 --token-embed 128) — no code change needed.
4. **Then r463:** 2D birth-KL training loss (#47, --birth-kl-loss-weight-2d).
5. **Multi-seed protocol is sacred.** No single-seed promotions.
6. **What we do not need more of:** scalar knob micro-sweeps that don't yield 4-seed claims.

