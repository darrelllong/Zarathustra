# IDEAS-LLNL.md

LLNL backlog of architecture, loss, and post-hoc bets for the cachesim
HRC-MAE race. Status as of **2026-05-19**: post-Constitution restart;
no banked claims from either team. Wikipedia IRD-renewal claims (LLNL
R288.W / LANL r290) remain under audit. Active front: R302 Wikipedia LSTM
with birth-KL + FiLM + 2D birth + WS-rank sampler (wired in
`llgan/trace_lstm_ws.py`). Every idea here is judged by *expected
multi-seed mean cachesim HRC-MAE delta vs the standing LANL claim*, not
by training-time diagnostics.

Numbering continues from the closed-failed GAN-era backlog (#22–#25
addressed by current implementation, #26 still open). New ideas start
at **#27**.

Status legend: `queued` → `wired` → `running` → `closed-{positive,negative,marginal}`.

---

### 22. Hierarchical Trace Generation — `closed-marginal`

Phase / Markov / Stack-distance compound state in `markov_atlas.py`
realises this idea. R155 192-state atlas held its own on alibaba but
did not retake the corpus. Per-stream conditioning was the missing
ingredient (folded into #27, #28).

### 23. Explicit Long-Term Object Memory — `closed-marginal`

`retrieval_memory.py` (IDEA #17) wires this. Bank-saturation issues
(Gemini R3 P1 #2 / IDEA #117) addressed by `--retrieval-train-carry`.
The mechanism is structurally sound but does not move the cachesim
needle on its own; subsumed by atlas + chunk-ensemble.

### 24. Cache-Aware Training Loss — `closed-superseded`

The cachesim is now in the loop *post-hoc* via `chunk_ensemble.py` and
*as a selector* via the multi-seed claim protocol. A fully
differentiable LRU is not the bottleneck — see #29 for the
non-differentiable RL replacement.

### 25. Window-Boundary Alignment Loss — `closed-marginal`

`chunk_stitching.py` (BS + OC sub-losses) wires this. Useful guard
against cross-window drift in long rollouts; not race-decisive.

### 26. Atlas-Fit IRD-Shape Loss (R284 followup) — `queued`

Original wording preserved. Still the right idea: LLNL atlases hit a
~0.10 LRU HRC-MAE per-trace floor on alibabaBlock_521 because the
IRD distribution is approximated indirectly via state-conditioned rank
PMFs. Direct objective on IRD shape during atlas fit closes some-or-
all of the 2DIO gap. Implementation sketch unchanged. **Risk
update:** generating per-epoch eval batches is expensive; gate behind
`--atlas-fit-hrc-loss` flag and run every K epochs. **Promotion
gate:** ≥10% per-trace LRU HRC-MAE drop on v521 with corpus mean
unchanged or improved.

---

### 27. Multi-seed the IRD-renewal port across KV-shaped corpora — `wired` (R288), **highest near-term ROI**

**Targets:** Meta KV (LANL 0.0109 vs LLNL 0.05587, **80.5% gap**),
Wikipedia (0.01146 vs 0.01727, **33.6%**), CloudPhysics (0.0267 vs
0.0311, **14.1%**).

**Why this is the breakthrough candidate:** R286 LLNL port was broken
(used LRU stack distances → 0.20 MAE floor). R288 rewrote `llgan/ird_renewal.py`
with correct position-based IRD + heap renewal scheduler — the right algorithm.
`llgan/launch_ird_renewal_multiseed.py` is the sweep launcher. Sweep pending
compute on vinge.

**Implementation sketch (updated for R288):**
1. Run `python3 -m llgan.launch_ird_renewal_multiseed --real REF
   --corpus wiki --seeds 42,80,81,82
   --spec "s32_ip10:ird_s=32.0,ip=0.10"
   --spec "rb32_s32:ird_s=32.0,ip=0.10,rb=32"
   --spec "rb32_s32_sm:ird_s=32.0,ip=0.10,rb=32,smooth=1"
   --spec "rb16_ps:ird_s=32.0,ip=0.10,rb=16,ps=1"` on Wikipedia
   ref. The 4×4×4×4 = 256-cell × 4-seed = 1024 evals is heavy; cut to
   the LANL-published cell first (verify reproduction within range
   0.000533) before sweeping.
2. Repeat on `meta_kv_real.csv` and `cloudphysics_real.csv`. CP
   has the 8-policy surface — use it.
3. Post 4-seed tables to `RESPONSE-LLNL.md`.

**Expected impact:** Meta KV 0.05587 → ~0.011 (ports LANL recipe
verbatim). Wikipedia 0.01727 → ~0.0114. CloudPhysics 0.0311 → ~0.027.
**Three retakes from one idea.**

**Risk:** LANL's CloudPhysics seed-80 is an outlier (0.0295 dragging
mean to 0.0267). LLNL might match or beat the mean while landing in
the same high-variance basin. Mitigation: probe `rank_ird_buckets ∈
{48,64,96}` to reduce the seed-80-class outlier and *land below 0.027
with range < 0.002* — that is a clean retake on tighter uncertainty.

---

### 28. Per-stream IRD calibration with multi-stream references — `queued`

**Targets:** Wikipedia, Twitter, MSR Exchange — corpora whose real
reference has distinct multi-stream rows.

**Why:** Both LANL's IRD-renewal and LLNL's neural atlas pool all
streams into a single empirical IRD. For corpora with heterogeneous
streams (heavy-write tenant vs read-only tenant; CDN edge vs origin),
per-stream IRD shapes diverge by 2-5× in tail. Pooling underfits the
heavy stream and overfits the light one.

**Implementation sketch:**
1. Detect multi-stream structure in the reference CSV (`stream_id`
   column or implicit per-tenant grouping).
2. For each stream, fit an independent IRD distribution and an
   independent rank-conditioned IRD if `rank_ird_buckets > 0`.
3. At generation, sample which stream produced the next event from a
   stream weight vector (or honour `n-streams` parallel rollouts), and
   draw IRD from that stream's distribution.
4. Add `--per-stream` flag to both `ird_renewal.py` and
   `neural_atlas.py`.

**Expected impact:** Closes the structural gap on heterogeneous
corpora. Wikipedia (LANL global-renewal best, but heterogeneous
multi-stream) is the strongest candidate: ~10-15% drop expected, which
**moves the row to LLNL's column.**

**Risk:** Single-stream corpora won't benefit; flag must default off.
LANL has not published per-stream variants on Wikipedia, so there is
free space.

---

### 29. Black-box cachesim optimisation in chunk-ensemble — `queued`

**Targets:** Alibaba (4.7% gap), Tencent (1.3% gap), Twitter (13.1%
gap) — all already won by LANL via chunk-surface selectors.

**Why:** The current `chunk_ensemble.py` greedy guard pass evaluates
every donor at every chunk and keeps the best. With a 6-policy
cachesim call per candidate, this is O(donors × chunks × policies ×
caps). It picks the *one-step locally optimal* swap — provably worse
than coordinate descent, let alone CMA-ES or RL.

**Implementation sketch:**
1. Replace the greedy pass with two-stage optimisation:
   a. **Stage 1 (CMA-ES, 1024 evals):** parameterise the swap policy
      as a soft selector — donor weights as Dirichlet logits; let
      CMA-ES tune the weights against cachesim mean HRC-MAE on a
      held-out 200k validation slice of the real ref.
   b. **Stage 2 (greedy refine, 100 chunks):** apply current greedy
      pass over the CMA-ES initialisation.
2. Cache `cachesim` calls keyed on `(chunk_hash, policy, cap)` — 95%
   of evals across the search are repeats.
3. Score on the **mean over a subset of seed indices** so per-seed
   overfitting doesn't sneak in.

**Expected impact:** 5-10% additional drop on the chunk-ensemble
contribution alone, on top of whatever the base atlas gives. Combined
with #27/#28, retakes Alibaba and Tencent outright (and probably
Twitter once paired with the right atlas base).

**Risk:** Hyperparameter search overfits the validation slice. Hold
out 4 fresh seeds (46/47/48/49) for the *promotion test* — never let
CMA-ES see them.

---

### 30. Workload-class router (atlas vs IRD-renewal) — `queued`

**Why:** The LANL leaderboard exposes a clear architectural split.
Storage corpora (Alibaba, Tencent, MSR Exchange, Baleen24) are best
served by phase-conditioned atlases with LRU stack decoders. KV/CDN
corpora (CloudPhysics, Wikipedia, Meta KV, Meta CDN) are best served
by IRD-renewal because their HRC has cliff structure that scale-
sharpened atlases over-concentrate. Twitter sits between.

**Implementation sketch:**
1. Compute four corpus descriptors offline: median IRD, IRD coefficient
   of variation, rank=0 mass, and reuse-vs-new ratio.
2. Train a 2-class classifier (storage vs KV) on the four descriptors
   from labelled corpora (Alibaba/Tencent/MSR storage; Wiki/MetaKV/CP
   KV; Twitter, Baleen24 unlabeled tertiary).
3. At generation time, predict class and route to neural-atlas or
   ird-renewal pipeline.
4. Bonus: emit a `corpus_type` field in the manifest header so the
   tooling can pick the right path automatically.

**Expected impact:** Reduces "wrong recipe on wrong corpus" errors.
The Twitter and Baleen24 retakes likely fall to whichever pipeline the
classifier assigns. Removes a class of "we used the storage recipe on
a KV corpus" failures that have appeared in the round logs.

**Risk:** Two-class is too coarse for tertiary corpora. Mitigation:
ablate by training a 3-class with a "blend" tier, or expose a
`--blend-fraction` knob that mixes both pipelines.

---

### 31. Per-policy decoder ensemble — `queued`

**Targets:** all corpora; specifically the cases where the
multi-policy mean HRC-MAE hides a lopsided per-policy profile.

**Why:** A generator that wins LRU/ARC by 30% but loses SIEVE by 50%
can have the same mean HRC-MAE as one that's mediocre on all six. The
mean is the race metric, but it hides which locality structure each
generator gets right. SIEVE rewards adjacent-duplicate sequencing;
ARC rewards balance between recency and frequency; SLRU rewards stable
hot-set; CAR rewards CLOCK-bit dynamics. **Different policies require
different generation knobs.**

**Implementation sketch:**
1. Compute per-policy HRC-MAE for each candidate and identify which
   policies dominate the mean for each corpus.
2. For the worst policy, fit a *post-hoc decoder pass* tuned for that
   policy's invariants:
   - SIEVE → adjacency injector (`stack_adj_dup_prob` swept against
     SIEVE only).
   - LRU → IRD shape (the standard knob).
   - ARC → recency/frequency balance (vary tail-reuse vs hot-pool
     mix).
3. Ensemble: generate K=3 candidates with different per-policy
   tunings, then pick per-chunk the one with lowest cachesim mean on
   that chunk's slice.

**Expected impact:** 3-7% mean drop, drawn from policies that
contribute most to the current MAE. Especially useful where LANL's
chunk-ensemble has the lead but per-policy spread is wide (Twitter,
Baleen24).

**Risk:** Over-fits to the policy-specific heuristics; multi-seed
range may grow. Mitigation: only ensemble when per-seed range is
already tight; abandon if range exceeds 0.001.

---

### 32. Hybrid atlas + IRD-renewal generator — `queued`

**Targets:** all 9 corpora; especially the "neither pure atlas nor
pure IRD-renewal wins" middle ground (Twitter, Meta CDN, Baleen24).

**Why:** The neural atlas decides *action class* (NEW vs REUSE) well
because it conditions on the per-file profile, but its rank PMF over-
concentrates. IRD-renewal nails the IRD shape but ignores per-file
conditioning. The natural composition: **atlas decides the action
class; IRD-renewal decides the rank.**

**Implementation sketch:**
1. At each step, run the trained `neural_atlas` to sample
   `next_state ∈ {NEW, REUSE_b1..REUSE_b5}`.
2. If NEW, allocate a fresh ID via the existing path.
3. If REUSE, **ignore** the atlas's per-state rank PMF; instead, pull
   an IRD draw from the per-state empirical IRD distribution and read
   the LRU stack at that rank.
4. Keep the 4-seed protocol; ablate against pure-atlas and pure-IRD
   on each corpus.

**Expected impact:** The hybrid combines corpus-generalization (atlas
strength) with IRD fidelity (renewal strength). On Twitter and
Baleen24 — where neither pure pipeline holds the lead — this should
land in the LANL column or beat it. On the IRD-decisive corpora the
hybrid should at minimum match #27.

**Risk:** Per-state IRD distributions in the training corpus have
small support for the rare states (REUSE_b5). Mitigation: smooth with
a Dirichlet prior or fall back to the marginal IRD when state support
< 100 examples.

---

### 33. Direct IRD parametrisation à la 2DIO — `queued`, *2DIO comparison class*

**Target:** the 2DIO bar (per-trace HRC-MAE 0.02–0.05) on
alibabaBlock_521 and the four CloudPhysics traces named in
`LEADER-BOARD.md`. LLNL's atlas hits a 0.10 floor on v521 regardless
of capacity (R284.X 192 states; R284.Y 6 states). **Capacity is not
the bottleneck.**

**Why:** 2DIO directly parameterises the IRD distribution with a
3-parameter analytical form (heavy-tail mixture). LLNL's atlas
approximates that shape *indirectly* through state-conditioned rank
PMFs and pays a 2–5× MAE penalty for the indirection.

**Implementation sketch:**
1. New module `llgan/two_dio.py`. Per-trace fit of a 3-parameter IRD
   model (Pareto + log-normal mixture; or whatever `paper/2DIO.pdf`
   describes — re-derive, do not hand-copy).
2. Per-trace generation: roll out via IRD draws from the fitted
   parameters; LRU stack at that rank.
3. Multi-trace evaluation: report per-trace LRU HRC-MAE on the five
   benchmark traces (v521, w11/w24/w44/w82) under the same conditions
   2DIO does.

**Expected impact:** First LLNL number in the 2DIO comparison class.
Even matching 0.05 (the looser 2DIO bound) is a paper. Beating 0.02
on any of the five traces is **a publishable result** and sets up an
external positioning for the next NSF cycle.

**Risk:** This is a different metric class from the corpus race;
spending compute here trades against retaking the 9 corpora. Gate
behind explicit user request; do not autoschedule.

---

### 34. Short-reuse class weighting — `closed-positive (R303)`

**Target:** all corpora; especially Tencent and Wikipedia.

**Why:** The CE loss treats all rank bins equally.  Short-reuse ranks (< ws_windows[0]=32)
dominate HRC-MAE on small caches (32–512 entries in the official surface).  Amplifying
the CE gradient for these bins directly pressures the LSTM to get the most cache-relevant
access patterns right.

**R303 implementation:** `_compute_short_reuse_class_weights()` assigns weight
`1 + gain` to rank bins with midpoint < primary window (=32), tapering to 1.0 at
secondary window (=128), passed via `F.cross_entropy(..., weight=class_weights_t)`.
CLI: `--short-reuse-loss-weight` (float, default 0.0; recommended 1.0–3.0).

**Note:** LANL's generation-time `_apply_short_reuse_pressure` (dynamic WS-feedback
controller that biases toward short-reuse when current_ws > target_ws) is a complementary
idea not yet ported.  Tracked as idea #37.

---

### 35. Stack-depth conditioning — `wired (R304)`

**Target:** all corpora; especially Meta KV and Tencent where the LRU stack
depth varies by 3-4 orders of magnitude across the trace.

**Why:** A rank bin of [0, 32] means something different when the LRU stack
has 50 objects vs 50,000 objects.  Conditioning the LSTM on stack depth lets it
distinguish the two contexts.

**R304 implementation:** `--stack-depth-bins N` (default 0 = disabled; try 32).
During tokenize, `fp_tokens[t]` = bin of running unique-object count before t.
`build_model()` adds `fp_emb: Embedding(n_fp_bins, ws_embed)` concatenated to
LSTM input; FiLM uses the augmented context.  Generation tracks `running_fp_set`
and bins `len(set)` at each step.  Analogue to LANL r446.

**Expected impact:** 2-5% drop on large-footprint corpora where the LSTM currently
conflates early-trace (small stack) and steady-state (large stack) access patterns.

---

### 36. Wider model (hidden=512 / rank-embed=128) — `queued`

**Target:** all corpora; parameter budget is not the current bottleneck but
capacity matters once R303's training signal is calibrated.

**Why:** R303's recommended recipe is hidden=256 (~1.5M params with WS head).
LANL's r462 sweeps hidden=256 token-embed=128.  Once a 4-seed R303 baseline is
established, the next marginal improvement is expressivity.

**No code change needed.**  `--hidden 512 --rank-embed 128` on the R303 recipe.
Run only after a R303 4-seed baseline is established.

---

### 37. Generation-time WS-feedback pressure (short-reuse pressure) — `queued`

**Target:** all corpora.  Complement to idea #34 (training-time class weights).

**Why:** LANL's `_apply_short_reuse_pressure()` is a dynamic WS-feedback controller
at generation time: when current_ws > target_ws (WS has grown beyond what the empirical
WS-conditional distribution predicts), bias probabilities toward short-reuse rank bins
(rank < primary=32) and away from long-reuse bins.  This prevents WS runaway — the
positive feedback loop where the model generates too many FRESH tokens, inflating the
WS, causing it to generate even more FRESH tokens.

**R304 implementation:** `--short-reuse-pressure GAIN` (float, default 0.0; try 1.0–3.0).
After computing probs[], compares `probs[NEW_TOKEN]` to `birth_rate_by_ws0[ws0_bin]`.
If surplus > 0 (model predicts more fresh than expected), applies log-weight bonus
to rank bins with midpoint < primary=32.  Applied before birth-rate blend.
Zero-refit: works on any R303+ checkpoint.  Analogue to LANL's short_reuse_pressure.

**Expected impact:** 2-5% HRC-MAE drop on corpora with WS runaway (large seed-to-seed
variance in FRESH rate).  Directly addresses the root cause of R301's seed=80 failure.

---

### 38. Cache-ladder rank vocabulary alignment — `wired (R304)`

**Target:** all corpora.

**Why:** Log-spaced rank bins may straddle cache evaluation boundaries [32, 128,
512, 2048, 8192], causing draws to land on the wrong side of the HRC evaluation.
This is the same structural floor LANL identified in r448.

**R304 implementation:** `--cache-ladder` flag + `--ladder-sizes 32,128,512,2048,8192`.
Injects mandatory edges via `np.unique(concatenate([log_edges, mandatory]))` in
`make_rank_bins()`.  Also `--ws-cache-ladder` for WS bin edges.

---

### 40. Delta-WS conditioned empirical token blend — `wired (R305)`, *ahead of LANL #48*

**Target:** all corpora.

**Why:** Two WS states with identical ws0=W but opposite trajectories (rising vs falling)
have materially different rank distributions.  Rising WS → more fresh tokens and deep reuse.
Falling WS → concentrated shallow reuse.  LANL identified this as idea #48 but hadn't
implemented it yet (code needed).  LLNL implements it first.

**R305 implementation:** `--ws-token-blend-delta α` (default 0.0; try 0.3).
3D table `rank_token_freq_table_delta[ws0_bin][delta_sign][rank_token]` where
`delta_sign ∈ {0=falling, 1=stable, 2=rising}`.  Applied after 2D blend and before
short-reuse pressure.  Zero-refit; sparse bins fall back to 1D table automatically.

---

### 41. Constitution-compliant multiseed eval pipeline — `wired (R306)`

**Target:** all corpora.

**Why:** Article VI requires 4-seed mean + range + literal per-seed cachesim lines before
any claim lands on the leader-board. Previously this required running fit / generate /
cachesim as three separate manual steps.  Missing automation was the only thing blocking
a scored claim after R305 code landed.

**R306 implementation:** `multiseed` subcommand in `llgan/trace_lstm_ws.py`. Accepts all
fit and generation flags; optionally trains (`--fit`), then generates one CSV per seed,
runs `llgan.cachesim_eval.evaluate()` against the real reference, aggregates to 4-seed
mean + range, and appends a formatted markdown panel to `--append-markdown` plus a JSON
report to `--json-out`.  Single command → complete Constitution-compliant claim evidence.

---

### 39. Generation-time WS-conditioned empirical rank blend — `wired (R304)`

**Target:** all corpora.

**Why:** The LSTM's CE training does not guarantee calibrated P(rank|ws0).
Blending with the empirical distribution at generation time anchors rank
predictions without refit.  Zero-refit on any R303+ checkpoint.

**R304 implementation:**
- `--ws-token-blend α` (default 0.0; try 0.5): blends LSTM probs with
  `rank_token_freq_table[ws0_bin]` stored in checkpoint.
- `--ws-token-blend-2d α₂` (default 0.0; try 0.25): further blends with
  2D table `rank_token_freq_table_2d[ws0_bin, ws1_bin]`.
- `--ws-blend-confidence-tau τ` (default 0.0; try 50): scales α by
  `sqrt(bucket_count/τ)` to trust sparse bins less.
Applied before birth-rate blend.  Analogue to LANL r449/r450/r452.

---

### Operating notes (updated 2026-05-21, R306)

1. **Immediate next run:** R306 `multiseed` — single command fits + generates + evals:
   ```
   python3 -m llgan.trace_lstm_ws multiseed \
     --real $WIKI_REF \
     --fit \
     --film-cond --dropout 0.1 \
     --birth-kl-loss-weight 0.25 --birth-kl-loss-weight-2d 0.10 \
     --ws-kl-loss-weight 0.25 --aux-ws-loss-weight 0.1 \
     --short-reuse-loss-weight 1.0 --rank-sampler empirical \
     --cache-ladder --ws-cache-ladder --stack-depth-bins 32 \
     --label-smoothing 0.05 --grad-clip 1.0 --lr-schedule cosine \
     --lstm-layers 3 --epochs 25 \
     --seeds 42,80,81,82 --n 1000000 \
     --ws-token-blend 0.5 --ws-token-blend-2d 0.25 --ws-blend-confidence-tau 50 \
     --ws-token-blend-delta 0.3 \
     --birth-rate-blend 0.5 --birth-rate-blend-2d 0.25 \
     --short-reuse-pressure 2.0 \
     --tag wiki_r306 --outdir /tmp/r306 \
     --append-markdown VERSIONS-LLNL.md --json-out /tmp/r306/wiki_r306.json
   ```
   Target: 4-seed mean < 0.0115 to beat LANL r290 Wikipedia (AUDIT-PENDING).
2. **Generation-only sweep on any existing R303/R304/R305 checkpoint:** `--ws-token-blend`,
   `--ws-token-blend-delta`, and `--short-reuse-pressure` are all zero-refit.
   Use the `multiseed` subcommand with `--model` (no `--fit`) to get a Constitution
   claim panel without retraining.
3. **After Wikipedia claim:** port R306 recipe to Tencent and Alibaba.
4. **Next architectural ideas:** #32 (hybrid atlas + IRD-renewal — highest structural ROI),
   #36 (wider model hidden=512 after first 4-seed baseline), #29 (black-box cachesim
   optimization in chunk-ensemble).
5. **Methodology guard:** every claim still requires four seeds with
   mean and range. Tightness of range is part of the claim — a 0.001
   range with mean 0.027 outranks a 0.005 range with mean 0.026.
6. **What we do not need more of:** new GAN-track ideas, new
   training-time scalar knobs without a structural mechanism, scout
   sweeps that don't produce a 4-seed promotion. The race is
   structural.

