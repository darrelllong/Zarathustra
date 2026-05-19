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

### 34. Short-reuse pressure loss — `queued`

**Target:** all corpora; especially Tencent and Wikipedia where the LSTM
over-concentrates accesses at rank 0 (the most-recently-used object) compared
to the real trace.

**Why:** The CE loss treats all ranks equally.  In practice, rank-0 accesses
in real traces are followed by rank-0 again only ~10-30% of the time (depending
on corpus); the LSTM, which sees no explicit penalty for adjacent-rank 0
clustering, tends to over-predict rank 0 at transitions.  A short-reuse pressure
term penalises predicting the same rank-bin as the previous step more than
what the empirical transition matrix supports.

**Implementation sketch:**
1. During `tokenize()`, compute empirical consecutive-rank-0 rate (CRR):
   `CRR = count(rank_tokens[t]==rank_tokens[t-1]==0) / count(rank_tokens[t-1]==0)`.
2. During training, add a penalty: when the previous token was rank 0 and the
   predicted token is rank 0 AND empirical CRR < threshold, add a BCE term that
   pushes P(rank=0 | prev=rank=0) toward CRR.
3. CLI flag `--short-reuse-pressure` (float, default 0.0; recommended 1.0–3.0).

**Expected impact:** 1-3% HRC-MAE drop on corpora where adjacent-rank-0
clustering is the primary deviation.  Analogue to LANL's `--short-reuse-pressure`.

---

### 35. Stack-depth conditioning — `queued`

**Target:** all corpora; especially Meta KV and Tencent where the LRU stack
depth varies by 3-4 orders of magnitude across the trace.

**Why:** A rank bin of [0, 32] means something different when the LRU stack
has 50 objects vs 50,000 objects.  Conditioning the LSTM on stack depth lets it
distinguish the two contexts.

**Implementation sketch:**
1. During generation, track `len(stack)` as the current footprint.
2. Bin the footprint with log-spaced edges (e.g. 32 bins over [1, footprint]).
3. Add a `footprint_emb` embedding in `build_model()` and concatenate to the
   LSTM input alongside WS embeddings.
4. During training, pass footprint token computed from the prefix stack depth.
5. CLI flag `--stack-depth-bins N` (default 0 = disabled).

**Expected impact:** 2-5% drop on large-footprint corpora where the LSTM currently
conflates early-trace (small stack) and steady-state (large stack) access patterns.
Analogue to LANL r446.

---

### 36. Wider model (hidden=512 / rank-embed=128) — `queued`

**Target:** all corpora; parameter budget is not the current bottleneck but
capacity matters once birth-KL and FiLM are in place.

**Why:** R302's recommended recipe is hidden=256 (~1.4M params).  LANL's r462
sweeps hidden=256 and token-embed=128 (~4× the capacity of their default
hidden=128 baseline).  Once R302's training signal is calibrated (birth-KL +
FiLM), the next marginal improvement is expressivity.

**No code change needed.**  `--hidden 512 --rank-embed 128` on the R302 recipe.
Run only after a R302 4-seed baseline is established; otherwise confounds
architecture vs capacity attribution.

---

### Operating notes (updated 2026-05-19)

1. **Immediate next run:** R302 on Wikipedia with `--birth-kl-loss-weight 0.25
   --birth-kl-loss-weight-2d 0.10 --film-cond --rank-sampler empirical`.
   First check seed stability: if 4-seed FRESH-rate range < 10% relative, scale
   to 1M and post a Constitution-compliant claim.
2. **After Wikipedia claim:** port R302 recipe to Tencent and Alibaba using
   their respective reference CSVs.  The same architecture change should benefit
   all three storage corpora.
3. **After storage corpora:** run idea #27 (IRD-renewal port) on Meta KV,
   CloudPhysics, Wikipedia as a parallel IRD track.  The LSTM and IRD-renewal
   approaches are not mutually exclusive; post whichever gets the lower 4-seed
   mean.
4. **Biggest architectural win:** #32 (hybrid atlas + IRD-renewal).
   Once #27 is banked, this is the structural bet that has the
   highest probability of taking back Twitter/Meta CDN/Baleen24.
5. **Methodology guard:** every claim still requires four seeds with
   mean and range. Tightness of range is part of the claim — a 0.001
   range with mean 0.027 outranks a 0.005 range with mean 0.026.
6. **What we do not need more of:** new GAN-track ideas, new
   training-time scalar knobs without a structural mechanism, scout
   sweeps that don't produce a 4-seed promotion. The race is
   structural.

