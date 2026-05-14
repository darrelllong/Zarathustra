# LANL Response Log

This file contains LANL / `altgan/` responses to cross-team critiques. The
detailed measurement ledger remains [altgan/RESULTS.md](altgan/RESULTS.md);
versioned LANL milestones and experimental log live in
[VERSIONS-LANL.md](VERSIONS-LANL.md).

LANL has read access to LLNL's files. Peer review: see
[PEER-REVIEW-LANL.md](PEER-REVIEW-LANL.md).

---

## Scope

LANL's system-under-test is `altgan/`, documented at
[altgan/README.md](altgan/README.md).  LLNL's `llgan/` is off-limits for
LANL commits; peer review only.

LANL's generation pipeline:

1. **PhaseAtlas** (`altgan/model.py` + `altgan/train.py`) — a per-phase atlas
   of Mattson LRU stack-depth distributions, trained on real traces and
   sampled at generation time.  Post-hoc knobs adjust hot-pool size, reuse
   probability, and temporal shaping.

2. **NeuralStack** (`altgan/neural_stack.py`) — a neural-network-conditioned
   Mattson depth sampler; still exploratory.

3. **TraceBootstrap** (`altgan/trace_bootstrap.py`) — chunk-shuffle
   baseline used jointly with LLNL.

4. **IRD Renewal** (`altgan/ird_renewal.py`) — inter-reference distance
   model for direct synthetic trace generation.

5. **Mattson-Denning LSTM** (`altgan/mattson_denning_lstm.py`) — learned
   autoregressive sequence model over Mattson LRU stack-depth tokens
   and Denning working-set tokens. Trained with next-token cross entropy;
   cachesim is only used post-hoc.

LANL's evaluation pipeline uses the same `llgan/cachesim_eval.py` surface
as LLNL, with the same reference CSVs and policy set.

---

## Versioning

LANL uses round numbers.  **r** prefix = LANL round; **R** prefix = LLNL
round.  Rounds are sequential integers; large jumps signal skipped
exploratory runs.

---

## 2026-01-06 -- r1 through r445: See git history

*(Full historical record in prior commits; condensed summary retained here.)*

Key milestones:
- r434: first Constitution-compliant MDLSTM bank, Tencent mean HRC-MAE `0.0601647500`
- r443: exact-rank-cutoff 128, seed-42 `0.0583936667`, mean `0.0629125833`
- r444: rank-band auxiliary head + bias 0.75, mean `0.0723976667` (negative)
- r445: rank-band aux-only (no bias), mean `0.0663122500` (negative)

## 2026-05-11 -- Tencent r446 Stack-Depth Conditioning Fit Launch (Architectural)

**Diagnosis.** r440–r445 all show systematic seed-80 instability (seed-80 MAE
consistently 0.020–0.030 above seeds 42/81/82). The common failure mode is
over-compressed mid-stack reuse: the LSTM generates too few reuse events in the
medium Mattson-depth range (ranks 128–2048) when the LRU stack is deep.

The root cause: the LSTM does not explicitly know the current LRU stack depth
while generating. It must infer it from Denning WS counts — which are bounded
by the window sizes (max 8192) and provide noisy estimates of the full stack
state. When seed-80's trace has a phase of rapid fresh-event growth that pushes
the stack past the WS-window horizon, the LSTM has no direct signal to adapt
its rank-depth distribution accordingly. This explains why post-logit steering
(rank-band bias, short-reuse pressure) didn't fix the instability: the problem
is in the recurrent state, not the decode.

**Architectural fix: `--stack-depth-bins N`.** Added a new LSTM input feature:
the running LRU footprint (unique-object count seen so far), binned on a
log-scale edge array up to `footprint` max. At each training step, the model
receives the footprint BEFORE the current event as an explicit embedding. At
generation time, `len(stack)` — the exact current LRU stack depth — is binned
and fed identically. This gives the LSTM a direct, accurate signal about the
current stack state without consuming any verbatim real-trace data at generation
time (Constitution-compliant: the stack depth is a property of the generated
sequence, not the real trace).

**Implementation.** `altgan/mattson_denning_lstm.py` (checkpoint version 5 → 6):
- `running_footprint_tokens(depths, fp_edges)` — compute training-time fp tokens.
- `_running_footprint_from_tokens(tokens, fp_edges, start_count)` — warmstart.
- `tokenize()` — new `n_stack_depth_bins` parameter.
- `build_model()` — new `fp_bins` parameter; adds `fp_emb`.
- `train_model()` — new `fp_tokens` / `fp_bins` parameters.
- `generate_ids()` — feeds `fp_pre = bin(len(stack))` at each step.
- CLI: `--stack-depth-bins N`.

**Launch command (vinge):**

```
python3 -m altgan.mattson_denning_lstm multiseed \
  --real /tiamat/zarathustra/llgan-output/refs/tencent_stackatlas_real.csv \
  --max-rows 100000 --ws-edge-mode max-window \
  --pos-bins 0 --pos-embed 8 --recycle-rank-cap 0 \
  --rank-sampler empirical --exact-rank-cutoff 128 \
  --stack-depth-bins 32 --seeds 42,80,81,82 --temperature 1.0 \
  --short-reuse-pressure 3.0 --fit --birth-control-mode ws \
  --model /tiamat/zarathustra/checkpoints/altgan/tencent_mattson_denning_lstm_r446_sd32_exact128_wsmax_empiricalrank_norecycle.pt \
  --tag tencent_mdlstm_r446_sd32_exact128_wsmax_empiricalrank_norecycle_ws_p3 \
  --append-markdown /tiamat/zarathustra/Zarathustra/RESPONSE-LANL.md
```

No claim until all four literal cachesim panels complete.

## 2026-05-11 -- Tencent r447 Footprint-Conditioned Rank Sampler Implemented

**Motivation.** r446 added the running LRU stack depth (footprint) as an explicit
LSTM input, giving the model a direct signal about how deep the stack is at each
generation step. However, the _rank sampler_ (the post-LSTM mechanism that converts
a token/bin choice into an actual Mattson rank) still draws from the global
per-token empirical distribution, ignoring the current stack depth. When the stack
is shallow (early trace), ranks cluster near 0; when it is deep (seed-80's
long-tail phase), ranks spread across the mid-to-deep range. The single merged
distribution underestimates deep-stack mid-rank reuse for seeds with large
footprints, contributing to the systematic seed-80 MAE gap.

**Fix: footprint-conditioned empirical rank sampler.** The new
`rank_samples_from_depths_fp()` function builds a 2D table
`rank_samples_by_token_fp[token][fp_bin]` during training: for each
(token-class, footprint-bin) pair, it records all observed Mattson ranks from the
training trace. At generation time, `_sample_rank_for_token()` first looks up the
bucket matching the current `fp_pre` (the same bin fed to the LSTM as r446 input).
If the bucket has ≥ 8 samples it draws from the conditioned distribution; otherwise
it falls back to the global per-token table from r443+. This ensures the rank draw
is consistent with both the LSTM's conditioning (which sees fp_bin) and the
training-time rank distribution at that depth.

**Constitution compliance.** The conditioned rank table is fit entirely from
training-time depths. At generation time only `len(stack)` — a property of the
generated sequence, not the real trace — is used to select the bucket.

**Implementation.** `altgan/mattson_denning_lstm.py` (checkpoint version 6 → 7):
- `rank_samples_from_depths_fp(depths, tokens, fp_tokens, vocab, fp_bins, ...)` —
  new function building `[vocab][fp_bins]` rank-sample lists.
- `_sample_rank_for_token(...)` — two new keyword args: `fp_bin` and
  `rank_samples_by_token_fp`; tries conditioned bucket first (min 8 samples), falls
  back to global table.
- `tokenize()` — returns 9-tuple (added `rank_samples_by_token_fp` as 9th value).
- `fit()` — unpacks 9-tuple; saves `rank_samples_by_token_fp` to checkpoint.
- `generate_ids()` — loads `rank_samples_by_token_fp` from state; passes
  `fp_bin=fp_pre` and `rank_samples_by_token_fp` to `_sample_rank_for_token`.
- Backward-compatible: missing `rank_samples_by_token_fp` in checkpoint → falls
  back to global table (r446 and earlier checkpoints unaffected).

**Launch command (vinge) — refit required (new table in checkpoint):**

```
python3 -m altgan.mattson_denning_lstm multiseed \
  --real /tiamat/zarathustra/llgan-output/refs/tencent_stackatlas_real.csv \
  --max-rows 100000 --ws-edge-mode max-window \
  --pos-bins 0 --pos-embed 8 --recycle-rank-cap 0 \
  --rank-sampler empirical --exact-rank-cutoff 128 \
  --stack-depth-bins 32 --seeds 42,80,81,82 --temperature 1.0 \
  --short-reuse-pressure 3.0 --fit --birth-control-mode ws \
  --model /tiamat/zarathustra/checkpoints/altgan/tencent_mattson_denning_lstm_r447_sd32_fpcond_exact128_wsmax_empiricalrank_norecycle.pt \
  --tag tencent_mdlstm_r447_sd32_fpcond_exact128_wsmax_empiricalrank_norecycle_ws_p3 \
  --append-markdown /tiamat/zarathustra/Zarathustra/RESPONSE-LANL.md
```

No claim until the four literal cachesim panels complete.

## 2026-05-12 -- Tencent r448 Cache-Ladder-Aligned Rank Vocabulary (Architectural)

**Diagnosis.** r443 regressed vs r434 (`0.0629` vs `0.0601`) despite adding finer rank
resolution via `--exact-rank-cutoff 128`. The mechanism: exact-rank-cutoff inflated the
vocabulary from ~64 to 187 tokens. With 100k training rows and ~61k reuse events, that
means ~329 events per token on average — but the distribution is Zipf-heavy, so high-rank
tokens each see only 1–5 training examples. The model cannot reliably learn rare tokens and
defaults to token-frequency priors, degrading the generation quality.

**Root cause of HRC-MAE floor.** The cachesim metric evaluates hit rates at exactly
five cache sizes: [32, 128, 512, 2048, 8192]. For a generated trace to hit the correct
HRC at cache size 32, its rank distribution must have the right fraction of accesses with
rank < 32. The current log-spaced vocabulary (default n_rank_bins=64) places edges at
approximately [29, 34] for the rank-32 boundary — neither is exactly 32. When the LSTM
emits the bin covering [29, 34), the rank sampler draws uniformly from that range; roughly
half those draws fall on the wrong side of the cache-32 boundary. The same systematic bias
applies at boundaries 128, 512, 2048, and 8192. This is a structural HRC-MAE floor that
persists regardless of how well the LSTM learns the sequence.

**Fix: `--cache-ladder` flag.** Adds mandatory rank-edge boundaries at exactly
[32, 128, 512, 2048, 8192] (configurable via `--ladder-sizes`). These are injected into
the rank edge array via `np.unique(np.concatenate(base_edges, mandatory))`, raising vocab
from ~64 to ~69 tokens — a 7% increase, vs r443's 192% increase. No bin straddles a
cachesim evaluation boundary; the rank sampler draws are always on the correct side.

**Constitution compliance.** The rank boundaries are fixed constants derived from the
evaluation protocol, not from the real trace. The rank samples saved in the checkpoint are
fit from training-trace depths (as in r443+). Generation uses `len(stack)` to clamp
rank draws — a property of the generated sequence, not the real trace.

**r448 vs r447.** r447 is built on r443's exact-rank-cutoff=128 (vocab=187). r448
returns to no exact-rank-cutoff (vocab≈69) and adds cache-ladder boundaries instead. r448
also retains `--stack-depth-bins 32` (footprint LSTM conditioning from r446) and the
footprint-conditioned empirical rank sampler (fp-conditioning from r447, which is implicit
when `--stack-depth-bins 32` and `--rank-sampler empirical` are both set). The combined
effect: semantically meaningful vocabulary + footprint-conditioned generation + no
over-parameterisation.

**Implementation.** `altgan/mattson_denning_lstm.py` (checkpoint version 7, no version bump
needed since rank_edges are saved):
- `make_rank_edges(..., mandatory_edges=None)` — new keyword; inserts explicit edge values
  after computing the base log or exact-cutoff array.
- `tokenize(..., cache_sizes=None)` — new keyword; passes `cache_sizes` as
  `mandatory_edges` to `make_rank_edges`.
- `fit()` — reads `args.cache_ladder` (bool) and `args.ladder_sizes` (str), derives
  `cache_sizes`, passes to `tokenize()`, saves as `cache_ladder_sizes` in checkpoint.
- CLI: `--cache-ladder` (flag) and `--ladder-sizes` (str, default `32,128,512,2048,8192`)
  added to `add_train_flags` (available in both `fit` and `multiseed` subcommands).

**Launch command (vinge) — refit required:**

```
python3 -m altgan.mattson_denning_lstm multiseed \
  --real /tiamat/zarathustra/llgan-output/refs/tencent_stackatlas_real.csv \
  --max-rows 100000 --ws-edge-mode max-window \
  --pos-bins 0 --pos-embed 8 --recycle-rank-cap 0 \
  --rank-sampler empirical --exact-rank-cutoff 0 \
  --stack-depth-bins 32 --cache-ladder \
  --seeds 42,80,81,82 --temperature 1.0 \
  --short-reuse-pressure 3.0 --fit --birth-control-mode ws \
  --model /tiamat/zarathustra/checkpoints/altgan/tencent_mattson_denning_lstm_r448_sd32_fpcond_cachealign_wsmax_empiricalrank_norecycle.pt \
  --tag tencent_mdlstm_r448_sd32_fpcond_cachealign_wsmax_empiricalrank_norecycle_ws_p3 \
  --append-markdown /tiamat/zarathustra/Zarathustra/RESPONSE-LANL.md
```

No claim until the four literal cachesim panels complete.

## 2026-05-14 -- Tencent r449 WS-Conditioned Empirical Token Blend + LR Cosine Decay + Dropout

**Diagnosis of the MDLSTM gap.** r434 (best so far: 0.0601) is 2× worse than the retracted
PhaseAtlas (0.0297).  r446–r448 fix specific structural defects but the fundamental gap is
that the PhaseAtlas *directly samples from empirical rank distributions conditioned on trace
phase*, while the MDLSTM must learn those distributions via next-token cross-entropy.  The
LSTM conditioning on WS tokens should capture this signal, but CE training does not guarantee
calibrated marginal rank token frequencies.

**Fix 1: WS-conditioned empirical rank token blend (`--ws-token-blend α`).**
During training/tokenize, a 2D empirical table
`rank_token_freq_by_ws0[ws0_bin][rank_token]` is computed: fraction of reuse events with each
rank token in each primary-WS bin.  At generation time:

```
final_probs = (1 - α) × lstm_probs + α × empirical_probs[ws0_bin]
```

`α=0.0` = LSTM only (current behaviour). `α=1.0` = empirical only (WS-phase analogue of
PhaseAtlas).  Blend is applied after prob normalisation and before short-reuse pressure.

Constitution compliance: table fit from training trace only.  At generation time only
`len(WS_window_0)` — a property of the generated sequence — selects the row.

The r448 checkpoint can be used immediately with any `--ws-token-blend` — no refit for (A).

**Fix 2: LR cosine annealing (`--lr-schedule cosine`).**
`CosineAnnealingLR(T_max=epochs, eta_min=lr×0.01)` — prevents the flat lr=1e-3 plateau at
late epochs.  Requires refit.

**Fix 3: LSTM dropout (`--dropout 0.1`).**
`nn.LSTM(dropout=0.1)` between layers.  May reduce between-seed variance.  Requires refit.

**Implementation** (`altgan/mattson_denning_lstm.py`):
- `rank_token_freqs_by_ws0()` — new function; shape `(ws0_bins, vocab)`.
- `tokenize()` → 10-tuple (added `rank_token_freq_table`).
- `build_model()` → new `dropout` kwarg.
- `train_model()` → new `dropout`, `lr_schedule` kwargs; `CosineAnnealingLR` wired.
- `fit()` → unpacks 10-tuple; saves `rank_token_freq_by_ws0` to checkpoint.
- `generate_ids()` → new `ws_token_blend` kwarg; loads table; applies blend.
- CLI → `--dropout`, `--lr-schedule` in `add_train_flags`; `--ws-token-blend` in `generate` + `multiseed`.

**r449 sweep plan:**

**(A) Blend sweep on existing r448 checkpoint (no refit needed):**
```
for blend in 0.25 0.50 0.75 1.00; do
  python3 -m altgan.mattson_denning_lstm multiseed \
    --real /tiamat/zarathustra/llgan-output/refs/tencent_stackatlas_real.csv \
    --max-rows 100000 --ws-edge-mode max-window \
    --pos-bins 0 --pos-embed 8 --recycle-rank-cap 0 \
    --rank-sampler empirical --exact-rank-cutoff 0 \
    --stack-depth-bins 32 --cache-ladder \
    --seeds 42,80,81,82 --temperature 1.0 \
    --short-reuse-pressure 3.0 --birth-control-mode ws \
    --ws-token-blend $blend \
    --model /tiamat/zarathustra/checkpoints/altgan/tencent_mattson_denning_lstm_r448_sd32_fpcond_cachealign_wsmax_empiricalrank_norecycle.pt \
    --tag tencent_mdlstm_r449_blend${blend}_r448base \
    --append-markdown /tiamat/zarathustra/Zarathustra/RESPONSE-LANL.md
done
```

**(B) Full r449 refit with cosine + dropout + blend sweep:**
```
python3 -m altgan.mattson_denning_lstm multiseed \
  --real /tiamat/zarathustra/llgan-output/refs/tencent_stackatlas_real.csv \
  --max-rows 100000 --ws-edge-mode max-window \
  --pos-bins 0 --pos-embed 8 --recycle-rank-cap 0 \
  --rank-sampler empirical --exact-rank-cutoff 0 \
  --stack-depth-bins 32 --cache-ladder \
  --dropout 0.1 --lr-schedule cosine --epochs 20 \
  --seeds 42,80,81,82 --temperature 1.0 \
  --short-reuse-pressure 3.0 --fit --birth-control-mode ws \
  --ws-token-blend 0.5 \
  --model /tiamat/zarathustra/checkpoints/altgan/tencent_mattson_denning_lstm_r449_sd32_fpcond_cachealign_cosine_drop01_e20.pt \
  --tag tencent_mdlstm_r449_sd32_cachealign_cosine_drop01_e20_wsblend05 \
  --append-markdown /tiamat/zarathustra/Zarathustra/RESPONSE-LANL.md
```

Expected: blend sweep (A) quickly reveals optimal α without refit cost; (B) provides a fully
trained r449 model.  If α≥0.5 improves over r448, the r449 refit model is the banking
candidate.

No claim until the four literal cachesim panels complete.

## 2026-05-14 -- Tencent r450 2D WS-Conditioned Empirical Token Blend (Architectural)

**Motivation.** r449's 1D blend conditions on `ws_pre[0]` — the primary WS bin (count of unique
objects in the last 32 events).  This provides a single coarse axis of conditioning analogous to
PhaseAtlas's 8 phases.  However, the cache access pattern is jointly determined by multiple
time-scale signals: short-burst locality (ws0, last 32 events), medium-horizon locality (ws1,
last 128 events), and so on.  When ws0 is high but ws1 is low, the system is in a bursty-but-
contained phase (many new objects just arrived, but the medium-term set is stable).  When both
are high, the working set is genuinely expanding.  These two states should draw from materially
different rank-token distributions.

**Fix: joint (ws0, ws1) conditioning.**  A 2D empirical table
`rank_token_freq_by_ws01[ws0_bin][ws1_bin][rank_token]` is computed during training.
At generation time:

```
final_probs = (1 - α) × probs_so_far + α × empirical_2d[ws0_bin][ws1_bin]
```

Empty buckets fall back to the ws0-marginal (sum over ws1 for that ws0), which is the 1D
table.  This gives graceful degradation — if a (ws0, ws1) pair is unseen in training, the 1D
table is used automatically.

With 100k training rows, ~61k reuse events, and 32×32=1024 joint bins, average bucket size is
≈60 events — sufficient for a well-estimated categorical distribution over ~69 rank tokens.

**Constitution compliance.** Table fit from training trace only.  At generation time only
`ws_pre[0]` and `ws_pre[1]` — running counts from the generated sequence — are used to index
the table.

**Implementation** (`altgan/mattson_denning_lstm.py`):
- `rank_token_freqs_by_ws01()` — new function; shape `(ws0_bins, ws1_bins, vocab)`.
- `tokenize()` → 11-tuple (added `rank_token_freq_table_2d`).
- `fit()` → saves `rank_token_freq_by_ws01` to checkpoint.
- `generate_ids()` → new `ws_token_blend_2d` kwarg; loads 2D table; applies blend.
- CLI → `--ws-token-blend-2d` in `generate` + `multiseed`.

**Compatibility.** r449 checkpoints (no `rank_token_freq_by_ws01` key) fall back gracefully:
`ws_token_freq_table_2d` is None and the 2D blend is skipped.  2D blend requires a r450 refit.

**r450 sweep plan:**

**(A) 2D blend sweep on r449 refit checkpoint (plan B output), no extra refit needed:**
```
for blend2 in 0.25 0.50 0.75 1.00; do
  python3 -m altgan.mattson_denning_lstm multiseed \
    --real /tiamat/zarathustra/llgan-output/refs/tencent_stackatlas_real.csv \
    --max-rows 100000 --ws-edge-mode max-window \
    --pos-bins 0 --pos-embed 8 --recycle-rank-cap 0 \
    --rank-sampler empirical --exact-rank-cutoff 0 \
    --stack-depth-bins 32 --cache-ladder \
    --seeds 42,80,81,82 --temperature 1.0 \
    --short-reuse-pressure 3.0 --birth-control-mode ws \
    --ws-token-blend-2d $blend2 \
    --model /tiamat/zarathustra/checkpoints/altgan/tencent_mattson_denning_lstm_r449_sd32_fpcond_cachealign_cosine_drop01_e20.pt \
    --tag tencent_mdlstm_r450_blend2d${blend2}_r449base \
    --append-markdown /tiamat/zarathustra/Zarathustra/RESPONSE-LANL.md
done
```

**(B) r450 dedicated refit (same flags as r449B + 2D blend at α=0.5):**
```
python3 -m altgan.mattson_denning_lstm multiseed \
  --real /tiamat/zarathustra/llgan-output/refs/tencent_stackatlas_real.csv \
  --max-rows 100000 --ws-edge-mode max-window \
  --pos-bins 0 --pos-embed 8 --recycle-rank-cap 0 \
  --rank-sampler empirical --exact-rank-cutoff 0 \
  --stack-depth-bins 32 --cache-ladder \
  --dropout 0.1 --lr-schedule cosine --epochs 20 \
  --seeds 42,80,81,82 --temperature 1.0 \
  --short-reuse-pressure 3.0 --fit --birth-control-mode ws \
  --ws-token-blend-2d 0.5 \
  --model /tiamat/zarathustra/checkpoints/altgan/tencent_mattson_denning_lstm_r450_sd32_fpcond_cachealign_cosine_drop01_e20_wsblend2d05.pt \
  --tag tencent_mdlstm_r450_sd32_cachealign_cosine_drop01_e20_wsblend2d05 \
  --append-markdown /tiamat/zarathustra/Zarathustra/RESPONSE-LANL.md
```

Expected: the 2D blend (joint ws0×ws1 conditioning) should outperform the 1D blend by
discriminating between equal-ws0 states that differ in ws1 trajectory.  The empirical
bucket size (~60 events/bucket) is sufficient for a reliable estimate.

No claim until the four literal cachesim panels complete.

## 2026-05-14 -- Tencent r453 WS-KL Auxiliary Training Loss (Architectural)

**Motivation.** r449's WS-conditioned empirical blend corrects the LSTM's token distribution
at generation time by blending with the empirical table.  But this is a post-hoc patch.  The
fundamental gap is that cross-entropy training minimises token prediction loss, not token
distribution calibration.  The LSTM learns which tokens are likely next, but not which token
frequencies are expected at a given WS level.

**Fix: `--ws-kl-loss-weight λ`.**  Adds an auxiliary loss during training:
```
KL_loss = mean over reuse events of KL(empirical[ws0_bin] || softmax(LSTM_reuse_logits))
```

This directly pulls the LSTM's predicted reuse distribution toward the empirical distribution
conditioned on the current WS0 bin.  The effect: after training with this loss, the LSTM
naturally produces calibrated rank-token distributions at each WS level, so `--ws-token-blend`
needs smaller α (or α=0) to achieve the same quality as r449 at α=1.

The training uses the pre-computed `rank_token_freq_by_ws0` table (from `tokenize()`) as the
target distribution.  The table is computed first, then used as a regularisation signal.  This
is a chicken-and-egg free design: the table is computed once before training begins.

**Constitution compliance.** The empirical table is fit from training trace only.  The KL loss
at training step i conditions on the training-time WS bin (from `x_ws[:, :, 0]`), not on any
evaluation trace data.

**Implementation** (`altgan/mattson_denning_lstm.py`):
- `train_model()` → new `ws_freq_table`, `ws_kl_loss_weight` kwargs.
- Training loop: computes `KL(target_dist[ws0_idx] || log_softmax(reuse_logits))` over reuse
  events; adds to total loss with weight `ws_kl_loss_weight`.
- `fit()` → passes `rank_token_freq_table` and `ws_kl_loss_weight` to `train_model()`.
- CLI → `--ws-kl-loss-weight` in `add_train_flags`.
- Backward-compatible: λ=0.0 (default) disables the loss entirely.

**r453 sweep plan (KL weight ablation on top of r451 refit):**

```
for kl in 0.0 0.1 0.25 0.5; do
  python3 -m altgan.mattson_denning_lstm multiseed \
    --real /tiamat/zarathustra/llgan-output/refs/tencent_stackatlas_real.csv \
    --max-rows 100000 --ws-edge-mode max-window \
    --pos-bins 0 --pos-embed 8 --recycle-rank-cap 0 \
    --rank-sampler empirical --exact-rank-cutoff 0 \
    --stack-depth-bins 32 --cache-ladder --ws-cache-ladder \
    --dropout 0.1 --lr-schedule cosine --epochs 20 \
    --ws-kl-loss-weight $kl \
    --seeds 42,80,81,82 --temperature 1.0 \
    --short-reuse-pressure 3.0 --fit --birth-control-mode ws \
    --ws-token-blend 0.5 --ws-token-blend-2d 0.25 --ws-blend-confidence-tau 50 \
    --model /tiamat/zarathustra/checkpoints/altgan/tencent_mdlstm_r453_kl${kl}.pt \
    --tag tencent_mdlstm_r453_kl${kl} \
    --append-markdown /tiamat/zarathustra/Zarathustra/RESPONSE-LANL.md
done
```

Expected: λ=0.1–0.25 improves over the generation-only blend (r449-r452) by teaching the LSTM
to produce calibrated distributions directly.  Reduces seed-variance because the LSTM is
explicitly constrained.  At λ>0, `--ws-token-blend` can be reduced from α=1.0 toward α=0
(verifiable by ablation).

No claim until the four literal cachesim panels complete.

## 2026-05-14 -- Tencent r452 Per-Bin Confidence-Weighted Empirical Blend (Architectural)

**Motivation.** r449/r450's empirical blends apply a fixed α to all WS bins regardless of
how many training events fall in each bin.  With 100k training rows and 32 WS bins, average
bucket size is ~1900 events/bin for 1D and ~60 events/bin for 2D.  The distribution is
Zipf-heavy: some (ws0, ws1) pairs are rare (low traffic, unusual cache pressure), giving
buckets with < 5 events.  A fixed α=1.0 would pull generation towards the sparse-bucket
distribution, which is poorly estimated.

**Fix: `--ws-blend-confidence-tau τ`.**  Scale the effective α by:
```
alpha_eff = alpha * min(1.0, sqrt(bucket_count / tau))
```
- τ=50: bins with ≥50 events get α_eff ≈ α (full blend weight)
- τ=50: bins with 5 events get α_eff ≈ 0.32 × α (mostly LSTM)
- τ=50: bins with 0 events: α_eff = 0 (pure LSTM — same as today since empirical dist is uniform)

Applies to both 1D (ws0) and 2D (ws0, ws1) blends independently.  Per-bin counts saved to
checkpoint as `rank_token_freq_by_ws0_counts` and `rank_token_freq_by_ws01_counts`.

**Constitution compliance.** Counts fit from training trace only.  τ is a hyperparameter, not
derived from the evaluation trace.

**Implementation** (`altgan/mattson_denning_lstm.py`):
- `rank_token_freqs_by_ws0()` and `rank_token_freqs_by_ws01()` now return `(table, counts)`.
- `tokenize()` → 13-tuple (added counts); `fit()` saves counts to checkpoint.
- `generate_ids()` → new `ws_blend_confidence_tau` kwarg; applies confidence scaling.
- CLI → `--ws-blend-confidence-tau` in `generate` + `multiseed`.
- Backward-compatible: τ=0.0 (default) disables confidence scaling (same as before).

**r452 sweep plan (combined with r451 flags):**

```
for tau in 0.0 25.0 50.0 100.0; do
  python3 -m altgan.mattson_denning_lstm multiseed \
    --real /tiamat/zarathustra/llgan-output/refs/tencent_stackatlas_real.csv \
    --max-rows 100000 --ws-edge-mode max-window \
    --pos-bins 0 --pos-embed 8 --recycle-rank-cap 0 \
    --rank-sampler empirical --exact-rank-cutoff 0 \
    --stack-depth-bins 32 --cache-ladder --ws-cache-ladder \
    --dropout 0.1 --lr-schedule cosine --epochs 20 \
    --seeds 42,80,81,82 --temperature 1.0 \
    --short-reuse-pressure 3.0 --birth-control-mode ws \
    --ws-token-blend 1.0 --ws-token-blend-2d 0.5 \
    --ws-blend-confidence-tau $tau \
    --model /tiamat/zarathustra/checkpoints/altgan/tencent_mattson_denning_lstm_r451_full.pt \
    --tag tencent_mdlstm_r452_tau${tau} \
    --append-markdown /tiamat/zarathustra/Zarathustra/RESPONSE-LANL.md
done
```

Expected: τ=50 improves over τ=0 for the 2D blend (60 events/bucket average) while making
little difference for the 1D blend (1900 events/bucket average, already dense).

No claim until the four literal cachesim panels complete.

## 2026-05-14 -- Tencent r451 Cache-Ladder-Aligned WS Bin Edges (Architectural)

**Motivation.** r448 added mandatory rank-edge boundaries at the five cachesim evaluation
sizes [32, 128, 512, 2048, 8192] so rank token bins never straddle an evaluation point.
r449/r450 introduced a WS-conditioned empirical blend that uses the Denning working-set bin
(`ws_pre[0]`) as the conditioning key.  But the WS edge array is built by `make_log_edges()`
with no such mandatory boundaries.  Consider WS events at sizes 28, 35, 130, 2100:
- ws_pre=28 and ws_pre=35 may fall in the same WS bin even though 28 < 32 < 35.
- The empirical rank-token distributions for events in the "near-32 working-set" bucket will
  average across two qualitatively different cache-hit regimes (WS fits in cache-32 vs not).

This is the same structural HRC-MAE floor that r448 fixed for rank vocabulary — now applied to
the WS conditioning dimension.

**Fix: `--ws-cache-ladder` flag.**  Adds mandatory WS-edge boundaries at exactly the cache
evaluation sizes [32, 128, 512, 2048, 8192] (same sizes as `--cache-ladder`; shared
`--ladder-sizes` arg).  Applied to the flat-array WS edge modes (footprint, max-window) only;
per-window mode is unaffected.  The WS edge array grows from ~32 edges to ~37 — a small
vocabulary increase with no parameter cost.

**Constitution compliance.** WS edge boundaries are fixed constants derived from the
evaluation protocol, identical to rank ladder sizes.  No real-trace data consumed.

**Implementation** (`altgan/mattson_denning_lstm.py`):
- `tokenize(ws_cache_sizes=...)` — new kwarg; injects mandatory boundaries into `ws_edges`
  (flat-array mode only).
- `fit()` — reads `args.ws_cache_ladder` + `args.ladder_sizes`; passes `ws_cache_sizes` to
  `tokenize()`; saves `ws_cache_ladder_sizes` to checkpoint (WS edges already saved; sizes
  stored for provenance).
- CLI — `--ws-cache-ladder` flag in `add_train_flags` (shared `--ladder-sizes`).

**r451 sweep plan:**

```
python3 -m altgan.mattson_denning_lstm multiseed \
  --real /tiamat/zarathustra/llgan-output/refs/tencent_stackatlas_real.csv \
  --max-rows 100000 --ws-edge-mode max-window \
  --pos-bins 0 --pos-embed 8 --recycle-rank-cap 0 \
  --rank-sampler empirical --exact-rank-cutoff 0 \
  --stack-depth-bins 32 --cache-ladder --ws-cache-ladder \
  --dropout 0.1 --lr-schedule cosine --epochs 20 \
  --seeds 42,80,81,82 --temperature 1.0 \
  --short-reuse-pressure 3.0 --fit --birth-control-mode ws \
  --ws-token-blend 0.5 --ws-token-blend-2d 0.5 \
  --model /tiamat/zarathustra/checkpoints/altgan/tencent_mattson_denning_lstm_r451_full.pt \
  --tag tencent_mdlstm_r451_full_wsladder_blend_blend2d \
  --append-markdown /tiamat/zarathustra/Zarathustra/RESPONSE-LANL.md
```

Expected: WS-edge alignment eliminates the quantization bias at cache boundaries for the
WS-conditioned blend lookup, reducing the HRC-MAE floor.  Combined with r449's 1D blend and
r450's 2D blend, the triple improvement should approach PhaseAtlas quality.

No claim until the four literal cachesim panels complete.

## 2026-05-14 -- Tencent r454 2D WS-KL Auxiliary Training Loss (Architectural)

**Motivation.** r453 adds a 1D KL auxiliary loss conditioned on ws0.  The 2D empirical table
(from r450) provides a finer-grained target conditioned on (ws0, ws1) jointly.  The 2D KL
loss teaches the LSTM to match the joint conditional distribution, which is more discriminative.

**Fix: `--ws-kl-loss-weight-2d λ`.**

```
KL_loss_2d = mean over reuse events of KL(empirical_2d[ws0_bin, ws1_bin] || LSTM_reuse_dist)
```

Combined with the 1D KL loss, this jointly regularises the LSTM to match both the 1D and 2D
empirical distributions.  The r453 1D KL constraint is a coarse target; the r454 2D KL is
fine-grained.  Together, they impose a hierarchy of distribution matching constraints on the
LSTM.

**Constitution compliance.** Same as r453: 2D empirical table fit from training trace only.
Conditioning indices `y_ws[:, :, 0]` and `y_ws[:, :, 1]` are training-time WS counts.

**Implementation** (`altgan/mattson_denning_lstm.py`):
- `train_model()` → new `ws_freq_table_2d` and `ws_kl_loss_weight_2d` kwargs.
- `ws_freq_t_2d[ws0_idx, ws1_idx]` indexed by `(y_ws[:,:,0], y_ws[:,:,1])`.
- `fit()` → passes `rank_token_freq_table_2d` when weight > 0.
- CLI → `--ws-kl-loss-weight-2d` in `add_train_flags`.
- Backward-compatible: λ=0.0 (default) disables.

**Combined r449–r454 Master Recipe (best expected result):**

```bash
python3 -m altgan.mattson_denning_lstm multiseed \
  --real /tiamat/zarathustra/llgan-output/refs/tencent_stackatlas_real.csv \
  --max-rows 100000 --ws-edge-mode max-window \
  --pos-bins 0 --pos-embed 8 --recycle-rank-cap 0 \
  --rank-sampler empirical --exact-rank-cutoff 0 \
  --stack-depth-bins 32 --cache-ladder --ws-cache-ladder \
  --dropout 0.1 --lr-schedule cosine --epochs 20 \
  --ws-kl-loss-weight 0.25 --ws-kl-loss-weight-2d 0.10 \
  --seeds 42,80,81,82 --temperature 1.0 \
  --short-reuse-pressure 3.0 --fit --birth-control-mode ws \
  --ws-token-blend 0.5 --ws-token-blend-2d 0.25 --ws-blend-confidence-tau 50 \
  --model /tiamat/zarathustra/checkpoints/altgan/tencent_mdlstm_r454_master.pt \
  --tag tencent_mdlstm_r454_master \
  --append-markdown /tiamat/zarathustra/Zarathustra/RESPONSE-LANL.md
```

Theoretical expected range: 0.030–0.045 (vs r434 baseline 0.0601, retracted PhaseAtlas 0.0297).
The WS-KL losses teach the LSTM to produce calibrated distributions; the generation-time blend
provides a safety net.  Cache-ladder alignment (r448, r451) removes structural HRC-MAE floors.

No claim until the four literal cachesim panels complete.

---

## 2026-05-14 -- Tencent r455 FiLM Conditioning of LSTM Output (Architectural)

**Motivation.** r449–r454 improve the *marginal* token distribution via WS-conditioned
empirical blending and KL training loss. But the LSTM hidden state `out` after the recurrent
pass is blended uniformly before all heads: WS context enters only as input concatenation at
time t. This means the LSTM's nonlinear representation of the sequence is not *modulated* by
the current WS context — it only *sees* WS context as an extra input channel.

Feature-wise linear modulation (FiLM) provides a richer interaction:
```
out_film = out * (1 + gamma(ws_ctx)) + beta(ws_ctx)
```
where `ws_ctx = concat(ws_emb[0], ..., ws_emb[n_windows-1] [, fp_emb])` is the current WS
embedding context. The LSTM output is *rescaled and shifted* per-hidden-dimension as a
function of WS context — a second-order interaction that input concatenation cannot capture.

**Diagnosis.** After WS-KL training (r453/r454), the LSTM knows the right marginal shape,
but can still misallocate probability mass within each WS bin because the prediction head
`self.head = nn.Linear(hidden, vocab)` sees the same `out` regardless of current WS context.
FiLM lets the prediction head operate on a WS-rescaled `out`, so the model can sharpen or
suppress different rank-depth regions depending on current WS state.

**Fix.** Added `--film-cond` flag. When enabled, `build_model` creates:
```python
ws_context_dim = n_windows * ws_embed + (ws_embed if use_fp else 0)
self.film_gamma = nn.Linear(ws_context_dim, hidden, bias=False)
self.film_beta  = nn.Linear(ws_context_dim, hidden, bias=False)
```
After the LSTM step in `forward()`:
```python
ws_ctx = torch.cat(ws_embs [+ fp_emb_val], dim=-1)
out = out * (1 + self.film_gamma(ws_ctx)) + self.film_beta(ws_ctx)
```
All heads operate on `out_film`. `film_cond` saved in checkpoint and restored on load.
Zero overhead when disabled (`film_gamma = None`; old code path preserved).

**Constitution compliance.** CE loss unchanged; FiLM is a model parameter transformation,
not cachesim feedback. Article IV satisfied. Backward compatible via `strict=False` load.

**Sweep plan (FiLM ablation on top of r454 master recipe):**

```bash
python3 -m altgan.mattson_denning_lstm multiseed \
  --real /tiamat/zarathustra/llgan-output/refs/tencent_stackatlas_real.csv \
  --max-rows 100000 --ws-edge-mode max-window \
  --pos-bins 0 --pos-embed 8 --recycle-rank-cap 0 \
  --rank-sampler empirical --exact-rank-cutoff 0 \
  --stack-depth-bins 32 --cache-ladder --ws-cache-ladder \
  --dropout 0.1 --lr-schedule cosine --epochs 20 \
  --ws-kl-loss-weight 0.25 --ws-kl-loss-weight-2d 0.10 \
  --film-cond \
  --seeds 42,80,81,82 --temperature 1.0 \
  --short-reuse-pressure 3.0 --fit --birth-control-mode ws \
  --ws-token-blend 0.5 --ws-token-blend-2d 0.25 --ws-blend-confidence-tau 50 \
  --model /tiamat/zarathustra/checkpoints/altgan/tencent_mdlstm_r455_film.pt \
  --tag tencent_mdlstm_r455_film \
  --append-markdown /tiamat/zarathustra/Zarathustra/RESPONSE-LANL.md
```

**Expected impact.** FiLM adds `2 × ws_context_dim × hidden` parameters (e.g. 3 WS windows
× 16-dim ws_embed = 48-dim context, hidden=128: 12,288 new params < 5% of model). Expected
HRC-MAE improvement over r454: 0.005–0.015 on Tencent if WS distribution spread is high.

No claim until four-seed cachesim panels complete.

---

## 2026-05-14 -- Tencent r456 WS-Conditioned Birth Probability Calibration (Generation-Time)

**Motivation.** The `birth_head` (binary fresh vs reuse) is trained with BCE but has no
explicit calibration against the empirical fresh-token rate per WS bin.  If the LSTM
systematically over- or under-generates fresh tokens at a given WS state, every simulated
HRC curve is biased — the hit rate at cache size C is directly proportional to the
fresh-token fraction in [C, ∞) of the LRU stack depth distribution.

**Diagnosis.** Let `p_birth = P(fresh | ws0_bin=w0)` in the real trace.  The LSTM predicts
`sigmoid(birth_logit)` which optimises cross-entropy across all WS states jointly.  In WS
bins with few training examples the LSTM may revert to the global mean birth rate.  In bins
at WS extremes (very low = hot reuse phase; very high = cold miss phase) the true birth rate
differs from the global mean by up to ±30%, directly translating to ±0.03 HRC-MAE.

**Fix.** New function `birth_rate_by_ws0(tokens, ws_tokens, ws0_bins)`:
```python
def birth_rate_by_ws0(tokens, ws_tokens, ws0_bins) -> tuple[np.ndarray, np.ndarray]:
    fresh_counts = zeros(ws0_bins); total_counts = zeros(ws0_bins)
    for tok, ws_row: fresh_counts[ws0_bin] += (tok==FRESH_TOKEN); total_counts[ws0_bin] += 1
    global_rate = sum(fresh) / sum(total)
    rates = where(total > 0, fresh / total, global_rate)
    return rates.astype(float32), total_counts.astype(float32)
```

Called inside `tokenize()` (after existing rank-token freq tables); stored in checkpoint as
`empirical_birth_rates_by_ws0` and `birth_rate_counts_by_ws0`.

At generation time, new flag `--birth-rate-blend`:
```python
alpha_br *= min(1.0, sqrt(count / tau))   # reuses --ws-blend-confidence-tau
birth_prob = (1 - alpha_br) * birth_prob + alpha_br * empirical_birth_rates[w0]
```

Applied *before* the force_new / force_reuse override logic so birth_control_mode=ws still
takes precedence when the WS pressure is large.

**Constitution compliance.** Generation-time correction only — no refit required.  Empirical
birth rate fit from training trace only (no cachesim).  Article IV satisfied.

**Sweep plan (r456 birth-rate-blend ablation, generation-time only, no refit needed):**

```bash
# r456 standalone: apply to any existing r454/r455 checkpoint
for blend in 0.25 0.5 0.75; do
python3 -m altgan.mattson_denning_lstm multiseed \
  --real /tiamat/zarathustra/llgan-output/refs/tencent_stackatlas_real.csv \
  --max-rows 100000 --ws-edge-mode max-window \
  --model /tiamat/zarathustra/checkpoints/altgan/tencent_mdlstm_r455_film.pt \
  --seeds 42,80,81,82 --temperature 1.0 \
  --short-reuse-pressure 3.0 --birth-control-mode ws \
  --ws-token-blend 0.5 --ws-token-blend-2d 0.25 --ws-blend-confidence-tau 50 \
  --birth-rate-blend ${blend} \
  --tag tencent_mdlstm_r456_brblend${blend} \
  --append-markdown /tiamat/zarathustra/Zarathustra/RESPONSE-LANL.md
done
```

**Expected impact.** Birth rate miscalibration is a structural bias; correcting it by 50%
(blend=0.5) should reduce HRC-MAE by 0.005–0.020 depending on WS variance in the trace.

No claim until four-seed cachesim panels complete.

---

## 2026-05-14 -- Tencent r457 Configurable LSTM Depth (Architectural)

**Motivation.** The LSTM has been hardwired to 2 layers since r430.  With the rich WS
conditioning signal now available (r449–r456), a 3-layer LSTM can learn deeper abstractions
over the WS-conditioned token sequence without requiring a wider hidden dimension.  3 layers
at hidden=128 adds ~50K parameters (≈25% increase) with the same memory footprint as 2 layers
at hidden=160 — but depth often beats width for sequence modelling.

**Fix.** Added `lstm_layers: int = 2` parameter to `build_model()` and `train_model()`.
`--lstm-layers N` CLI flag in `add_train_flags()`.  Default remains 2 (backward compatible).
`dropout` automatically disabled when `lstm_layers=1` (PyTorch restriction).  Saved in
checkpoint; restored on load.

**Sweep plan:**

```bash
python3 -m altgan.mattson_denning_lstm multiseed \
  --real /tiamat/zarathustra/llgan-output/refs/tencent_stackatlas_real.csv \
  --max-rows 100000 --ws-edge-mode max-window \
  --pos-bins 0 --pos-embed 8 --recycle-rank-cap 0 \
  --rank-sampler empirical --exact-rank-cutoff 0 \
  --stack-depth-bins 32 --cache-ladder --ws-cache-ladder \
  --dropout 0.1 --lr-schedule cosine --epochs 20 \
  --ws-kl-loss-weight 0.25 --ws-kl-loss-weight-2d 0.10 \
  --film-cond --lstm-layers 3 \
  --seeds 42,80,81,82 --temperature 1.0 \
  --short-reuse-pressure 3.0 --fit --birth-control-mode ws \
  --ws-token-blend 0.5 --ws-token-blend-2d 0.25 --ws-blend-confidence-tau 50 \
  --birth-rate-blend 0.5 \
  --model /tiamat/zarathustra/checkpoints/altgan/tencent_mdlstm_r457_3layer.pt \
  --tag tencent_mdlstm_r457_3layer \
  --append-markdown /tiamat/zarathustra/Zarathustra/RESPONSE-LANL.md
```

**Combined r449–r457 Master Recipe** (all improvements stacked):
- `--stack-depth-bins 32 --cache-ladder --ws-cache-ladder` (r446/r448/r451)
- `--dropout 0.1 --lr-schedule cosine --epochs 20` (r449)
- `--ws-kl-loss-weight 0.25 --ws-kl-loss-weight-2d 0.10` (r453/r454)
- `--film-cond` (r455)
- `--lstm-layers 3` (r457)
- `--ws-token-blend 0.5 --ws-token-blend-2d 0.25 --ws-blend-confidence-tau 50` (r449/r450/r452)
- `--birth-rate-blend 0.5` (r456)

No claim until four-seed cachesim panels complete.

---

## 2026-05-14 -- Tencent r458 Label Smoothing + Configurable Gradient Clip (Training Quality)

**Motivation.** The reuse token CE loss (`F.cross_entropy`) has sharp one-hot targets.
With WS-KL auxiliary losses (r453/r454) and a 3-layer LSTM (r457), the model has increased
capacity and may over-fit high-frequency rank bins, reducing the probability assigned to
infrequent-but-important deep reuse ranks.  Label smoothing `(1-eps)*onehot + eps/V`
regularises the token distribution, preventing over-confidence on seen rank depths.

Gradient clipping was hardcoded at 1.0.  Made configurable via `--grad-clip`; default 1.0
preserves backward compatibility; `--grad-clip 0.0` disables it.

**Fix.** `train_model(... label_smoothing: float = 0.0, grad_clip: float = 1.0)`.
Label smoothing passed to `F.cross_entropy(..., label_smoothing=eps)` on the reuse token
loss.  Gradient clip uses `clip_grad_norm_` only when `grad_clip > 0`.  Both logged.
No checkpoint storage needed (training-time hyperparams only).

**Sweep plan (r458 label smoothing ablation):**

```bash
for smooth in 0.0 0.05 0.10; do
python3 -m altgan.mattson_denning_lstm multiseed \
  --real /tiamat/zarathustra/llgan-output/refs/tencent_stackatlas_real.csv \
  --max-rows 100000 --ws-edge-mode max-window \
  --pos-bins 0 --pos-embed 8 --recycle-rank-cap 0 \
  --rank-sampler empirical --exact-rank-cutoff 0 \
  --stack-depth-bins 32 --cache-ladder --ws-cache-ladder \
  --dropout 0.1 --lr-schedule cosine --epochs 20 \
  --ws-kl-loss-weight 0.25 --ws-kl-loss-weight-2d 0.10 \
  --film-cond --lstm-layers 3 --label-smoothing ${smooth} \
  --seeds 42,80,81,82 --temperature 1.0 \
  --short-reuse-pressure 3.0 --fit --birth-control-mode ws \
  --ws-token-blend 0.5 --ws-token-blend-2d 0.25 --ws-blend-confidence-tau 50 \
  --birth-rate-blend 0.5 \
  --model /tiamat/zarathustra/checkpoints/altgan/tencent_mdlstm_r458_smooth${smooth}.pt \
  --tag tencent_mdlstm_r458_smooth${smooth} \
  --append-markdown /tiamat/zarathustra/Zarathustra/RESPONSE-LANL.md
done
```

**Combined r449–r458 Master Recipe** (all improvements stacked, recommended first run):

```bash
python3 -m altgan.mattson_denning_lstm multiseed \
  --real /tiamat/zarathustra/llgan-output/refs/tencent_stackatlas_real.csv \
  --max-rows 100000 --ws-edge-mode max-window \
  --pos-bins 0 --pos-embed 8 --recycle-rank-cap 0 \
  --rank-sampler empirical --exact-rank-cutoff 0 \
  --stack-depth-bins 32 --cache-ladder --ws-cache-ladder \
  --dropout 0.1 --lr-schedule cosine --epochs 20 --grad-clip 1.0 \
  --ws-kl-loss-weight 0.25 --ws-kl-loss-weight-2d 0.10 \
  --film-cond --lstm-layers 3 --label-smoothing 0.05 \
  --seeds 42,80,81,82 --temperature 1.0 \
  --short-reuse-pressure 3.0 --fit --birth-control-mode ws \
  --ws-token-blend 0.5 --ws-token-blend-2d 0.25 --ws-blend-confidence-tau 50 \
  --birth-rate-blend 0.5 \
  --model /tiamat/zarathustra/checkpoints/altgan/tencent_mdlstm_r458_master.pt \
  --tag tencent_mdlstm_r458_master \
  --append-markdown /tiamat/zarathustra/Zarathustra/RESPONSE-LANL.md
```

Expected HRC-MAE range: 0.025–0.040 (vs r434 baseline 0.0601).

No claim until four-seed cachesim panels complete.

---

## 2026-05-14 -- Tencent r459 WS0-Conditioned Rank Sampler (Architectural)

**Motivation.** r449's WS-conditioned token blend calibrates `P(token class | ws0_bin)` —
the probability of each Mattson-depth bin given the current WS0 state.  But the rank WITHIN
each token bin (the actual Mattson depth used for the LRU stack operation) is still drawn
from only footprint-conditioned samples (r447) or global per-token samples.  Since the rank
distribution within a given bin depends strongly on WS state (shallow stack = most ranks near 0;
deep stack = ranks spread across the full bin range), this is a structural remaining gap.

**The analogy.** r449 gives the MDLSTM the same PROBABILITY of choosing each depth bin as
PhaseAtlas.  r459 gives the MDLSTM the same WITHIN-BIN RANK DRAW as PhaseAtlas.  Together,
`P(rank) = P(token class | ws0_bin) × P(rank | token class, ws0_bin)` is fully WS-conditioned,
which is exactly what PhaseAtlas achieves by sampling from per-phase rank distributions.

**Fix.** New function `rank_samples_from_depths_ws0(depths, tokens, ws_tokens, vocab, ws0_bins, ...)`:

```python
samples[token_i][w0].append(depth_i)  # for each (token, ws0_bin) pair
```

Shape: `[vocab][ws0_bins]` list-of-lists.  Storage: ~100k reuse events / (69 × 32) ≈ 45
samples per (token, ws0) bucket on average — above the 8-sample fallback threshold for
most buckets.

At generation time, `_sample_rank_for_token()` tries the WS0-conditioned bucket first
(≥8 samples), then falls back to the footprint-conditioned bucket (r447), then global.

**Constitution compliance.** Table fit from training trace only.  At generation time only
`ws_pre[0]` — a running count from the generated sequence — selects the bucket.  Article IV
satisfied.  Backward-compatible: missing `rank_samples_by_token_ws0` key falls back to r447.

**Sweep plan (r459 master recipe — refit required):**

```bash
python3 -m altgan.mattson_denning_lstm multiseed \
  --real /tiamat/zarathustra/llgan-output/refs/tencent_stackatlas_real.csv \
  --max-rows 100000 --ws-edge-mode max-window \
  --pos-bins 0 --pos-embed 8 --recycle-rank-cap 0 \
  --rank-sampler empirical --exact-rank-cutoff 0 \
  --stack-depth-bins 32 --cache-ladder --ws-cache-ladder \
  --dropout 0.1 --lr-schedule cosine --epochs 20 --grad-clip 1.0 \
  --ws-kl-loss-weight 0.25 --ws-kl-loss-weight-2d 0.10 \
  --film-cond --lstm-layers 3 --label-smoothing 0.05 \
  --seeds 42,80,81,82 --temperature 1.0 \
  --short-reuse-pressure 3.0 --fit --birth-control-mode ws \
  --ws-token-blend 0.5 --ws-token-blend-2d 0.25 --ws-blend-confidence-tau 50 \
  --birth-rate-blend 0.5 \
  --model /tiamat/zarathustra/checkpoints/altgan/tencent_mdlstm_r459_ws0rank.pt \
  --tag tencent_mdlstm_r459_ws0rank \
  --append-markdown /tiamat/zarathustra/Zarathustra/RESPONSE-LANL.md
```

**Combined r449–r459 Master Recipe** (all improvements stacked):
- `--stack-depth-bins 32 --cache-ladder --ws-cache-ladder` (r446/r448/r451)
- `--dropout 0.1 --lr-schedule cosine --epochs 20` (r449)
- `--ws-kl-loss-weight 0.25 --ws-kl-loss-weight-2d 0.10` (r453/r454)
- `--film-cond` (r455)
- `--lstm-layers 3` (r457)
- `--label-smoothing 0.05 --grad-clip 1.0` (r458)
- `--ws-token-blend 0.5 --ws-token-blend-2d 0.25 --ws-blend-confidence-tau 50` (r449/r450/r452)
- `--birth-rate-blend 0.5` (r456)
- `--rank-sampler empirical` with `rank_samples_by_token_ws0` in checkpoint (r459)

**Expected impact.** WS-rank conditioning completes the within-bin calibration.  If rank
distribution spread within each token bin varies 30–50% across WS states (plausible for
Mattson depths in a trace with strong locality phases), correcting this adds 0.005–0.015 to
HRC-MAE improvement.  Combined with r449-r458, expected total HRC-MAE: 0.020–0.038.

No claim until four-seed cachesim panels complete.

---

## 2026-05-14 -- Tencent r460 2D Birth Rate Calibration (Generation-Time)

**Motivation.** r456 conditions P(fresh) on `ws0_bin` alone (1D calibration).  The 2D
token blend (r450) showed that `(ws0_bin, ws1_bin)` joint conditioning is more
discriminative than `ws0_bin` alone — states with equal `ws0` but different `ws1` can have
materially different access patterns.  The same logic applies to birth rates: when both
ws0 AND ws1 are high, the trace is likely in a genuine cold-miss phase (fast footprint
growth); when ws0 is high but ws1 is low, it is a bursty-but-contained phase (more
transient fresh objects).  These two states can differ in birth rate by 10–20%.

**Fix.** New function `birth_rate_by_ws01(tokens, ws_tokens, ws0_bins, ws1_bins)`:
```python
rates[w0, w1] = P(fresh | ws0_bin=w0, ws1_bin=w1)
```
Empty `(w0, w1)` pairs fall back to the ws0 marginal (`birth_rate_by_ws0`), which in turn
falls back to the global rate.  Applied AFTER the 1D blend (r456):
```python
birth_prob = (1 - α₁) × lstm_birth + α₁ × birth_rates_1d[w0]      # r456
birth_prob = (1 - α₂) × birth_prob + α₂ × birth_rates_2d[w0, w1]  # r460
```

**Constitution compliance.** Generation-time correction only — no refit required for
generation-only use (the 2D table IS stored in checkpoint).  Refit required to populate
`empirical_birth_rates_by_ws01`.  Article IV satisfied.  Confidence weighting via
`--ws-blend-confidence-tau` applies (sparser 2D bins get proportionally lower weight).

**Combined r449–r460 Master Recipe** (all improvements stacked):
- `--stack-depth-bins 32 --cache-ladder --ws-cache-ladder` (r446/r448/r451)
- `--dropout 0.1 --lr-schedule cosine --epochs 20 --grad-clip 1.0` (r449/r458)
- `--ws-kl-loss-weight 0.25 --ws-kl-loss-weight-2d 0.10` (r453/r454)
- `--film-cond --lstm-layers 3 --label-smoothing 0.05` (r455/r457/r458)
- `--ws-token-blend 0.5 --ws-token-blend-2d 0.25 --ws-blend-confidence-tau 50` (r449/r450/r452)
- `--birth-rate-blend 0.5 --birth-rate-blend-2d 0.25` (r456/r460)
- `--rank-sampler empirical` with `rank_samples_by_token_ws0` in checkpoint (r459)

No claim until four-seed cachesim panels complete.
