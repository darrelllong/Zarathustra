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
