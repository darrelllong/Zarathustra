# Zarathustra Peer Review

Reviewed on March 27, 2026 after pulling `main` and reading:

- `TODO.md`
- `README.md`
- `llgan/config.py`
- `llgan/dataset.py`
- `llgan/model.py`
- `llgan/train.py`
- `llgan/generate.py`
- `llgan/eval.py`

This review focuses on the current code, not the earlier bugs already documented as fixed in `TODO.md`.

## Executive Summary

The project is in a much better state than the original LLGAN baseline. The biggest wins already landed are the file-local windowing, the locality-aware `obj_id` representation, the split latent design, and the latent AE/supervisor path.

The next step should not be "add more papers." It should be:

1. Fix the current correctness bugs in training and generation.
2. Make the validation/evaluation path harder to game.
3. Add one or two architecture changes that match the present bottlenecks.

Right now there are a few places where the code is still undermining the model before the unchecked TODO items even get a fair test.

## Highest-Priority Fixes

### 1. `wgan-gp` currently uses the wrong generator loss

In `llgan/train.py`, the critic step correctly treats both `wgan-sn` and `wgan-gp` as Wasserstein losses, but the generator step only uses `-C(fake).mean()` for `wgan-sn`. For `wgan-gp`, it falls into the BCE branch instead.

Relevant code:

- `llgan/train.py:454-480`
- `llgan/train.py:495-503`

Why this matters:

- The critic is being trained as a Wasserstein critic while the generator is being trained as a BCE GAN.
- Any `wgan-gp` ablation is currently not trustworthy.
- This is more important than adding AGP, RpGAN, or any other critic-side upgrade.

Recommendation:

- Treat both `wgan-sn` and `wgan-gp` as Wasserstein objectives in the generator step.
- Then re-run the `wgan-gp` experiments before adding more regularizers.

### 2. The supervisor is effectively frozen during joint training, and part of the ER loss is a no-op

The latent AE path pretrains `S`, but during joint training there is no optimizer step for `opt_S`. At the same time, the ER step adds a supervisor term that does not update the parameters in `opt_ER`.

Relevant code:

- `llgan/train.py:257-260`
- `llgan/train.py:515-523`
- `llgan/train.py:563-576`

Why this matters:

- `S` can become stale while `E`, `R`, and `G` continue moving.
- In the ER step, `S_on_real = S(H_real_grad.detach())` and the target is also detached, but `opt_ER` only updates `E` and `R`. That means `loss_sup_e` does not help the parameters actually being stepped there.
- The current joint phase is not really doing the full TimeGAN/SeriesGAN-style supervisor coupling the comments imply.

Recommendation:

- Decide explicitly between two designs:
- Freeze `S` after pretraining and remove the dead supervisor term from the ER step.
- Or train `S` jointly with its own optimizer during phase 3 so it tracks the evolving latent space.

### 3. `--n-streams` generation is currently mixing independent traces into one output file

`generate.py` correctly maintains separate hidden states across windows, but then flattens all streams into one array and runs `prep.inverse_transform()` over the merged sequence.

Relevant code:

- `llgan/generate.py:65-100`

Why this matters:

- With `n_streams > 1`, the output file interleaves distinct synthetic traces.
- Timestamp and `obj_id` reconstruction then apply cumulative sums across stream boundaries.
- The result is not "parallel independent traces"; it is a stitched artifact.

Recommendation:

- If `n_streams > 1`, write one file per stream, or at minimum preserve `stream_id` and inverse-transform each stream independently.
- Until then, I would treat `--n-streams` as unsafe for scientific results.

### 4. Multi-file validation is still contaminated and optimistic

Training no longer windows across file boundaries, but the validation path still concatenates held-out files into one array and windows across that combined sequence.

Relevant code:

- `llgan/train.py:204-217`

Why this matters:

- Validation windows can still cross file boundaries.
- The validation files are sampled from `all_files` without being excluded from later epoch sampling, so the "held-out" signal is not actually file-held-out.
- Model selection on `best.pt` can be biased upward.

Recommendation:

- Build validation exactly the same way as training: per-file `TraceDataset`, no cross-file windows.
- Keep a fixed held-out file set that is never sampled for training.

## Most Valuable Model Improvements After the Fixes Above

### 5. Add the feature-space discriminator now, not later

The latent critic is the right stability move, but by itself it can miss decode-space artifacts. The TODO already points toward a dual discriminator; I think that should move up the queue.

Why this matters:

- Right now the critic only sees the latent sequence in AE mode.
- The generator is still judged only indirectly on decoded feature realism, through moment loss, FFT loss, and recovery.
- If the latent space is smooth but the decoder maps it to bad feature-space artifacts, the current training loop has a blind spot.

Recommendation:

- Add a lightweight feature-space critic on `R(H_fake)` and `real_batch`.
- Keep the latent critic for stability and early gradients.
- This is a better next investment than adding yet another scalar regularizer.

### 6. Turn `z_global` into real workload conditioning instead of pure random identity noise

The split latent design is good, but the global code is still just noise. Given the current goal, that leaves controllability and corpus structure on the table.

Relevant code:

- `llgan/model.py:102-145`
- `llgan/train.py:449-451`

Why this matters:

- You now have the right injection point for conditioning, but not the conditioning signal.
- Mixed-corpus training without workload metadata risks learning an average workload rather than a family of workloads.
- This is especially important once Tencent and Alibaba are both in play.

Recommendation:

- Compute simple per-file descriptors at load time: read ratio, mean/log-mean IAT, mean size, size entropy, repeat rate, signed-`obj_id`-delta histogram, tenant cardinality, maybe a working-set proxy.
- Feed those descriptors through an MLP into the generator initial state and optionally FiLM the recurrent activations.
- Keep stochastic `z_global`, but concatenate or affine-modulate it with the workload descriptor.

### 7. Add locality-specific losses and metrics now that `obj_id` is no longer raw regression

The signed-delta representation is a strong change, but the loss still treats locality mostly through generic moments and FFT.

Relevant code:

- `llgan/dataset.py:177-193`
- `llgan/dataset.py:304-330`
- `llgan/train.py:530-554`

Why this matters:

- I/O realism is not just about marginals; it is heavily about locality structure.
- The current objective has no direct pressure on repeat accesses, jump-size distribution, stride patterns, or working-set churn.
- That means the model can still look better on MMD than it does on replay-relevant locality.

Recommendation:

- Add explicit decoded-sequence penalties or diagnostics for:
- repeat-access rate
- signed `obj_id`-delta histogram
- run-length distribution of repeated/small-stride accesses
- working-set growth proxy over the window

If I had to choose one, I would start with a histogram-matching loss on signed `obj_id` deltas plus a repeat-rate loss.

### 8. Increase effective context before adding more paper-driven objective terms

The model is still fundamentally operating on `timestep=12` windows.

Relevant code:

- `llgan/config.py:9`
- `llgan/model.py:136-145`

Why this matters:

- A lot of the recent losses are compensating for missing temporal span.
- Even with hidden-state carry-over at generation time, the training signal is still local.
- Long bursts, phase shifts, and diurnal-style regime changes are going to be hard to learn inside 12-step windows.

Recommendation:

- Move one step toward hierarchy before moving toward more exotic GAN objectives.
- Good next options:
- patch embedding before the critic LSTM
- a coarse regime encoder per file/window chunk
- a two-level generator where a slow latent controls multiple fine windows

I would prioritize this above diffusion-style work.

## Evaluation Recommendations

### 9. Do not use the model's own encoder as the only Context-FID backbone for model selection

`eval.py` computes Context-FID using the checkpoint's own trained encoder.

Relevant code:

- `llgan/eval.py:160-207`
- `llgan/eval.py:357-360`

Why this matters:

- The feature extractor is co-adapted with the generator and recovery path.
- That can make the metric partially self-serving.
- It is useful as a diagnostic, but I would be careful about making it the primary checkpoint selector.

Recommendation:

- Keep the current Context-FID, but add an external feature space for selection:
- a frozen encoder trained only on real data
- or a downstream predictor/critic trained separately and never jointly with the generator

### 10. `best.pt` should not be picked on MMD alone

Right now the best-checkpoint logic keys off MMD only.

Relevant code:

- `llgan/train.py:588-611`

Why this matters:

- Your own TODO already shows cases where MMD looked good while DMD-GEN or recall still looked bad.
- MMD alone will keep selecting models that are too smooth or too mode-collapsed.

Recommendation:

- Promote a small multi-objective score for checkpoint selection.
- My suggested order of importance is:
- recall / coverage
- DMD-GEN
- Context-FID
- MMD

If you want a single scalar, use a weighted rank or weighted z-score rather than a hard-coded linear formula too early.

## What I Would Do Next, In Order

1. Fix `wgan-gp` generator loss.
2. Fix the stale/no-op supervisor behavior in joint training.
3. Fix `generate.py --n-streams` so multiple traces are not merged into one sequence.
4. Make validation file-held-out and boundary-safe.
5. Add the feature-space discriminator.
6. Add workload conditioning to `z_global`.
7. Add locality-specific losses/metrics on decoded sequences.
8. Only then spend more time on AGP, RpGAN, AVATAR, Sig-WGAN, or larger architectural swaps.

## Bottom Line

The repo is past the stage where the biggest gains come from reading another paper and adding another loss term. The main opportunity now is to tighten the current training/evaluation loop so the architecture you already built gets an honest signal.

Once the four correctness issues above are fixed, the most promising model improvement is a conditioned dual-critic latent AE, not a broader grab-bag of regularizers.
