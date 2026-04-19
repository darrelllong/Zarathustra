# Peer Review from Gemini

We use the latest Gemini Pro models to review the code.

## Round 1

### Code-Level Findings: Math, State, and Inference Bugs

While previous rounds reviewed architecture and training dynamics, this pass focuses on direct code-level logic. The codebase currently contains several undetected bugs that actively distort metrics or break the newest model features at evaluation time.

1. `[P1]` `compute_window_descriptors` has a silent shape mismatch with `cond_dim=13`. In `llgan/dataset.py`, the fallback method `compute_window_descriptors` is hardcoded to return a `(B, 10)` tensor. However, `profile_to_cond_vector` was expanded to 13 dimensions to support the R-analysis additions. If `cond_dim=13` is set in the config, models instantiated with it expect a 13-dim descriptor. The fallback in `eval.py` (`cond = compute_window_descriptors(rw)`) will either silently concatenate a 10-dimensional condition into a shape mismatch further down, or crash outright.

2. `[P1]` The Cache Fidelity (HRC-MAE) metric contains an artificial padding bug that significantly deflates the reported error limit. In `eval.py`'s `_per_window_hrc`, cache sizes are uniformly sampled using `np.unique(np.linspace(1, T, n_pts).astype(int))`. For a window of `T=12` and `n_pts=20`, this yields exactly 12 unique integers. The array is then padded out to length 20 using the last hit ratio (which is typically high/1.0). Because both `real` and `fake` arrays repeat their final value 9 times, the MAE average over the 20 points over-weights the large cache-size tail where error is minimal. This mathematically suppresses the measured error at smaller, more discriminative cache sizes, making the generated workloads look much better at cache fidelity than they truly are.

3. `[P1]` Conditional evaluation completely bypasses `cond_encoder`, `regime_sampler`, and `gmm_components` (again). In `llgan/eval.py`'s `_sample_fake`, `z_global` is constructed via a direct `z_g = torch.cat([cond, noise], dim=1)`. However, `train.py`'s `_make_z_global` properly routes `cond` through `G.cond_encoder` and `G.regime_sampler`, and models `noise` via `G.sample_noise()`. By bypassing these components at inference, `eval.py` passes raw numerical attributes into a generator that expects encoded regime factors and specialized noise. This renders the evaluation of `v86` or any models utilizing these newer modules invalid.

4. `[P2]` The `_reuse_rate` metric is still evaluating the wrong features due to hardcoded column indices. Despite earlier comments acknowledging the locality split, the function computes `reuse = seqs[:, 1:, 3]`. If the `tenant` column is auto-dropped during preprocessing—a known behavior for single-tenant traces—then `obj_id_reuse` shifts to index 2. The function will then incorrectly measure `obj_id_stride` or another feature. This needs to be dynamically resolved using `prep.col_names` like it is in `train.py`.

5. `[P2]` The preprocessor fit creates cross-run leakage via inconsistent seeding. In `train.py`, `_fit_prep_on_files` picks `n_seed` files via `random.sample`. Even though `random.seed(seed)` is called, experiments using different initial seeds will fit slightly different min/max bounds and potentially auto-drop different zero-variance columns. This explains why reproducing previous bests with fresh seeds fails: the preprocessor mappings and schema physically change between runs. The preprocessor manifest must be fit universally or frozen independently of the generator's stochastic components.

### What I Would Fix Before Another Run

1. Stop trusting the HRC-MAE metric out of `eval.py` until the cache size scaling and padding math is fixed to weight all sizes correctly.
2. Fix `_sample_fake` in `eval.py` (and any inference scripts) to use the fully encoded `z_global` rather than naive concatenation. Your best conditional models are currently being evaluated as garbage distributions.
3. Make `compute_window_descriptors` respect the dynamically configured `cond_dim` to prevent shape collapse when relying on fallback conditional inputs.

## Round 2

### Code-Level Findings: Math and Information Leakage Bugs

After reviewing the newest inference features and core metrics, here is the next set of code-level logic bugs. These issues do not overlap with Round 1's shape mismatch or the existing architectural critiques, but instead affect the Classifier-Free Guidance (CFG) validity and mathematical purity of the metrics.

1. `[P1]` **CFG Information Leakage through GMM Prior**: In `llgan/train.py`'s `_make_z_global`, the `noise` vector is generated conditionally upon `cond` (`noise = G.sample_noise(B, device, cond=cond)`) *before* the Classifier-Free Guidance (CFG) dropout mask is applied. When a batch item drops its condition (`cond=0`), its unique `noise` still retains the specific GMM cluster signature of its *original* workload. This allows the generator to "cheat" by deducing the missing condition from the sampled noise subspace, completely defeating CFG's goal of learning a true unconditional prior. The CFG dropout mask must be determined and applied *before* any downstream components (`cond_encoder`, `regime_sampler`, or `gmm_prior`) process the condition vector.

2. `[P2]` **Mathematically Biased MMD² Estimator**: The `mmd` function in `llgan/mmd.py` computes `_rbf_kernel(...).mean()`. This naturally includes the diagonal matrices where distance is 0 and the kernel evaluates to exactly `1.0`. While perfectly functional as a comparative distance proxy, claiming it as an "Unbiased" estimator (as noted in the docstring) is mathematically incorrect and introduces a systematic scaling bias of `1/N`. A mathematically pure unbiased estimator must explicitly subtract the trace (diagonal elements) before calculating the mean, otherwise the metric is artificially inflated for small evaluation samples.

3. `[P2]` **Zero-division instability in DMD-GEN**: The `_dmd_subspace` function in `llgan/mmd.py` calculates `A_tilde` using `np.diag(1.0 / S_r)` directly on the raw singular values. For synthetic trace sequences that suffer from mode-collapse (producing constant, unchanging feature values over 12 timesteps), the singular values `S_r` will aggressively approach absolute zero. Instead of catching this cleanly, `1.0 / S_r` floats explode to infinity, completely destroying the calculation and periodically throwing silent `NaNs`. Because early generator training naturally includes frequent mode-collapse, the DMD metric needs to use a pseudo-inverse like `np.linalg.pinv(np.diag(S_r))` or enforce a strict `S_r > tol` threshold cut-off to remain numerically robust across all epochs.

## Round 3

1. `[P1]` **Mathematical Bug: `boundary_latent_smoothness` Forces a Palindrome**
**Location:** `llgan/chunk_stitching.py`
**Issue:** The chunk stitching logic intended to enforce continuity across adjacent generated windows contains a severe mathematical flaw. It computes the MSE between the trailing steps of chunk A and the leading steps of chunk B, but it reverses chunk A using `.flip(dims=[1])`. 
- **Effect:** This compares `A[T-1]` to `B[0]` and `A[T-2]` to `B[1]`. For a continuous sequence, enforcing `A[T-2] == B[1]` penalizes any directional trend and forces the generated time-series to have zero velocity at the boundary, turning the time-series into a local palindrome. This destroys directional temporal dynamics and explains why continuity losses fail to improve long-horizon realism.
- **Fix:** Remove the `.flip(dims=[1])` and adjust the comparison so it correctly compares equivalent overlapped points, or compute a forward derivative difference `(B[1]-B[0]) - (A[T-1]-A[T-2])` to penalize sharp velocity shifts without halting the trajectory.

2. `[P1]` Train/Eval Mismatch: Retrieval Memory is Never Saturated During Training
**Location:** `llgan/train.py` and `llgan/retrieval_memory.py`
**Issue:** The newly added `RetrievalMemory` is designed to be persistent across very long sequences during evaluation (up to 100,000 steps). However, during training in `train.py`, `retrieval_state` is completely omitted from the recurrent thread. 
- **Effect:** The generator only trains on isolated `T=12` step windows with a fresh memory each time. Since the default memory capacity is `mem_size=32`, the memory never actually fills up during training. The critical logic for memory eviction (`evict_score`) and attending over a saturated memory space never receives any supervision. 
- **Fix:** `train.py` must maintain and thread `retrieval_state` across sequential training chunks, just as it does with `h_carry` for the RNN/SSM backbone. Without this, persistent retrieval is completely out-of-distribution during long rollouts.

3. `[P2]` **Legacy `boundary` Mode for Overlap Consistency Invokes the Palindrome Bug**
**Location:** `llgan/train.py` (around line 1784)
**Issue:** When using the legacy `--overlap-consistency-mode boundary`, the `train.py` script mistakenly invokes `boundary_latent_smoothness` to compute feature-space MSE.
- **Effect:** This applies the exact same palindrome-forcing zero-velocity boundary conditions to the feature space (`feat_c1`, `feat_c2`). The WaveStitch-style `overlap` mode (the new default) correctly uses the time-aligned `overlap_consistency` and is unaffected.
- **Fix:** If the legacy boundary mode is to be preserved for provenance, write a dedicated continuous feature MSE loss rather than reusing `boundary_latent_smoothness`'s flipped logic.
