## Round 10

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
