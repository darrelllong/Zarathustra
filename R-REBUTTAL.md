# R Characterization: Gaps, Critiques, and Missed Opportunities

Review of the R-based family characterization (30,628 files, 30 families) with focus
on the two training corpora: tencent_block (4,996 files) and alibaba (1,000 files).

---

## 1. Missing Conditioning Features

The GAN conditioning vector (cond_dim=10) uses:

```
write_ratio, reuse_ratio, burstiness_cv, iat_q50, obj_size_q50,
opcode_switch_ratio, iat_lag1_autocorr, tenant_unique,
forward_seek_ratio, backward_seek_ratio
```

R measured but we **don't use**:

- **`object_unique`** — Tencent median=1,788, Alibaba median=2,182. Object diversity
  directly controls the reuse/seek balance the model must learn. A file with 50 unique
  objects has fundamentally different locality than one with 3,000.

- **`signed_stride_lag1_autocorr`** — Tencent median=-0.173, Alibaba median=-0.410.
  Alibaba's strong negative autocorrelation indicates alternating stride patterns
  (forward-backward-forward). This is a *different signal* from `iat_lag1_autocorr`
  (which captures temporal burstiness, not spatial pattern). The model has no way to
  know about stride memory from the current conditioning.

- **`obj_size_std`** — We use `obj_size_q50` but not the *spread*. A file with uniform
  4KB objects vs one with bimodal 4KB/1MB objects looks identical through q50 alone.
  Tencent obj_size_std CV=1.64, which is real variation.

**Recommendation:** Expand cond_dim 10 → 13 with `log(object_unique)`,
`signed_stride_lag1_autocorr`, and `log1p(obj_size_std)`.

---

## 2. Feature Redundancy in the Characterization

The R analysis measured many correlated features without collapsing them:

| Pair | Tencent r | Alibaba r |
|------|-----------|-----------|
| iat_mean ↔ iat_q50 | 0.961 | — |
| iat_mean ↔ iat_q90 | 0.912 | **0.995** |
| iat_std ↔ iat_q99 | 0.933 | 0.973 |
| abs_stride_mean ↔ abs_stride_q90 | 0.938 | — |
| abs_stride_mean ↔ abs_stride_std | 0.924 | — |
| obj_size_std ↔ obj_size_q90 | 0.911 | 0.955 |
| forward_seek ↔ backward_seek | — | **-0.966** |

Alibaba's iat_mean ↔ iat_q90 at r=0.995 means q90 is almost perfectly predictable
from the mean — the IAT distribution is so right-skewed that quantiles collapse into
the first two moments. The R analysis measured q10, q25, q50, q90, q99 for each feature
family but never performed decorrelation or factor analysis.

**What R should have done:** PCA or factor analysis *within* each feature group
(IAT quantiles, stride quantiles, object size quantiles) to extract 1-2 orthogonal
factors per group. This would have given a cleaner, lower-dimensional feature space
for both clustering and GAN conditioning.

**Impact on us:** Our cond_dim uses forward_seek AND backward_seek (r=-0.97 in
alibaba — literally `backward = 1 - forward - reuse`). That's a wasted conditioning
dimension. Similarly, `opcode_switch_ratio` and `write_ratio` are both zero for these
corpora — two more wasted dimensions after auto-drop of the column but before
conditioning cleanup.

---

## 3. Cluster Analysis Limitations

### K-means with fixed K=6

R used `k = min(6, n_rows - 1)` for k-means. This is arbitrary. The BIC-selected
GMM found the structure, but k-means at K=6 gave misleading cluster balance:

- Tencent: 74% in 2 clusters, 0.2% in cluster 3 (10 files)
- Alibaba: 78% in 1 cluster, 0.2% in cluster 2 (2 files)

A 2-file cluster is not a cluster — it's noise. K-means without silhouette analysis
or gap statistic gives no information about the *right* K.

**What R should have done:** Silhouette width analysis across K=2..20 to find the
elbow, or at minimum report the within-cluster sum of squares (WSS) curve. The GMM
BIC selection partially addresses this, but the k-means results as reported are
misleading about cluster structure.

### DBSCAN epsilon selection

R set `eps = quantile(kNNdist, 0.90)` — the 90th percentile of k-nearest-neighbor
distances. This is a reasonable heuristic but the choice of 0.90 is not justified.
Different quantiles (0.85, 0.95) would give different cluster counts and noise
fractions. The 8.2% noise fraction for Tencent (416 files) may overcount or
undercount true outliers.

**What R should have done:** Report the kNN-distance curve (the "elbow plot" for
DBSCAN) so the user can visually assess whether 0.90 is at the elbow or arbitrary.

---

## 4. Temporal Analysis Gaps

### Changepoint detection found structure but didn't characterize it

R found 23 changepoints for Tencent and 21 for Alibaba, then reported `suggested_modes
= min(8, count+1)`. But it didn't answer the key question: **what changes at each
changepoint?**

The PELT algorithm detects shifts in mean and variance of PC1. But which *features*
drive those shifts? Is it IAT changing? Stride changing? Object size changing?
Without per-changepoint feature attribution, we can't design regime-specific losses
or conditioning.

**What R should have done:** For each regime segment between consecutive changepoints,
compute the feature-level means and report which features differ most between adjacent
regimes. This would tell us whether regime transitions are driven by burstiness changes,
object size shifts, or access pattern switches.

### Changepoint spacing is informative but underexploited

Tencent changepoints are roughly evenly spaced (96-164 file gaps). Alibaba has a
two-phase structure: dense early changepoints (6 in 82 files, ~13-file gaps) then
sparse late (227-file gaps). This temporal asymmetry could inform curriculum training
(train on stable late regimes first, then introduce volatile early regimes).

**Not analyzed:** Whether the temporal structure correlates with calendar time
(daily/weekly cycles), trace collection campaigns, or infrastructure changes.

---

## 5. Hurst Exponent Interpretation

R reported Hurst exponents: Tencent=0.79, Alibaba=0.98.

Alibaba's H=0.98 is near the upper bound (1.0 = perfectly persistent series).
This means consecutive files' PC1 scores are almost perfectly correlated — each file
is nearly identical to its temporal neighbor in the feature space. This has a direct
implication: **random file sampling for alibaba loses the temporal structure entirely.**
Files are NOT exchangeable.

**What R should have done:** Test whether random file sampling (our current approach)
significantly degrades mode coverage compared to sequential or block-sampling. With
H=0.98, block sampling would produce batches with more realistic within-batch
diversity.

**Impact on us:** Our `files-per-epoch=12` random sampling treats files as i.i.d.
For alibaba (H=0.98), this is incorrect. The generator never sees the temporal
coherence between adjacent files. This may explain why alibaba training is harder
than tencent (H=0.79, weaker temporal dependence).

---

## 6. Outlier Analysis Gaps

R computed Mahalanobis outlier scores on PCA scores. The top outliers are extreme:

| Family | Rank 1 | Rank 2 | Ratio |
|--------|--------|--------|-------|
| Tencent | 1356.8 | 609.9 | 2.2× |
| Alibaba | 675.8 | 102.5 | 6.6× |

But R didn't:

1. **Profile the outliers** — what makes them outliers? Is it one feature or many?
   The Mahalanobis score is a single number. We had to manually discover the stride
   outlier problem by comparing individual features.

2. **Recommend exclusion thresholds** — at what outlier score should files be
   excluded from training? The 8.2% DBSCAN noise fraction (416 files) is one answer,
   but it may be too aggressive (loses 8% of training data) or too lenient (doesn't
   catch the stride extremes that caused v54's collapse).

3. **Test sensitivity** — how do summary statistics change when top-N outliers are
   removed? If removing 10 files shifts the stride CV from 2.4 to 1.1, those 10 files
   are dominating the distribution shape.

**What we discovered independently:** The stride outlier (89× median, 36σ) caused
mode collapse when sampled during training. R flagged it as an outlier but didn't
connect outlier → training instability risk. The missing step was a feature-level
outlier decomposition.

---

## 7. Missing Analyses

### No distributional tests

R computed summary statistics (mean, median, std, skewness, kurtosis, quantiles) but
never tested distributional fit. Are IAT distributions log-normal? Power-law?
Exponential mixture? Knowing the *family* of distributions would inform:
- Whether log-transform is the right preprocessing (it is for log-normal, not for
  power-law with α < 2)
- Whether the generator's Tanh output can represent the tails after inverse transform
- Whether KL/MMD metrics are even appropriate for heavy-tailed distributions

### No conditional independence tests

R measured pairwise correlations but not conditional independence. `iat_mean` and
`obj_size_mean` may be correlated through a confounding factor (e.g., both scale
with file size). Partial correlations or graphical model structure would reveal the
true dependency graph — crucial for knowing which features the GAN must capture
jointly vs which can be generated independently.

### No within-file temporal analysis

R characterized *across files* (each file = one row in the feature matrix). But
the GAN generates *within-file* sequences (windows of 100 timesteps). The
connection between file-level characterization and window-level generation quality
is assumed, not tested.

**Critical gap:** A file with `burstiness_cv=10` could have either (a) uniformly
bursty windows, or (b) a mix of calm and explosive windows. The GAN needs to capture
(b) for realistic generation, but the R analysis can't distinguish them.

### No cross-validation of suggested_modes

R's `suggested_modes = max(1, min(8, mclust_modes), min(8, regime_modes))` is a
heuristic capped at 8. But it was never validated against GAN output quality.
Does K=8 actually produce better combined scores than K=4 or K=12? The cap at 8
is "for practicality" but the actual changepoint counts (23, 21) suggest more
modes exist. An ablation study on K would be more informative than a heuristic.

---

## 8. Cross-Family Comparison: Structural Differences

| Property | Tencent | Alibaba | Implication |
|----------|---------|---------|-------------|
| Cluster balance | Bimodal (39%/35%) | Unimodal (78%) | Different mode coverage needs |
| Reuse ratio | 0.010 median | 0.0002 median | Tencent 50× more reuse |
| Backward seeking | 17.6% | 32.0% | Alibaba 2× more backward |
| Stride autocorr | -0.173 | -0.410 | Alibaba has strong alternating patterns |
| Hurst exponent | 0.79 | 0.98 | Alibaba files highly non-exchangeable |
| Outlier extremity | 2.2× ratio | 6.6× ratio | Alibaba has sharper outlier spike |
| Changepoint spacing | Even | Two-phase (dense early) | Different temporal curriculum |

**R's guidance said "do not pool."** Correct. But R didn't go further to recommend
*how* to leverage these differences — e.g., alibaba needs block sampling (H=0.98),
tencent needs balanced cluster sampling (bimodal), both need outlier exclusion but
at different thresholds.

---

## Summary of Actionable Gaps

1. **Add 3 conditioning features:** object_unique, signed_stride_lag1_autocorr, obj_size_std
2. **Remove 2 redundant features:** backward_seek_ratio (= 1 - forward - reuse), opcode_switch_ratio (always 0 after auto-drop)
3. **Per-changepoint feature attribution** — what drives each regime transition?
4. **Distributional fit tests** — log-normal vs power-law for IAT, stride, obj_size
5. **Outlier feature decomposition** — which features make each outlier extreme?
6. **Block sampling for alibaba** — H=0.98 means random sampling loses temporal structure
7. **K ablation for suggested_modes** — validate K=8 against K=4, K=12
8. **Within-file temporal analysis** — file-level stats don't capture window-level dynamics
