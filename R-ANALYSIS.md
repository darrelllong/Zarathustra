# R Analysis

## Scope

This document describes exactly what the R-based characterization pipeline did.

Important boundary:

- Raw trace decoding was not done in R.
- Raw trace decoding and first-pass per-file profiling were done in Python in [parsers/core.py](/Users/darrell/Zarathustra/parsers/core.py).
- R consumed the normalized per-file characterization JSONL produced from that parser layer and then performed family-level statistical analysis, clustering, regime detection, outlier scoring, and report generation.

The main R entrypoints are:

- [R-scripts/extract_family_features.R](/Users/darrell/Zarathustra/R-scripts/extract_family_features.R)
- [R-scripts/analyze_family.R](/Users/darrell/Zarathustra/R-scripts/analyze_family.R)
- [R-scripts/render_family_report.R](/Users/darrell/Zarathustra/R-scripts/render_family_report.R)
- [R-scripts/run_corpus_analysis.R](/Users/darrell/Zarathustra/R-scripts/run_corpus_analysis.R)

## Data Flow

1. Python produced normalized per-file rows in:
   `trace_characterizations.normalized.jsonl`
2. R flattened each per-file JSON profile into a rectangular feature table.
3. R grouped rows by `logical_family_id = paste(dataset, family, sep = "__")`.
4. R ran family-level analysis on each group's `features.csv`.
5. R wrote:
   - `analysis.json`
   - `metric_summary.csv`
   - optional `cluster_assignments.csv`
   - optional `top_correlations.csv`
   - optional `outliers.csv`
   - family Markdown report

## What R Read From Python

R did not re-open trace binaries. It used parser-derived scalar summaries such as:

- request-sequence fields:
  `ts_duration`, `sample_record_rate`, `iat_zero_ratio`, `iat_lag1_autocorr`, `burstiness_cv`, `iat_*`, `obj_size_*`, `write_ratio`, `opcode_switch_ratio`, `reuse_ratio`, `forward_seek_ratio`, `backward_seek_ratio`, `signed_stride_lag1_autocorr`, `abs_stride_*`, `tenant_*`, `object_*`, `response_time_*`, `lcs_version`, `ttl_present`
- aggregate time-series fields:
  `sampling_interval_*`, `read_iops_*`, `write_iops_*`, `total_iops_*`, `read_bw_*`, `write_bw_*`, `total_bw_*`, `disk_usage_*`, `idle_ratio`, `write_share_iops_mean`, `total_iops_lag1_autocorr`, `disk_usage_lag1_autocorr`
- structured-table fields:
  `schema_column_count`, `schema_numeric_cols`, `schema_mixed_cols`, `schema_high_cardinality_cols`, `first_numeric_monotone_ratio`, `first_numeric_diff_*`

Those values were flattened in [R-scripts/extract_family_features.R](/Users/darrell/Zarathustra/R-scripts/extract_family_features.R).

## Parser-Side Math Feeding R

These formulas were computed in Python and then passed into R.

For request-sequence traces:

- inter-arrival times:
  `iat_i = max(0, ts_i - ts_{i-1})`
- duration:
  `ts_duration = ts_last - ts_first`
- sample record rate:
  `sample_record_rate = n / ts_duration` if `ts_duration > 0`
- zero inter-arrival ratio:
  `iat_zero_ratio = count(iat_i = 0) / len(iat)`
- burstiness coefficient of variation:
  `burstiness_cv = sd(iat) / mean(iat)` when `mean(iat) != 0`
- write ratio:
  `write_ratio = count(op < 0) / len(op)`
- opcode switch ratio:
  `opcode_switch_ratio = count(op_i != op_{i-1}) / (n - 1)`
- object-id deltas:
  `delta_i = obj_id_i - obj_id_{i-1}`
- reuse ratio:
  `reuse_ratio = count(delta_i = 0) / len(delta)`
- forward seek ratio:
  `forward_seek_ratio = count(delta_i > 0) / len(delta)`
- backward seek ratio:
  `backward_seek_ratio = count(delta_i < 0) / len(delta)`
- lag-1 autocorrelation:
  `corr(x_t, x_{t-1})` using the usual Pearson numerator and denominator

For aggregate time-series traces:

- sampling intervals:
  `dt_i = max(0, ts_i - ts_{i-1})`
- total IOPS:
  `total_iops_i = read_iops_i + write_iops_i`
- idle ratio:
  `idle_ratio = count(total_iops_i = 0) / n`
- write IOPS share:
  `write_share_iops_mean = sum(write_iops) / sum(total_iops)` when total is nonzero

For all numeric summary blocks:

- quantiles were linear-interpolated empirical quantiles at `q50`, `q90`, `q99`
- standard deviation was population-style:
  `sqrt(sum((x_i - mean(x))^2) / n)`

## R Feature Extraction

R's flattening step was deterministic field extraction. No estimation happened here beyond:

- converting missing JSON paths to `NA`
- turning booleans like `ttl_present` into `0/1`
- counting structured-schema properties:
  - `schema_column_count = length(columns)`
  - `schema_numeric_cols = count(numeric_ratio >= 0.8)`
  - `schema_mixed_cols = count(0.2 < numeric_ratio < 0.8)`
  - `schema_high_cardinality_cols = count(token_summary.unique >= 100)`

## R Summary Statistics

For each family and each numeric feature, R computed the following in [R-scripts/analyze_family.R](/Users/darrell/Zarathustra/R-scripts/analyze_family.R):

- finite sample count:
  `n = count(is.finite(x))`
- missing fraction:
  `missing_frac = mean(!is.finite(x_raw))`
- mean:
  `mean(x)`
- median:
  `median(x)`
- MAD:
  `mad(x, center = median(x), constant = 1)`
- standard deviation:
  `sd(x)` from R's sample standard deviation
- coefficient of variation:
  `cv = sd(x) / abs(mean(x))` when `abs(mean(x)) > 1e-12`
- skewness:
  `e1071::skewness(x, type = 2)` when available, else `moments::skewness(x)`
- kurtosis:
  `e1071::kurtosis(x, type = 2)` when available, else `moments::kurtosis(x)`
- quantiles:
  `q10`, `q25`, `q75`, `q90` using `stats::quantile(..., type = 7)`
- min and max

These were written to `metric_summary.csv`.

## Heterogeneity Score

R defined a family-level `heterogeneity_score` as:

- compute `cv` for every numeric metric with valid data
- clip each `cv` at `25`
- take the median of the clipped values

So:

`heterogeneity_score = median(min(cv_j, 25))` over all finite metric CVs

Interpretation:

- higher means larger cross-file dispersion across many features
- near zero means files within the family look very similar in the available feature space

## Feature Matrix Used For Multivariate Analysis

R built a numeric feature matrix as follows:

1. keep only columns listed in `numeric_columns`
2. coerce all to numeric
3. keep columns with at least `max(4, floor(0.5 * n_rows))` finite entries
4. keep only complete-case rows for PCA and clustering
5. drop zero-variance columns

This means PCA and clustering ran on a stricter subset than the univariate summaries.

## Correlation Analysis

R computed:

`cor_mat = cor(feature_matrix, use = "pairwise.complete.obs")`

Then it enumerated all off-diagonal pairs `(i, j)` and ranked them by:

`abs_correlation = abs(cor_mat[i, j])`

The top 8 pairs were written into the report.

## PCA

R ran:

`prcomp(numeric_matrix, center = TRUE, scale. = TRUE)`

So the PCA math was on z-scored columns:

`z_ij = (x_ij - mean_j) / sd_j`

Outputs used:

- PC scores
- rotation/loadings
- variance explained from `summary(pca)$importance`
- `pc1_variance = importance[2, 1]`

Plots written:

- `pca_scree.png`
- `pca_scatter.png`
- `pc_order.png`

## Clustering

R ran three clustering styles where data size permitted.

### K-means

R used:

`k = min(6, n_rows - 1)`

then:

`kmeans(numeric_matrix, centers = k, nstart = 20)`

### Gaussian Mixture Model

R used `mclust::Mclust` with:

`G = 1:min(6, n_rows - 1)`

Outputs retained:

- selected component count
- model name
- best BIC
- mean uncertainty

### DBSCAN

R set:

- `minPts = max(4, min(20, floor(log(n_rows) * 3)))`
- `eps = quantile(kNNdist(numeric_matrix, k = minPts), 0.90)`

then ran:

`dbscan(numeric_matrix, eps = eps, minPts = minPts)`

Outputs retained:

- number of nonzero clusters
- noise fraction:
  `mean(cluster == 0)`

## Outlier Scoring

Outliers were computed from PCA scores.

Let `S` be the retained PCA score matrix after dropping zero-variance PCs.

Primary score:

`d_i^2 = (s_i - mu)^T Sigma^{-1} (s_i - mu)`

using `stats::mahalanobis`.

Fallback if covariance inversion failed:

`outlier_score_i = sum(scale(S)_i^2)`

The top 8 files by outlier score were written to the report.

## Regime Detection

R attempted regime detection only when:

- `ts_start` existed
- PCA existed
- at least 8 ordered files were available

Procedure:

1. order files by `(ts_start, rel_path)`
2. merge ordered files with PCA scores
3. take the ordered `PC1` series
4. run changepoint detection:

`changepoint::cpt.meanvar(series, method = "PELT", penalty = "SIC")`

Outputs:

- `changepoint_count`
- changepoint indices

R also computed optional `tsfeatures` on the ordered `PC1` series:

- entropy
- lumpiness
- stability
- hurst
- flat_spots

## Suggested GAN Modes

R defined:

- `mclust_modes = components from Mclust`, if available
- `regime_modes = changepoint_count + 1`, if available

Then:

`suggested_modes = max(1, min(8, mclust_modes), min(8, regime_modes))`

So the recommendation is intentionally capped at 8 for practicality even if changepoint counts are much larger.

## Split-By-Format Flag

R set:

`split_by_format = (number of unique formats > 1) OR (number of unique parsers > 1)`

This is a hard warning that a family likely should not be pooled naively in GAN training.

## Textual GAN Guidance

The Markdown guidance bullets were rule-based, not learned.

Examples:

- if `split_by_format = TRUE`, warn to split by encoding
- if `heterogeneity_score > 2.5`, warn against single unconditional training
- if mixture components >= 3, note multiple modes
- if changepoints exist, note regime structure
- if read share > 0.9, warn that opcode priors are strongly read-skewed
- if write ratio > 0.6, emphasize write-burst preservation
- if `reuse_ratio` is high, emphasize locality-aware losses and conditioning
- if `burstiness_cv > 10`, emphasize inter-arrival, FFT, and ACF losses
- if tenant diversity is high, emphasize tenant/context conditioning

## Resource Policy

During remote execution on `vinge.local`:

- jobs ran under `nice -n 10`
- jobs ran under `ionice -c3`
- BLAS/OpenMP thread env vars were pinned to 1
- the R orchestration script detected active `train.py` jobs and reported `worker_cap=6`
- the heavy parser normalization was not rerun for the deep R pass; R reused the existing normalized corpus

## Outputs Produced

Repo outputs:

- [characterizations/README.md](/Users/darrell/Zarathustra/characterizations/README.md)
- [characterizations/families.csv](/Users/darrell/Zarathustra/characterizations/families.csv)
- [characterizations/rollup.json](/Users/darrell/Zarathustra/characterizations/rollup.json)
- [characterizations/families](/Users/darrell/Zarathustra/characterizations/families)

Remote heavy outputs:

- `/tiamat/zarathustra/r-output/deep_results_20260405_1/`

## Limitations

- R did not estimate raw sequence distributions from trace binaries directly.
- The family analysis is only as rich as the parser-derived per-file profile surface.
- Structured-table families like parquet and text schemas currently get more schema and column-shape analysis than true request-dynamics analysis.
- `suggested_modes` is a practical heuristic, not a proven optimal number of latent regimes.
