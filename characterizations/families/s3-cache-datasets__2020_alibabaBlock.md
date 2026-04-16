# s3-cache-datasets / 2020_alibabaBlock

- Files: 1001
- Bytes: 251207505629
- Formats: oracle_general, text_zst
- Parsers: generic_text, oracle_general
- ML Use Cases: request_sequence, structured_table
- Heterogeneity Score: 1.322
- Suggested GAN Modes: 8
- Split By Format: yes

## Observations

- Predominantly read-heavy.
- Very weak short-window reuse.
- Ordered feature trajectories show regime boundaries.

## GAN Guidance

- Family spans multiple encodings; keep format-aware preprocessing and avoid blindly pooling structured-table and request-sequence variants.
- Ordered PC1 changepoints suggest 22 regimes when files are ordered by trace start time.
- Sequential blocks are much more internally coherent than random file batches; block or curriculum sampling is likely safer than pure iid file sampling.
- Opcode balance is extremely read-skewed; generation should not assume symmetric read/write behavior.
- Strongest feature coupling in this pass: iat_mean vs iat_q90 (corr=1).
- A small set of files are strong multivariate outliers; consider holding them out for ablation or separate mode inspection.
- Current characterization suggests extra conditioning value from: object_unique, signed_stride_lag1_autocorr, obj_size_std.

## Conditioning Audit

| Item | Value |
|---|---|
| Near-constant current conditioning features | write_ratio, iat_q50, opcode_switch_ratio, tenant_unique |
| Recommended candidate additions | object_unique, signed_stride_lag1_autocorr, obj_size_std |
| Highly redundant current pairs | forward_seek_ratio vs backward_seek_ratio (-0.966) |

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| oracle_general | 1000 | oracle_general |
| text_zst | 1 | generic_text |

## Clustering And Regimes

| Item | Value |
|---|---|
| K-means selected K | 2 |
| Best silhouette K | 2 |
| DBSCAN clusters | 2 |
| DBSCAN noise fraction | 0.071 |
| Ordered PC1 changepoints | 21 |
| PCA variance explained by PC1 | 0.219 |
| Hurst exponent on ordered PC1 | 0.978 |
| Block/random distance ratio | 0.556 |
| Sampling recommendation | block_sampling_preserves_temporal_coherence |

### K Selection

| K | Within-SS | Silhouette |
|---:|---:|---:|
| 2 | 17367942549319839100436480 | 0.881 |
| 3 | 9439268086990004914487296 | 0.797 |
| 4 | 5421116206412136687075328 | 0.78 |
| 5 | 4135199035010834856148992 | 0.775 |
| 6 | 4225301089994964469809152 | 0.695 |
| 7 | 3289078135122815751290880 | 0.667 |
| 8 | 3152950977043638601646080 | 0.633 |
| 9 | 3095274466820970202005504 | 0.434 |
| 10 | 3052373980863136893239296 | 0.437 |
| 11 | 3016765134300148374962176 | 0.465 |
| 12 | 2997584362132720328900608 | 0.4 |

## Regime Transition Drivers

| Transition | Driver 1 | Effect | Driver 2 | Effect | Driver 3 | Effect |
|---|---|---:|---|---:|---|---:|
| 1 -> 2 | size_bytes | 1.732 | iat_q90 | 1.5 | reuse_ratio | 1.414 |
| 2 -> 3 | reuse_ratio | 1.414 | abs_stride_q50 | 1.132 | burstiness_cv | 0.755 |
| 3 -> 4 | size_bytes | 3.27 | iat_std | 2.799 | iat_q99 | 2.546 |
| 4 -> 5 | obj_size_q50 | 1.414 | abs_stride_mean | 1.24 | abs_stride_q50 | 1.178 |
| 5 -> 6 | reuse_ratio | 2.236 | obj_size_q99 | 2.024 | obj_size_std | 2.008 |
| 6 -> 7 | abs_stride_q90 | 4.052 | abs_stride_q99 | 3.778 | abs_stride_std | 3.767 |
| 7 -> 8 | abs_stride_q99 | 3.305 | abs_stride_std | 2.256 | iat_zero_ratio | 2.173 |
| 8 -> 9 | object_top1_share | 2.414 | abs_stride_q99 | 2.406 | forward_seek_ratio | 1.882 |
| 9 -> 10 | obj_size_q90 | 27.813 | obj_size_q99 | 6.196 | obj_size_std | 5.45 |
| 10 -> 11 | obj_size_q90 | 56.568 | tenant_top1_share | 6.769 | obj_size_std | 4.364 |
| 11 -> 12 | obj_size_q99 | 15.122 | obj_size_mean | 14.76 | backward_seek_ratio | 6.131 |
| 12 -> 13 | burstiness_cv | 1.916 | backward_seek_ratio | 1.657 | tenant_top1_share | 1.655 |
| 13 -> 14 | obj_size_q50 | 1.414 | abs_stride_q50 | 1.354 | abs_stride_mean | 1.192 |
| 14 -> 15 | object_unique | 2.162 | object_top1_share | 1.438 | obj_size_q50 | 1.414 |
| 15 -> 16 | obj_size_q99 | 1.423 | abs_stride_q50 | 1.418 | obj_size_q90 | 1.414 |
| 16 -> 17 | iat_zero_ratio | 1.889 | reuse_ratio | 1.372 | ts_duration | 1.368 |
| 17 -> 18 | iat_q90 | 7.071 | iat_zero_ratio | 2.289 | iat_mean | 2.016 |
| 18 -> 19 | size_bytes | 1.849 | iat_std | 1.455 | iat_q99 | 1.429 |
| 19 -> 20 | abs_stride_q99 | 1.457 | abs_stride_std | 1.258 | obj_size_q90 | 1.16 |
| 20 -> 21 | reuse_ratio | 5.657 | iat_zero_ratio | 2.45 | obj_size_min | 2.34 |
| 21 -> 22 | sample_record_rate | 3.318 | reuse_ratio | 2 | abs_stride_q50 | 1.414 |

## Strongest Correlations

| Metric A | Metric B | Correlation |
|---|---|---:|
| iat_mean | iat_q90 | 0.995 |
| iat_std | iat_q99 | 0.973 |
| iat_mean | iat_std | 0.97 |
| forward_seek_ratio | backward_seek_ratio | -0.966 |
| obj_size_mean | obj_size_q50 | 0.966 |
| iat_mean | iat_q99 | 0.965 |
| obj_size_std | obj_size_q90 | 0.955 |
| obj_size_mean | obj_size_q90 | 0.95 |

## Metrics

| Metric | Mean | Median | CV | Skew | Kurtosis | Missing | Q10 | Q90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| size_bytes | 250956549 | 12159046 | 19.129 | 30.732 | 960.579 | 0 | 129190 | 118892594 |
| sample_records | 4051.168 | 4096 | 0.095 | -8.936 | 80.937 | 0 | 4096 | 4096 |
| ttl_present | 0 | 0 | N/A | N/A | N/A | 0 | 0 | 0 |
| iat_q90 | 226.127 | 1 | 24.788 | 29.696 | 908.719 | 0.001 | 0 | 6 |
| iat_mean | 74.817 | 0.463 | 21.934 | 29.062 | 880.817 | 0.001 | 0.004 | 4.549 |
| iat_q99 | 554.012 | 5 | 16.56 | 23.861 | 636.374 | 0.001 | 0 | 51.766 |
| iat_std | 189.843 | 1.372 | 14.377 | 24.964 | 697.598 | 0.001 | 0.1 | 16.471 |
| iat_q50 | 0.098 | 0 | 11.743 | 19.894 | 471.833 | 0.001 | 0 | 0 |
| ts_duration | 28093.1 | 1888 | 7.842 | 10.925 | 122.121 | 0.001 | 14 | 18630 |
| iat_lag1_autocorr | -0.022 | -0.009 | 7.7 | 0.44 | 6.978 | 0.001 | -0.171 | 0.1 |
| reuse_ratio | 0.004 | 0 | 6.5 | 18.349 | 389.668 | 0.001 | 0 | 0.007 |
| abs_stride_q50 | 4140612431 | 5566464 | 4.846 | 9.196 | 102.36 | 0.001 | 17305.6 | 4643355034 |
| sample_record_rate | 90.676 | 2.158 | 3.497 | 5.653 | 41.287 | 0.001 | 0.22 | 275.017 |
| obj_size_q50 | 41127.17 | 4096 | 2.753 | 3.095 | 8.121 | 0.001 | 4096 | 98124.8 |
| abs_stride_q99 | 68421707404 | 24548759921 | 2.343 | 11.239 | 190.592 | 0.001 | 13477125915 | 168754881462 |
| abs_stride_q90 | 30087000635 | 11194333594 | 2.339 | 6.558 | 67.002 | 0.001 | 14557594 | 65235863060 |
| abs_stride_mean | 10676758645 | 3774368419 | 2.291 | 5.711 | 43.55 | 0.001 | 1019158465 | 24258202243 |
| abs_stride_std | 17048914605 | 6147959222 | 2.156 | 7.651 | 87.506 | 0.001 | 3094033065 | 35565159045 |

## Outlier Files

| rel_path | outlier_score | top drivers |
|---|---:|---|
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_alibabaBlock/1K/alibabaBlock_809.oracleGeneral.zst | 676.741 | iat_q90 (z=173029.6); iat_mean (z=109473.6) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_alibabaBlock/alibabaBlock_207.oracleGeneral.zst | 102.674 | abs_stride_q99 (z=345.278); abs_stride_std (z=230.293) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_alibabaBlock/1K/alibabaBlock_811.oracleGeneral.zst | 99.494 | iat_q90 (z=37947.8); iat_mean (z=23900.27) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_alibabaBlock/1K/alibabaBlock_816.oracleGeneral.zst | 82.477 | iat_std (z=18030.15); iat_q99 (z=16200.08) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_alibabaBlock/alibabaBlock_791.oracleGeneral.zst | 60.919 | abs_stride_std (z=159.39); abs_stride_mean (z=133.708) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_alibabaBlock/alibabaBlock_369.oracleGeneral.zst | 50.513 | abs_stride_q99 (z=203.433); abs_stride_std (z=152.572) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_alibabaBlock/alibabaBlock_771.oracleGeneral.zst | 41.674 | abs_stride_q50 (z=48329.38); abs_stride_mean (z=101.652) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_alibabaBlock/10K/alibabaBlock_805.oracleGeneral.zst | 29.994 | abs_stride_q50 (z=32140.92); iat_q99 (z=4175.6) |

## Outlier Sensitivity

| N Removed | Metric | Baseline Median | Trimmed Median | Relative Shift |
|---:|---|---:|---:|---:|
| 10 | abs_stride_mean | 3774368419 | 3745788580 | -0.008 |
| 5 | abs_stride_mean | 3774368419 | 3767070647 | -0.002 |
| 10 | burstiness_cv | 3.828 | 3.834 | 0.002 |
| 5 | burstiness_cv | 3.828 | 3.833 | 0.002 |
| 3 | burstiness_cv | 3.828 | 3.833 | 0.001 |
| 1 | burstiness_cv | 3.828 | 3.833 | 0.001 |
| 1 | object_unique | 2181.5 | 2182 | 0 |
| 3 | object_unique | 2181.5 | 2182 | 0 |
| 5 | object_unique | 2181.5 | 2182 | 0 |
| 1 | abs_stride_mean | 3774368419 | 3774048681 | 0 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv | ts_duration |
|---|---|---:|---:|---:|---:|
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_alibabaBlock/alibabaBlock_4.oracleGeneral.zst | oracle_general | 0 | 0 | 63.984 | 1 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_alibabaBlock/alibabaBlock_810.oracleGeneral.zst | oracle_general | 0 | 0 | 63.899 | 55970 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_alibabaBlock/1M/alibabaBlock_808.oracleGeneral.zst | oracle_general | 0 | 0 | 54.095 | 211798 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_alibabaBlock/10K/alibabaBlock_821.oracleGeneral.zst | oracle_general | 0 | 0.012 | 51.321 | 606253 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_alibabaBlock/10K/alibabaBlock_822.oracleGeneral.zst | oracle_general | 0 | 0 | 45.625 | 956754 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_alibabaBlock/100K/alibabaBlock_831.oracleGeneral.zst | oracle_general | 0 | 0.001 | 45.238 | 2 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_alibabaBlock/100K/alibabaBlock_838.oracleGeneral.zst | oracle_general | 0 | 0.001 | 45.238 | 2 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_alibabaBlock/100K/alibabaBlock_839.oracleGeneral.zst | oracle_general | 0 | 0.001 | 45.238 | 2 |




## Model-Aware Guidance

- Closest learned anchor: alibaba (distance 0.539)
- Sampling: split-by-format-first
- Regime recipe: K≈8
- Char-file conditioning: yes
- PCF: promising
- Multi-scale critic: promising
- Mixed-type recovery: mixed
- Retrieval memory: unknown
- Why: ordered files show temporal persistence; family looks multi-regime or high-heterogeneity; formats/parsers are mixed
- Candidate conditioning additions: object_unique,signed_stride_lag1_autocorr,obj_size_std
