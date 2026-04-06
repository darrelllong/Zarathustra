# s3-cache-datasets / 2017_systor

- Files: 6
- Bytes: 34993572898
- Formats: text_zst
- Parsers: systor_text
- ML Use Cases: request_sequence
- Heterogeneity Score: 0.352
- Suggested GAN Modes: 1
- Split By Format: no

## Observations

- Very weak short-window reuse.

## GAN Guidance

- Strongest feature coupling in this pass: ts_duration vs iat_mean (corr=1).
- A small set of files are strong multivariate outliers; consider holding them out for ablation or separate mode inspection.
- Current characterization suggests extra conditioning value from: object_unique, signed_stride_lag1_autocorr, obj_size_std.

## Conditioning Audit

| Item | Value |
|---|---|
| Near-constant current conditioning features | tenant_unique |
| Recommended candidate additions | object_unique, signed_stride_lag1_autocorr, obj_size_std |
| Highly redundant current pairs | forward_seek_ratio vs backward_seek_ratio (-0.999) |

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| text_zst | 6 | systor_text |

## Clustering And Regimes

| Item | Value |
|---|---|
| PCA variance explained by PC1 | 0.523 |

## Strongest Correlations

| Metric A | Metric B | Correlation |
|---|---|---:|
| ts_duration | iat_mean | 1 |
| obj_size_q50 | schema_high_cardinality_cols | 1 |
| burstiness_cv | iat_std | 0.999 |
| forward_seek_ratio | backward_seek_ratio | -0.999 |
| ts_duration | burstiness_cv | 0.999 |
| burstiness_cv | iat_mean | 0.999 |
| ts_duration | iat_std | 0.998 |
| iat_mean | iat_std | 0.998 |

## Metrics

| Metric | Mean | Median | CV | Skew | Kurtosis | Missing | Q10 | Q90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| iat_std | 1.091 | 0.022 | 2.414 | 2.449 | 6 | 0 | 0.003 | 3.249 |
| abs_stride_q50 | 72462262955 | 8564356096 | 2.127 | 2.42 | 5.882 | 0 | 445407488 | 208377025280 |
| burstiness_cv | 11.68 | 2.943 | 1.942 | 2.44 | 5.964 | 0 | 1.334 | 30.762 |
| ts_duration | 95.122 | 28.281 | 1.87 | 2.424 | 5.904 | 0 | 9.171 | 247.913 |
| iat_mean | 0.023 | 0.007 | 1.87 | 2.424 | 5.904 | 0 | 0.002 | 0.061 |
| reuse_ratio | 0.002 | 0.001 | 1.167 | 1.187 | 0.67 | 0 | 0 | 0.005 |
| obj_size_q90 | 60074.67 | 32768 | 0.924 | 0.893 | -1.875 | 0 | 16384 | 131072 |
| sample_record_rate | 217.795 | 149.05 | 0.85 | 0.594 | -1.71 | 0 | 57.694 | 446.641 |
| obj_size_mean | 22318.75 | 13207.69 | 0.738 | 0.951 | -1.763 | 0 | 10308.12 | 43440.44 |
| iat_q99 | 0.088 | 0.101 | 0.716 | -0.351 | -2.044 | 0 | 0.014 | 0.149 |
| iat_lag1_autocorr | 0.075 | 0.089 | 0.7 | -0.539 | -1.155 | 0 | 0.012 | 0.124 |
| iat_zero_ratio | 0.001 | 0.001 | 0.601 | -0.224 | -1.864 | 0 | 0 | 0.002 |
| write_ratio | 0.469 | 0.479 | 0.564 | 0.185 | -1.723 | 0 | 0.19 | 0.739 |
| iat_q90 | 0.014 | 0.014 | 0.502 | 0.073 | -1.85 | 0 | 0.006 | 0.021 |
| object_top1_share | 0.007 | 0.006 | 0.481 | 0.364 | -0.326 | 0 | 0.004 | 0.01 |
| obj_size_std | 34254.03 | 30237.28 | 0.437 | 0.452 | -1.759 | 0 | 20309.48 | 52215.34 |
| object_top10_share | 0.035 | 0.033 | 0.425 | -0.166 | 0.478 | 0 | 0.021 | 0.051 |
| abs_stride_mean | 507137153926 | 480827761082 | 0.392 | 0.285 | -1.968 | 0 | 300684856942 | 739898843755 |

## Outlier Files

| rel_path | outlier_score | top drivers |
|---|---:|---|
| s3-cache-datasets/cache_dataset_txt/2017_systor/2016_LUN3.csv.sort.zst | 3.999 | iat_std (z=497.523); burstiness_cv (z=49.26) |
| s3-cache-datasets/cache_dataset_txt/2017_systor/2016_LUN4.csv.sort.zst | 3.698 | reuse_ratio (z=3.727); iat_q50 (z=2.67) |
| s3-cache-datasets/cache_dataset_txt/2017_systor/2016_LUN6.csv.sort.zst | 2.256 | iat_q50 (z=-4.239); object_top10_share (z=2.803) |
| s3-cache-datasets/cache_dataset_txt/2017_systor/2016_LUN0.csv.sort.zst | 2.067 | abs_stride_q50 (z=46.51); obj_size_mean (z=9.822) |
| s3-cache-datasets/cache_dataset_txt/2017_systor/2016_LUN1.csv.sort.zst | 1.557 | obj_size_q99 (z=9.521); size_bytes (z=-1.309) |
| s3-cache-datasets/cache_dataset_txt/2017_systor/2016_LUN2.csv.sort.zst | 1.423 | obj_size_mean (z=11.031); obj_size_q90 (z=6) |

## Outlier Sensitivity

| N Removed | Metric | Baseline Median | Trimmed Median | Relative Shift |
|---:|---|---:|---:|---:|
| 3 | reuse_ratio | 0.001 | 0 | -1 |
| 5 | reuse_ratio | 0.001 | 0 | -1 |
| 5 | obj_size_std | 30237.28 | 53698.07 | 0.776 |
| 3 | obj_size_std | 30237.28 | 50732.61 | 0.678 |
| 5 | burstiness_cv | 2.943 | 1.307 | -0.556 |
| 3 | burstiness_cv | 2.943 | 1.361 | -0.538 |
| 3 | abs_stride_mean | 480827761082 | 729780372283 | 0.518 |
| 5 | abs_stride_mean | 480827761082 | 729780372283 | 0.518 |
| 3 | object_unique | 3371.5 | 3892 | 0.154 |
| 5 | object_unique | 3371.5 | 3892 | 0.154 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv | ts_duration |
|---|---|---:|---:|---:|---:|
| s3-cache-datasets/cache_dataset_txt/2017_systor/2016_LUN3.csv.sort.zst | text_zst | 0.836 | 0.004 | 57.931 | 457.341 |
| s3-cache-datasets/cache_dataset_txt/2017_systor/2016_LUN6.csv.sort.zst | text_zst | 0.6 | 0.001 | 3.593 | 33.037 |
| s3-cache-datasets/cache_dataset_txt/2017_systor/2016_LUN4.csv.sort.zst | text_zst | 0.359 | 0.006 | 3.185 | 23.524 |
| s3-cache-datasets/cache_dataset_txt/2017_systor/2016_LUN1.csv.sort.zst | text_zst | 0.642 | 0 | 2.702 | 38.485 |
| s3-cache-datasets/cache_dataset_txt/2017_systor/2016_LUN0.csv.sort.zst | text_zst | 0.198 | 0.001 | 1.361 | 9.233 |
| s3-cache-datasets/cache_dataset_txt/2017_systor/2016_LUN2.csv.sort.zst | text_zst | 0.181 | 0 | 1.307 | 9.109 |
