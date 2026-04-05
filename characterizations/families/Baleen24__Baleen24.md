# Baleen24 / Baleen24

- Files: 374
- Bytes: 31376065904
- Formats: baleen24
- Parsers: baleen24
- ML Use Cases: request_sequence
- Heterogeneity Score: 0.423
- Suggested GAN Modes: 8
- Split By Format: no

## Observations

- Substantial write pressure across sampled files.
- Ordered feature trajectories show regime boundaries.

## GAN Guidance

- Ordered PC1 changepoints suggest 36 regimes when files are ordered by trace start time.
- Write pressure is material; preserve write bursts and opcode transitions in conditioning.
- Tenant diversity is high; tenant/context conditioning is likely useful.
- Strongest feature coupling in this pass: ts_duration vs iat_mean (corr=1).
- A small set of files are strong multivariate outliers; consider holding them out for ablation or separate mode inspection.

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| baleen24 | 374 | baleen24 |

## Clustering And Regimes

| Item | Value |
|---|---|
| DBSCAN clusters | 3 |
| DBSCAN noise fraction | 0.045 |
| Ordered PC1 changepoints | 35 |
| PCA variance explained by PC1 | 0.337 |

## Strongest Correlations

| Metric A | Metric B | Correlation |
|---|---|---:|
| ts_duration | iat_mean | 1 |
| iat_std | iat_q99 | 0.988 |
| reuse_ratio | backward_seek_ratio | -0.985 |
| reuse_ratio | forward_seek_ratio | -0.984 |
| reuse_ratio | object_top1_share | 0.984 |
| forward_seek_ratio | object_top1_share | -0.972 |
| backward_seek_ratio | object_top1_share | -0.965 |
| abs_stride_mean | abs_stride_q90 | 0.94 |

## Metrics

| Metric | Mean | Median | CV | Skew | Kurtosis | Missing | Q10 | Q90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sample_record_rate | 4.969 | 0.483 | 8.947 | 13.986 | 211.426 | 0 | 0.164 | 3.479 |
| size_bytes | 83893224 | 18261826 | 4.852 | 14.438 | 231.947 | 0 | 2864525 | 171296852 |
| iat_min | 0 | 0 | 2.802 | 3.491 | 12.884 | 0 | 0 | 0 |
| obj_size_min | 88337.03 | 27 | 1.916 | 3.328 | 11.205 | 0 | 3 | 131072 |
| iat_q99 | 18.694 | 13 | 1.758 | 6.521 | 54.686 | 0 | 2 | 39.242 |
| iat_std | 4.036 | 2.911 | 1.696 | 5.774 | 42.329 | 0 | 0.502 | 8.124 |
| iat_q90 | 6.723 | 6 | 1.34 | 4.905 | 35.723 | 0 | 1 | 15.831 |
| tenant_unique | 120.971 | 28 | 1.284 | 1.35 | 0.784 | 0 | 18 | 341.8 |
| iat_lag1_autocorr | 0.087 | 0.103 | 1.145 | -0.41 | 0.453 | 0 | -0.073 | 0.173 |
| ts_duration | 10165.25 | 8474 | 1.114 | 2.296 | 7.059 | 0 | 1177.479 | 24931.6 |
| iat_mean | 2.482 | 2.069 | 1.114 | 2.296 | 7.059 | 0 | 0.288 | 6.088 |
| iat_zero_ratio | 0.27 | 0.246 | 1.049 | 0.582 | -1.09 | 0 | 0 | 0.734 |
| iat_q50 | 1.025 | 1 | 0.957 | 0.887 | -0.051 | 0 | 0 | 2.571 |
| object_unique | 575.396 | 722 | 0.839 | 0.086 | -1.557 | 0 | 67 | 1163.9 |
| abs_stride_q50 | 1266830 | 1215373 | 0.596 | 0.533 | 3.028 | 0 | 0 | 2097152 |
| reuse_ratio | 0.368 | 0.311 | 0.57 | 1.926 | 2.746 | 0 | 0.202 | 0.743 |
| abs_stride_q99 | 11387090 | 9877586 | 0.493 | 1.091 | 3.215 | 0 | 7602176 | 19124757 |
| obj_size_q50 | 1124316 | 1048576 | 0.443 | 1.677 | 5.44 | 0 | 786432 | 1703936 |

## Outlier Files

| rel_path | outlier_score |
|---|---:|
| Baleen24/extracted/storage_0.1_10/storage/201910/Region2/full_0.2_0.1.trace | 47.488 |
| Baleen24/extracted/storage_10/storage/201910/Region2/full_0.2_0.1.trace | 47.488 |
| Baleen24/extracted/storage/storage/201910/Region2/full_0.2_0.1.trace | 47.488 |
| Baleen24/extracted/storage_0.1_10/storage/202110/Region4/full_0.1_0.1.trace | 45.75 |
| Baleen24/extracted/storage_10/storage/202110/Region4/full_0.1_0.1.trace | 45.75 |
| Baleen24/extracted/storage/storage/202110/Region4/full_0.1_0.1.trace | 45.75 |
| Baleen24/extracted/storage_all_Region3/201910/Region3/full.trace | 27.506 |
| Baleen24/extracted/storage_0.1_10/storage/201910/Region3/full_0.3_0.1.trace | 15.964 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv | ts_duration |
|---|---|---:|---:|---:|---:|
| Baleen24/extracted/storage_0.1_10/storage/201910/Region2/full_0.2_0.1.trace | baleen24 | 0.844 | 0.854 | 4.278 | 59175.57 |
| Baleen24/extracted/storage_10/storage/201910/Region2/full_0.2_0.1.trace | baleen24 | 0.844 | 0.854 | 4.278 | 59175.57 |
| Baleen24/extracted/storage/storage/201910/Region2/full_0.2_0.1.trace | baleen24 | 0.844 | 0.854 | 4.278 | 59175.57 |
| Baleen24/extracted/storage_10/storage/202110/Region4/full_3_1.trace | baleen24 | 0.666 | 0.201 | 2.398 | 1299 |
| Baleen24/extracted/storage/storage/202110/Region4/full_3_1.trace | baleen24 | 0.666 | 0.201 | 2.398 | 1299 |
| Baleen24/extracted/storage_10/storage/202110/Region4/full_7_1.trace | baleen24 | 0.69 | 0.204 | 2.196 | 1195 |
| Baleen24/extracted/storage/storage/202110/Region4/full_7_1.trace | baleen24 | 0.69 | 0.204 | 2.196 | 1195 |
| Baleen24/extracted/storage_10/storage/202110/Region4/full_1_1.trace | baleen24 | 0.699 | 0.193 | 2.141 | 1334 |
