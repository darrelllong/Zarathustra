# Baleen24 / Baleen24

- Files: 374
- Bytes: 31376065904
- Formats: baleen24
- Parsers: baleen24
- ML Use Cases: request_sequence
- Heterogeneity Score: 0.423
- Suggested GAN Modes: 1
- Split By Format: no

## Observations

- Substantial write pressure across sampled files.

## GAN Guidance

- Sequential blocks are much more internally coherent than random file batches; block or curriculum sampling is likely safer than pure iid file sampling.
- Write pressure is material; preserve write bursts and opcode transitions in conditioning.
- Tenant diversity is high; tenant/context conditioning is likely useful.
- Strongest feature coupling in this pass: ts_duration vs iat_mean (corr=1).
- A small set of files are strong multivariate outliers; consider holding them out for ablation or separate mode inspection.
- Current characterization suggests extra conditioning value from: object_unique, signed_stride_lag1_autocorr, obj_size_std.

## Conditioning Audit

| Item | Value |
|---|---|
| Near-constant current conditioning features | none flagged |
| Recommended candidate additions | object_unique, signed_stride_lag1_autocorr, obj_size_std |
| Highly redundant current pairs | reuse_ratio vs backward_seek_ratio (-0.985); reuse_ratio vs forward_seek_ratio (-0.984) |

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| baleen24 | 374 | baleen24 |

## Clustering And Regimes

| Item | Value |
|---|---|
| K-means selected K | 2 |
| Best silhouette K | 2 |
| PCA variance explained by PC1 | 0.337 |
| Block/random distance ratio | 0.339 |
| Sampling recommendation | block_sampling_preserves_temporal_coherence |

### K Selection

| K | Within-SS | Silhouette |
|---:|---:|---:|
| 2 | 14921684591957014528 | 0.976 |
| 3 | 13569605214776494080 | 0.843 |
| 4 | 223420333441869952 | 0.847 |
| 5 | 13489379162946998272 | 0.693 |
| 6 | 143194281612374240 | 0.697 |
| 7 | 145289363867156384 | 0.585 |
| 8 | 121658415783095664 | 0.536 |
| 9 | 115367004443761296 | 0.558 |
| 10 | 112800421530280720 | 0.544 |
| 11 | 111745680703889296 | 0.527 |
| 12 | 111965629863014432 | 0.512 |

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
| sample_record_rate | 4.969 | 0.483 | 8.947 | N/A | N/A | 0 | 0.164 | 3.479 |
| size_bytes | 83893224 | 18261826 | 4.852 | N/A | N/A | 0 | 2864525 | 171296852 |
| iat_min | 0 | 0 | 2.802 | N/A | N/A | 0 | 0 | 0 |
| obj_size_min | 88337.03 | 27 | 1.916 | N/A | N/A | 0 | 3 | 131072 |
| iat_q99 | 18.694 | 13 | 1.758 | N/A | N/A | 0 | 2 | 39.242 |
| iat_std | 4.036 | 2.911 | 1.696 | N/A | N/A | 0 | 0.502 | 8.124 |
| iat_q90 | 6.723 | 6 | 1.34 | N/A | N/A | 0 | 1 | 15.831 |
| tenant_unique | 120.971 | 28 | 1.284 | N/A | N/A | 0 | 18 | 341.8 |
| iat_lag1_autocorr | 0.087 | 0.103 | 1.145 | N/A | N/A | 0 | -0.073 | 0.173 |
| ts_duration | 10165.25 | 8474 | 1.114 | N/A | N/A | 0 | 1177.479 | 24931.6 |
| iat_mean | 2.482 | 2.069 | 1.114 | N/A | N/A | 0 | 0.288 | 6.088 |
| iat_zero_ratio | 0.27 | 0.246 | 1.049 | N/A | N/A | 0 | 0 | 0.734 |
| iat_q50 | 1.025 | 1 | 0.957 | N/A | N/A | 0 | 0 | 2.571 |
| object_unique | 575.396 | 722 | 0.839 | N/A | N/A | 0 | 67 | 1163.9 |
| abs_stride_q50 | 1266830 | 1215373 | 0.596 | N/A | N/A | 0 | 0 | 2097152 |
| reuse_ratio | 0.368 | 0.311 | 0.57 | N/A | N/A | 0 | 0.202 | 0.743 |
| abs_stride_q99 | 11387090 | 9877586 | 0.493 | N/A | N/A | 0 | 7602176 | 19124757 |
| obj_size_q50 | 1124316 | 1048576 | 0.443 | N/A | N/A | 0 | 786432 | 1703936 |

## Outlier Files

| rel_path | outlier_score | top drivers |
|---|---:|---|
| Baleen24/extracted/storage_0.1_10/storage/201910/Region2/full_0.2_0.1.trace | 47.488 | obj_size_min (z=5241.8); iat_q99 (z=30.567) |
| Baleen24/extracted/storage_10/storage/201910/Region2/full_0.2_0.1.trace | 47.488 | obj_size_min (z=5241.8); iat_q99 (z=30.567) |
| Baleen24/extracted/storage/storage/201910/Region2/full_0.2_0.1.trace | 47.488 | obj_size_min (z=5241.8); iat_q99 (z=30.567) |
| Baleen24/extracted/storage_0.1_10/storage/202110/Region4/full_0.1_0.1.trace | 45.75 | obj_size_min (z=50.88); obj_size_q99 (z=-38.605) |
| Baleen24/extracted/storage_10/storage/202110/Region4/full_0.1_0.1.trace | 45.75 | obj_size_min (z=50.88); obj_size_q99 (z=-38.605) |
| Baleen24/extracted/storage/storage/202110/Region4/full_0.1_0.1.trace | 45.75 | obj_size_min (z=50.88); obj_size_q99 (z=-38.605) |
| Baleen24/extracted/storage_all_Region3/201910/Region3/full.trace | 27.506 | obj_size_min (z=5241.8); sample_record_rate (z=2302.995) |
| Baleen24/extracted/storage_0.1_10/storage/201910/Region3/full_0.3_0.1.trace | 15.964 | obj_size_min (z=5241.8); obj_size_std (z=-14.395) |

## Outlier Sensitivity

| N Removed | Metric | Baseline Median | Trimmed Median | Relative Shift |
|---:|---|---:|---:|---:|
| 10 | object_unique | 722 | 766 | 0.061 |
| 3 | object_unique | 722 | 760 | 0.053 |
| 5 | object_unique | 722 | 760 | 0.053 |
| 3 | reuse_ratio | 0.311 | 0.309 | -0.005 |
| 10 | reuse_ratio | 0.311 | 0.31 | -0.002 |
| 5 | burstiness_cv | 1.415 | 1.413 | -0.002 |
| 10 | burstiness_cv | 1.415 | 1.414 | -0.001 |
| 1 | abs_stride_mean | 2351121 | 2352739 | 0.001 |
| 3 | abs_stride_mean | 2351121 | 2352739 | 0.001 |
| 5 | abs_stride_mean | 2351121 | 2352739 | 0.001 |

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
