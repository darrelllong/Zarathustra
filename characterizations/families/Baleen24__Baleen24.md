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
| DBSCAN clusters | 3 |
| DBSCAN noise fraction | 0.045 |
| Ordered PC1 changepoints | 35 |
| PCA variance explained by PC1 | 0.337 |
| Hurst exponent on ordered PC1 | 0.961 |
| Block/random distance ratio | 0.342 |
| Sampling recommendation | block_sampling_preserves_temporal_coherence |

### K Selection

| K | Within-SS | Silhouette |
|---:|---:|---:|
| 2 | 14921684591957014528 | 0.976 |
| 3 | 11145993694647515136 | 0.784 |
| 4 | 223420333441869952 | 0.847 |
| 5 | 13496182607265103872 | 0.584 |
| 6 | 173950468462876544 | 0.373 |
| 7 | 127649585874184320 | 0.544 |
| 8 | 120075366507085600 | 0.56 |
| 9 | 116943402401459248 | 0.535 |
| 10 | 113821793875128368 | 0.548 |
| 11 | 110851714527838144 | 0.542 |
| 12 | 110218087189753760 | 0.515 |

## Regime Transition Drivers

| Transition | Driver 1 | Effect | Driver 2 | Effect | Driver 3 | Effect |
|---|---|---:|---|---:|---|---:|
| 1 -> 2 | sample_record_rate | 4.622 | tenant_unique | 3.637 | iat_min | 2.602 |
| 2 -> 3 | sample_record_rate | 4.697 | obj_size_mean | 4.334 | tenant_unique | 4.178 |
| 3 -> 4 | signed_stride_lag1_autocorr | 18.967 | iat_lag1_autocorr | 3.121 | size_bytes | 2.264 |
| 4 -> 5 | iat_std | 1961.59 | ts_duration | 1514.891 | iat_mean | 1514.891 |
| 5 -> 6 | obj_size_mean | 80.05 | forward_seek_ratio | 66.939 | abs_stride_mean | 49.886 |
| 6 -> 7 | obj_size_std | 17.785 | object_top1_share | 4.193 | obj_size_mean | 2.898 |
| 7 -> 8 | size_bytes | 339.913 | sample_record_rate | 54.458 | write_ratio | 21.871 |
| 8 -> 9 | forward_seek_ratio | 2.193 | write_ratio | 2.071 | burstiness_cv | 1.802 |
| 9 -> 10 | size_bytes | 524.776 | sample_record_rate | 28.41 | iat_mean | 6.149 |
| 10 -> 11 | abs_stride_q50 | 9.9 | tenant_unique | 6 | write_ratio | 5.365 |
| 11 -> 12 | signed_stride_lag1_autocorr | 5.964 | iat_min | 5.833 | object_unique | 5.657 |
| 12 -> 13 | tenant_top10_share | 24.042 | iat_min | 21.431 | object_unique | 11.314 |
| 13 -> 14 | object_top10_share | 7.155 | backward_seek_ratio | 5.856 | reuse_ratio | 4.49 |
| 14 -> 15 | size_bytes | 39.227 | tenant_unique | 13.4 | sample_record_rate | 10.538 |
| 15 -> 16 | write_ratio | 4.933 | abs_stride_std | 1.994 | tenant_top1_share | 1.636 |
| 16 -> 17 | iat_std | 276.49 | iat_q99 | 249.81 | tenant_unique | 169.548 |
| 17 -> 18 | obj_size_min | 61776.62 | tenant_unique | 260.215 | iat_std | 198.109 |
| 18 -> 19 | iat_std | 70.54 | iat_mean | 61.426 | ts_duration | 61.426 |
| 19 -> 20 | write_ratio | 57.091 | obj_size_mean | 13.301 | obj_size_q99 | 10.107 |
| 20 -> 21 | backward_seek_ratio | 30.507 | object_top1_share | 6.487 | abs_stride_std | 5.653 |
| 21 -> 22 | iat_zero_ratio | 7.793 | obj_size_min | 7.425 | object_unique | 6.506 |
| 22 -> 23 | obj_size_q99 | 4.817 | signed_stride_lag1_autocorr | 2.184 | backward_seek_ratio | 2.059 |
| 23 -> 24 | obj_size_q99 | 6.372 | tenant_unique | 4.802 | tenant_top10_share | 4.339 |
| 24 -> 25 | tenant_unique | 15 | object_top10_share | 4.991 | tenant_top10_share | 4.42 |
| 25 -> 26 | obj_size_mean | 3.576 | size_bytes | 1.828 | iat_zero_ratio | 1.533 |
| 26 -> 27 | obj_size_mean | 3.576 | size_bytes | 2.511 | obj_size_q90 | 2.216 |
| 27 -> 28 | sample_record_rate | 18.352 | size_bytes | 18.131 | iat_q90 | 17.678 |
| 28 -> 29 | object_top1_share | 4.487 | signed_stride_lag1_autocorr | 4.049 | reuse_ratio | 4.038 |
| 29 -> 30 | backward_seek_ratio | 3.379 | object_unique | 3.202 | burstiness_cv | 1.719 |
| 30 -> 31 | abs_stride_q90 | 4.745 | abs_stride_std | 4.67 | tenant_unique | 4.243 |
| 31 -> 32 | obj_size_min | 5.091 | abs_stride_std | 4.845 | abs_stride_q90 | 2.86 |
| 32 -> 33 | tenant_top10_share | 17.515 | obj_size_min | 5.091 | signed_stride_lag1_autocorr | 4.803 |
| 33 -> 34 | tenant_top10_share | 912.168 | signed_stride_lag1_autocorr | 143.819 | iat_lag1_autocorr | 41.879 |
| 34 -> 35 | abs_stride_std | 11.89 | iat_q99 | 10.069 | tenant_top10_share | 8.264 |
| 35 -> 36 | abs_stride_q50 | 67.144 | tenant_top10_share | 8.632 | object_top1_share | 8.468 |

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




## Model-Aware Guidance

- Closest learned anchor: tencent_block (distance 111.661)
- Sampling: block
- Regime recipe: K≈8
- Char-file conditioning: yes
- PCF: validated
- Multi-scale critic: promising
- Mixed-type recovery: promising
- Retrieval memory: mixed
- Why: ordered files show temporal persistence; family looks multi-regime or high-heterogeneity; reuse/locality is not negligible
- Candidate conditioning additions: object_unique,signed_stride_lag1_autocorr,obj_size_std
