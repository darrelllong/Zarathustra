# s3-cache-datasets / 2020_tencentBlock

- Files: 4996
- Bytes: 300469429369
- Formats: oracle_general, text_zst
- Parsers: generic_text, oracle_general
- ML Use Cases: request_sequence, structured_table
- Heterogeneity Score: 2.006
- Suggested GAN Modes: 8
- Split By Format: yes

## Observations

- Predominantly read-heavy.
- Very weak short-window reuse.
- Ordered feature trajectories show regime boundaries.

## GAN Guidance

- Family spans multiple encodings; keep format-aware preprocessing and avoid blindly pooling structured-table and request-sequence variants.
- Ordered PC1 changepoints suggest 24 regimes when files are ordered by trace start time.
- Sequential blocks are much more internally coherent than random file batches; block or curriculum sampling is likely safer than pure iid file sampling.
- Opcode balance is extremely read-skewed; generation should not assume symmetric read/write behavior.
- Strongest feature coupling in this pass: iat_mean vs iat_q50 (corr=0.96).
- A small set of files are strong multivariate outliers; consider holding them out for ablation or separate mode inspection.
- Current characterization suggests extra conditioning value from: object_unique, signed_stride_lag1_autocorr, obj_size_std.

## Conditioning Audit

| Item | Value |
|---|---|
| Near-constant current conditioning features | write_ratio, iat_q50, obj_size_q50, opcode_switch_ratio, tenant_unique |
| Recommended candidate additions | object_unique, signed_stride_lag1_autocorr, obj_size_std |
| Highly redundant current pairs | none flagged |

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| oracle_general | 4995 | oracle_general |
| text_zst | 1 | generic_text |

## Clustering And Regimes

| Item | Value |
|---|---|
| K-means selected K | 2 |
| Best silhouette K | 2 |
| DBSCAN clusters | 5 |
| DBSCAN noise fraction | 0.076 |
| Ordered PC1 changepoints | 23 |
| PCA variance explained by PC1 | 0.213 |
| Hurst exponent on ordered PC1 | 0.792 |
| Block/random distance ratio | 0.615 |
| Sampling recommendation | block_sampling_preserves_temporal_coherence |

### K Selection

| K | Within-SS | Silhouette |
|---:|---:|---:|
| 2 | 37088965710773430085746688 | 0.91 |
| 3 | 23026187318546070599892992 | 0.884 |
| 4 | 13944683680108564390608896 | 0.859 |
| 5 | 11313146636521699194961920 | 0.503 |
| 6 | 10121161864929031902724096 | 0.504 |
| 7 | 9743688138637375375933440 | 0.471 |
| 8 | 7891928727637300265091072 | 0.478 |
| 9 | 7787775750266413429817344 | 0.433 |
| 10 | 7508922458803506411208704 | 0.396 |
| 11 | 7671763868706897854988288 | 0.426 |
| 12 | 6007281203405707936792576 | 0.441 |

## Regime Transition Drivers

| Transition | Driver 1 | Effect | Driver 2 | Effect | Driver 3 | Effect |
|---|---|---:|---|---:|---|---:|
| 1 -> 2 | abs_stride_q50 | 1.221 | backward_seek_ratio | 1.055 | iat_q90 | 1 |
| 2 -> 3 | abs_stride_q99 | 1.357 | abs_stride_std | 1.192 | abs_stride_mean | 0.619 |
| 3 -> 4 | reuse_ratio | 2.334 | iat_q90 | 1.414 | obj_size_min | 1.414 |
| 4 -> 5 | object_unique | 1.229 | size_bytes | 1.16 | object_top1_share | 1.069 |
| 5 -> 6 | iat_q99 | 0.392 | object_top1_share | 0.388 | reuse_ratio | 0.363 |
| 6 -> 7 | obj_size_min | 8.485 | abs_stride_q99 | 2.482 | abs_stride_std | 2.074 |
| 7 -> 8 | abs_stride_q50 | 2.05 | abs_stride_q99 | 1.833 | abs_stride_q90 | 1.476 |
| 8 -> 9 | iat_q90 | 1.414 | iat_lag1_autocorr | 1.071 | signed_stride_lag1_autocorr | 0.875 |
| 9 -> 10 | iat_mean | 2.483 | ts_duration | 2.483 | size_bytes | 2.15 |
| 10 -> 11 | tenant_top1_share | 1.837 | signed_stride_lag1_autocorr | 1.619 | size_bytes | 1.563 |
| 11 -> 12 | obj_size_min | 1.414 | iat_q90 | 1.414 | reuse_ratio | 1.248 |
| 12 -> 13 | obj_size_min | 2.124 | iat_q90 | 1.414 | iat_q99 | 1.265 |
| 13 -> 14 | reuse_ratio | 0.652 | abs_stride_q50 | 0.407 | forward_seek_ratio | 0.387 |
| 14 -> 15 | backward_seek_ratio | 0.371 | object_top1_share | 0.35 | signed_stride_lag1_autocorr | 0.315 |
| 15 -> 16 | iat_q90 | 1.414 | iat_zero_ratio | 1.296 | backward_seek_ratio | 1.277 |
| 16 -> 17 | iat_q99 | 4.427 | ts_duration | 4.184 | iat_mean | 4.184 |
| 17 -> 18 | obj_size_min | 1.414 | abs_stride_q50 | 1.398 | obj_size_q99 | 1.181 |
| 18 -> 19 | sample_record_rate | 2.872 | iat_q90 | 2.828 | burstiness_cv | 2.528 |
| 19 -> 20 | abs_stride_q90 | 2.609 | sample_record_rate | 2.388 | tenant_top1_share | 1.932 |
| 20 -> 21 | iat_q99 | 0.969 | reuse_ratio | 0.466 | object_unique | 0.42 |
| 21 -> 22 | iat_q99 | 1 | obj_size_q99 | 0.636 | abs_stride_std | 0.545 |
| 22 -> 23 | iat_q99 | 1 | abs_stride_q50 | 0.757 | iat_q90 | 0.632 |
| 23 -> 24 | sample_record_rate | 2.465 | abs_stride_q90 | 2.267 | size_bytes | 2.117 |

## Strongest Correlations

| Metric A | Metric B | Correlation |
|---|---|---:|
| iat_mean | iat_q50 | 0.961 |
| abs_stride_mean | abs_stride_q90 | 0.938 |
| iat_std | iat_q99 | 0.933 |
| abs_stride_std | abs_stride_q99 | 0.926 |
| abs_stride_mean | abs_stride_std | 0.924 |
| iat_q90 | iat_q99 | 0.911 |
| iat_mean | iat_q90 | 0.911 |
| obj_size_std | obj_size_q90 | 0.911 |

## Metrics

| Metric | Mean | Median | CV | Skew | Kurtosis | Missing | Q10 | Q90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| size_bytes | 60141999 | 10893978 | 32.335 | 70.282 | 4957.972 | 0 | 1254170 | 44922228 |
| sample_records | 4032.378 | 4096 | 0.122 | -7.741 | 58.483 | 0 | 4096 | 4096 |
| ttl_present | 0 | 0 | N/A | N/A | N/A | 0 | 0 | 0 |
| ts_duration | 12915.69 | 1448 | 6.134 | 8.571 | 74.579 | 0 | 194.4 | 5645.2 |
| obj_size_q50 | 13172.49 | 4096 | 4.267 | 8.054 | 66.87 | 0 | 4096 | 8192 |
| obj_size_min | 2650.612 | 4096 | 3.063 | 54.087 | 3422.552 | 0 | 512 | 4096 |
| obj_size_mean | 20632.45 | 8785.875 | 2.29 | 5.863 | 39.18 | 0 | 5852.65 | 28838.88 |
| obj_size_q90 | 46053.73 | 16384 | 2.245 | 4.107 | 15.807 | 0 | 8192 | 65536 |
| object_top1_share | 0.039 | 0.019 | 1.768 | 8.366 | 100.673 | 0 | 0.006 | 0.08 |
| obj_size_std | 28368.46 | 13375 | 1.544 | 3.319 | 11.128 | 0 | 6567.901 | 63913.97 |
| obj_size_q99 | 116023.6 | 65536 | 1.24 | 2.225 | 3.431 | 0 | 28672 | 427540.5 |
| object_top10_share | 0.195 | 0.131 | 0.925 | 1.992 | 4.487 | 0 | 0.042 | 0.387 |
| object_unique | 1785.607 | 1788 | 0.473 | 0.433 | 0.166 | 0 | 727 | 2916.6 |
| tenant_top1_share | 0.927 | 0.954 | 0.097 | -2.366 | 6.355 | 0 | 0.831 | 0.997 |
| tenant_unique | 1.984 | 2 | 0.062 | -7.816 | 59.115 | 0 | 2 | 2 |
| tenant_top10_share | 1 | 1 | 0 | N/A | N/A | 0 | 1 | 1 |
| write_ratio | 0 | 0 | N/A | N/A | N/A | 0 | 0 | 0 |
| iat_min | 76.799 | 0 | 44.01 | 48.351 | 2378.523 | 0.001 | 0 | 0 |

## Outlier Files

| rel_path | outlier_score | top drivers |
|---|---:|---|
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_tencentBlock/10K/tencentBlock_3330.oracleGeneral.zst | 1356.781 | ts_duration (z=272.178); iat_mean (z=271.493) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_tencentBlock/1K/tencentBlock_25497.oracleGeneral.zst | 609.951 | iat_mean (z=314756.6); iat_std (z=142360) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_tencentBlock/1M/tencentBlock_11839.oracleGeneral.zst | 529.65 | abs_stride_q90 (z=176.496); abs_stride_std (z=166.49) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_tencentBlock/100K/tencentBlock_12500.oracleGeneral.zst | 251.79 | abs_stride_mean (z=132.253); abs_stride_std (z=106.139) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_tencentBlock/1K/tencentBlock_26060.oracleGeneral.zst | 232.744 | iat_mean (z=207343); iat_std (z=101576) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_tencentBlock/1K/tencentBlock_25489.oracleGeneral.zst | 228.65 | iat_mean (z=195782.8); iat_std (z=101101) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_tencentBlock/1K/tencentBlock_25302.oracleGeneral.zst | 167.319 | iat_mean (z=110150); iat_std (z=86988.93) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_tencentBlock/1K/tencentBlock_25518.oracleGeneral.zst | 166.842 | iat_mean (z=95092.79); iat_std (z=82067.38) |

## Outlier Sensitivity

| N Removed | Metric | Baseline Median | Trimmed Median | Relative Shift |
|---:|---|---:|---:|---:|
| 10 | burstiness_cv | 2.958 | 2.961 | 0.001 |
| 5 | burstiness_cv | 2.958 | 2.961 | 0.001 |
| 10 | abs_stride_mean | 4166683813 | 4164124069 | -0.001 |
| 5 | object_unique | 1788 | 1789 | 0.001 |
| 10 | object_unique | 1788 | 1789 | 0.001 |
| 3 | abs_stride_mean | 4166683813 | 4165146346 | 0 |
| 5 | abs_stride_mean | 4166683813 | 4165146346 | 0 |
| 3 | burstiness_cv | 2.958 | 2.958 | 0 |
| 1 | object_unique | 1788 | 1788.5 | 0 |
| 3 | object_unique | 1788 | 1788.5 | 0 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv | ts_duration |
|---|---|---:|---:|---:|---:|
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_tencentBlock/1M/tencentBlock_25959.oracleGeneral.zst | oracle_general | 0 | 0.171 | 63.963 | 164361 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_tencentBlock/1M/tencentBlock_25878.oracleGeneral.zst | oracle_general | 0 | 0.063 | 63.177 | 90432 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_tencentBlock/100K/tencentBlock_25820.oracleGeneral.zst | oracle_general | 0 | 0.174 | 56.519 | 17287 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_tencentBlock/100K/tencentBlock_25815.oracleGeneral.zst | oracle_general | 0 | 0.15 | 54.844 | 17506 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_tencentBlock/1M/tencentBlock_25788.oracleGeneral.zst | oracle_general | 0 | 0.041 | 42.804 | 68 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_tencentBlock/1M/tencentBlock_25621.oracleGeneral.zst | oracle_general | 0 | 0.021 | 42.146 | 61 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_tencentBlock/1M/tencentBlock_25467.oracleGeneral.zst | oracle_general | 0 | 0.083 | 41.654 | 172841 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_tencentBlock/1M/tencentBlock_25585.oracleGeneral.zst | oracle_general | 0 | 0.041 | 40.796 | 60 |




## Model-Aware Guidance

- Closest learned anchor: alibaba (distance 0.553)
- Sampling: split-by-format-first
- Regime recipe: K≈8
- Char-file conditioning: yes
- PCF: promising
- Multi-scale critic: promising
- Mixed-type recovery: mixed
- Retrieval memory: unknown
- Why: ordered files show temporal persistence; family looks multi-regime or high-heterogeneity; formats/parsers are mixed
- Candidate conditioning additions: object_unique,signed_stride_lag1_autocorr,obj_size_std
