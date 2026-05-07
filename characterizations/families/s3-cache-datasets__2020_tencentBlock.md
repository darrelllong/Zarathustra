# s3-cache-datasets / 2020_tencentBlock

- Files: 4996
- Bytes: 300469429369
- Formats: oracle_general, text_zst
- Parsers: generic_text, oracle_general
- ML Use Cases: request_sequence, structured_table
- Heterogeneity Score: 2.006
- Suggested GAN Modes: 1
- Split By Format: yes

## Observations

- Predominantly read-heavy.
- Very weak short-window reuse.

## GAN Guidance

- Family spans multiple encodings; keep format-aware preprocessing and avoid blindly pooling structured-table and request-sequence variants.
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
| PCA variance explained by PC1 | 0.213 |
| Block/random distance ratio | 0.604 |
| Sampling recommendation | block_sampling_preserves_temporal_coherence |

### K Selection

| K | Within-SS | Silhouette |
|---:|---:|---:|
| 2 | 117052.7 | 0.685 |
| 3 | 97483.51 | 0.574 |
| 4 | 87717.17 | 0.232 |
| 5 | 80430.71 | 0.234 |
| 6 | 75946.44 | 0.238 |
| 7 | 72086.57 | 0.242 |
| 8 | 68608.5 | 0.251 |
| 9 | 64018.5 | 0.19 |
| 10 | 61641.42 | 0.168 |
| 11 | 58811.48 | 0.174 |
| 12 | 56599.71 | 0.184 |

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
| size_bytes | 60141999 | 10893978 | 32.335 | N/A | N/A | 0 | 1254170 | 44922228 |
| sample_records | 4032.378 | 4096 | 0.122 | N/A | N/A | 0 | 4096 | 4096 |
| ttl_present | 0 | 0 | N/A | N/A | N/A | 0 | 0 | 0 |
| ts_duration | 12915.69 | 1448 | 6.134 | N/A | N/A | 0 | 194.4 | 5645.2 |
| obj_size_q50 | 13172.49 | 4096 | 4.267 | N/A | N/A | 0 | 4096 | 8192 |
| obj_size_min | 2650.612 | 4096 | 3.063 | N/A | N/A | 0 | 512 | 4096 |
| obj_size_mean | 20632.45 | 8785.875 | 2.29 | N/A | N/A | 0 | 5852.65 | 28838.88 |
| obj_size_q90 | 46053.73 | 16384 | 2.245 | N/A | N/A | 0 | 8192 | 65536 |
| object_top1_share | 0.039 | 0.019 | 1.768 | N/A | N/A | 0 | 0.006 | 0.08 |
| obj_size_std | 28368.46 | 13375 | 1.544 | N/A | N/A | 0 | 6567.901 | 63913.97 |
| obj_size_q99 | 116023.6 | 65536 | 1.24 | N/A | N/A | 0 | 28672 | 427540.5 |
| object_top10_share | 0.195 | 0.131 | 0.925 | N/A | N/A | 0 | 0.042 | 0.387 |
| object_unique | 1785.607 | 1788 | 0.473 | N/A | N/A | 0 | 727 | 2916.6 |
| tenant_top1_share | 0.927 | 0.954 | 0.097 | N/A | N/A | 0 | 0.831 | 0.997 |
| tenant_unique | 1.984 | 2 | 0.062 | N/A | N/A | 0 | 2 | 2 |
| tenant_top10_share | 1 | 1 | 0 | N/A | N/A | 0 | 1 | 1 |
| write_ratio | 0 | 0 | N/A | N/A | N/A | 0 | 0 | 0 |
| iat_min | 76.799 | 0 | 44.01 | N/A | N/A | 0.001 | 0 | 0 |

## Outlier Files

| rel_path | outlier_score | top drivers |
|---|---:|---|
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_tencentBlock/10K/tencentBlock_3330.oracleGeneral.zst | 1356.781 | ts_duration (z=100); iat_mean (z=100) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_tencentBlock/1K/tencentBlock_25497.oracleGeneral.zst | 609.951 | ts_duration (z=100); iat_mean (z=100) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_tencentBlock/1M/tencentBlock_11839.oracleGeneral.zst | 529.65 | abs_stride_mean (z=100); abs_stride_std (z=100) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_tencentBlock/100K/tencentBlock_12500.oracleGeneral.zst | 251.79 | abs_stride_mean (z=100); abs_stride_std (z=100) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_tencentBlock/1K/tencentBlock_26060.oracleGeneral.zst | 232.744 | ts_duration (z=100); iat_mean (z=100) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_tencentBlock/1K/tencentBlock_25489.oracleGeneral.zst | 228.65 | ts_duration (z=100); iat_mean (z=100) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_tencentBlock/1K/tencentBlock_25302.oracleGeneral.zst | 167.319 | ts_duration (z=100); iat_mean (z=100) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_tencentBlock/1K/tencentBlock_25518.oracleGeneral.zst | 166.842 | ts_duration (z=100); iat_mean (z=100) |

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
