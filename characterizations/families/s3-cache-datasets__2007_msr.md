# s3-cache-datasets / 2007_msr

- Files: 14
- Bytes: 1637389703
- Formats: oracle_general
- Parsers: oracle_general
- ML Use Cases: request_sequence
- Heterogeneity Score: 1.047
- Suggested GAN Modes: 1
- Split By Format: no

## Observations

- Predominantly read-heavy.
- Very weak short-window reuse.

## GAN Guidance

- Opcode balance is extremely read-skewed; generation should not assume symmetric read/write behavior.
- Strongest feature coupling in this pass: ts_duration vs iat_mean (corr=1).
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
| oracle_general | 14 | oracle_general |

## Clustering And Regimes

| Item | Value |
|---|---|
| K-means selected K | 6 |
| Best silhouette K | 6 |
| PCA variance explained by PC1 | 0.222 |

### K Selection

| K | Within-SS | Silhouette |
|---:|---:|---:|
| 2 | 312.94 | 0.134 |
| 3 | 241.934 | 0.176 |
| 4 | 189.872 | 0.185 |
| 5 | 147.525 | 0.195 |
| 6 | 110.618 | 0.211 |
| 7 | 87.403 | 0.165 |
| 8 | 66.171 | 0.163 |
| 9 | 46.816 | 0.169 |
| 10 | 29.193 | 0.144 |
| 11 | 20.896 | 0.089 |
| 12 | 12.006 | 0.071 |

## Strongest Correlations

| Metric A | Metric B | Correlation |
|---|---|---:|
| ts_duration | iat_mean | 1 |
| abs_stride_std | abs_stride_q90 | 0.983 |
| object_top1_share | object_top10_share | 0.981 |
| abs_stride_mean | abs_stride_std | 0.976 |
| abs_stride_mean | abs_stride_q90 | 0.974 |
| sample_record_rate | tenant_unique | -0.962 |
| abs_stride_std | abs_stride_q99 | 0.957 |
| abs_stride_mean | abs_stride_q99 | 0.94 |

## Metrics

| Metric | Mean | Median | CV | Skew | Kurtosis | Missing | Q10 | Q90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| abs_stride_q50 | 3471272046 | 2449408 | 2.625 | N/A | N/A | 0 | 20377.6 | 12562913690 |
| sample_record_rate | 49.759 | 6.563 | 2.438 | N/A | N/A | 0 | 0.517 | 100.571 |
| obj_size_q50 | 8704 | 4096 | 1.898 | N/A | N/A | 0 | 1945.6 | 9267.2 |
| iat_std | 16.291 | 0.457 | 1.873 | N/A | N/A | 0 | 0.133 | 63.372 |
| iat_q90 | 0.357 | 0 | 1.773 | N/A | N/A | 0 | 0 | 1 |
| iat_lag1_autocorr | 0.068 | -0.001 | 1.662 | N/A | N/A | 0 | -0.009 | 0.243 |
| size_bytes | 116956407 | 50780346 | 1.536 | N/A | N/A | 0 | 18547146 | 208477173 |
| reuse_ratio | 0.024 | 0.004 | 1.515 | N/A | N/A | 0 | 0 | 0.071 |
| ts_duration | 2457.857 | 645 | 1.336 | N/A | N/A | 0 | 56 | 7915.9 |
| iat_mean | 0.6 | 0.158 | 1.336 | N/A | N/A | 0 | 0.014 | 1.933 |
| obj_size_min | 768 | 512 | 1.247 | N/A | N/A | 0 | 512 | 512 |
| abs_stride_mean | 27348729319 | 10734964167 | 1.186 | N/A | N/A | 0 | 1410542504 | 74684646661 |
| abs_stride_q90 | 96934738447 | 73684910899 | 1.142 | N/A | N/A | 0 | 1746132828 | 225030954680 |
| object_top1_share | 0.022 | 0.018 | 1.101 | N/A | N/A | 0 | 0.003 | 0.036 |
| abs_stride_q99 | 207679490267 | 154819312927 | 1.052 | N/A | N/A | 0 | 14143649626 | 514224999997 |
| object_top10_share | 0.089 | 0.063 | 1.05 | N/A | N/A | 0 | 0.019 | 0.124 |
| abs_stride_std | 49864332491 | 32792250352 | 1.044 | N/A | N/A | 0 | 3409250338 | 118723696472 |
| burstiness_cv | 13.605 | 6.699 | 1.028 | N/A | N/A | 0 | 2.838 | 33.429 |

## Outlier Files

| rel_path | outlier_score | top drivers |
|---|---:|---|
| s3-cache-datasets/cache_dataset_oracleGeneral/2007_msr/msr_proj_0.oracleGeneral.zst | 8.585 | abs_stride_q50 (z=31.878); iat_lag1_autocorr (z=17.217) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2007_msr/msr_prxy_1.oracleGeneral.zst | 6.021 | sample_record_rate (z=74.136); size_bytes (z=20.089) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2007_msr/msr_proj_4.oracleGeneral.zst | 3.263 | iat_std (z=100); iat_mean (z=12.74) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2007_msr/msr_usr_1.oracleGeneral.zst | 3.129 | abs_stride_q50 (z=49.43); iat_lag1_autocorr (z=17.257) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2007_msr/msr_proj_1.oracleGeneral.zst | 3.091 | abs_stride_q50 (z=100); abs_stride_mean (z=7.669) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2007_msr/msr_web_2.oracleGeneral.zst | 2.873 | iat_std (z=88.266); iat_mean (z=12.571) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2007_msr/msr_src1_0.oracleGeneral.zst | 2.632 | iat_std (z=100); ts_duration (z=12.916) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2007_msr/msr_prn_0.oracleGeneral.zst | 2.299 | reuse_ratio (z=26.571); iat_lag1_autocorr (z=11.693) |

## Outlier Sensitivity

| N Removed | Metric | Baseline Median | Trimmed Median | Relative Shift |
|---:|---|---:|---:|---:|
| 10 | abs_stride_mean | 10734964167 | 32671762307 | 2.043 |
| 3 | abs_stride_mean | 10734964167 | 32203910067 | 2 |
| 10 | reuse_ratio | 0.004 | 0.006 | 0.389 |
| 10 | obj_size_std | 17324.47 | 21393.08 | 0.235 |
| 5 | reuse_ratio | 0.004 | 0.005 | 0.222 |
| 1 | reuse_ratio | 0.004 | 0.003 | -0.222 |
| 3 | reuse_ratio | 0.004 | 0.003 | -0.222 |
| 10 | burstiness_cv | 6.699 | 5.877 | -0.123 |
| 1 | burstiness_cv | 6.699 | 7.338 | 0.095 |
| 3 | burstiness_cv | 6.699 | 6.06 | -0.095 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv | ts_duration |
|---|---|---:|---:|---:|---:|
| s3-cache-datasets/cache_dataset_oracleGeneral/2007_msr/msr_proj_4.oracleGeneral.zst | oracle_general | 0 | 0.001 | 47.245 | 7945 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2007_msr/msr_src1_0.oracleGeneral.zst | oracle_general | 0 | 0 | 38.625 | 8046 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2007_msr/msr_prxy_1.oracleGeneral.zst | oracle_general | 0 | 0.075 | 21.307 | 9 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2007_msr/msr_web_2.oracleGeneral.zst | oracle_general | 0 | 0 | 17.825 | 7848 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2007_msr/msr_proj_2.oracleGeneral.zst | oracle_general | 0 | 0 | 16.87 | 5359 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2007_msr/msr_prn_1.oracleGeneral.zst | oracle_general | 0 | 0.029 | 13.227 | 32 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2007_msr/msr_prn_0.oracleGeneral.zst | oracle_general | 0 | 0.118 | 7.338 | 112 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2007_msr/msr_prxy_0.oracleGeneral.zst | oracle_general | 0 | 0.005 | 6.06 | 168 |
