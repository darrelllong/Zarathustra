# s3-cache-datasets / 2007_msr

- Files: 14
- Bytes: 1637389703
- Formats: oracle_general
- Parsers: oracle_general
- ML Use Cases: request_sequence
- Heterogeneity Score: 1.047
- Suggested GAN Modes: 3
- Split By Format: no

## Observations

- Predominantly read-heavy.
- Very weak short-window reuse.
- Ordered feature trajectories show regime boundaries.

## GAN Guidance

- Ordered PC1 changepoints suggest 3 regimes when files are ordered by trace start time.
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
| K-means selected K | 2 |
| Best silhouette K | 2 |
| DBSCAN clusters | 1 |
| DBSCAN noise fraction | 0 |
| Ordered PC1 changepoints | 2 |
| PCA variance explained by PC1 | 0.222 |
| Hurst exponent on ordered PC1 | 0.5 |

### K Selection

| K | Within-SS | Silhouette |
|---:|---:|---:|
| 2 | 231644117329442167062528 | 0.693 |
| 3 | 56263123749828374298624 | 0.647 |
| 4 | 37631854305759727714304 | 0.55 |
| 5 | 30871683420196362518528 | 0.496 |
| 6 | 17615442438336840466432 | 0.414 |
| 7 | 10787884173284381556736 | 0.339 |
| 8 | 10352201047852961497088 | 0.298 |
| 9 | 5041157060940396494848 | 0.279 |
| 10 | 2376255311507161612288 | 0.244 |
| 11 | 275050467111513915392 | 0.211 |
| 12 | 91237103860140441600 | 0.138 |

## Regime Transition Drivers

| Transition | Driver 1 | Effect | Driver 2 | Effect | Driver 3 | Effect |
|---|---|---:|---|---:|---|---:|
| 1 -> 2 | burstiness_cv | 3.503 | tenant_top1_share | 2.68 | size_bytes | 2.041 |
| 2 -> 3 | burstiness_cv | 4.265 | iat_zero_ratio | 3.223 | object_top10_share | 2.645 |

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
| abs_stride_q50 | 3471272046 | 2449408 | 2.625 | 2.63 | 6.378 | 0 | 20377.6 | 12562913690 |
| sample_record_rate | 49.759 | 6.563 | 2.438 | 3.324 | 11.453 | 0 | 0.517 | 100.571 |
| obj_size_q50 | 8704 | 4096 | 1.898 | 3.616 | 13.314 | 0 | 1945.6 | 9267.2 |
| iat_std | 16.291 | 0.457 | 1.873 | 1.914 | 2.601 | 0 | 0.133 | 63.372 |
| iat_q90 | 0.357 | 0 | 1.773 | 1.687 | 2.214 | 0 | 0 | 1 |
| iat_lag1_autocorr | 0.068 | -0.001 | 1.662 | 0.909 | -0.687 | 0 | -0.009 | 0.243 |
| size_bytes | 116956407 | 50780346 | 1.536 | 2.96 | 9.591 | 0 | 18547146 | 208477173 |
| reuse_ratio | 0.024 | 0.004 | 1.515 | 1.644 | 2.106 | 0 | 0 | 0.071 |
| ts_duration | 2457.857 | 645 | 1.336 | 1.089 | -0.686 | 0 | 56 | 7915.9 |
| iat_mean | 0.6 | 0.158 | 1.336 | 1.089 | -0.686 | 0 | 0.014 | 1.933 |
| obj_size_min | 768 | 512 | 1.247 | 3.742 | 14 | 0 | 512 | 512 |
| abs_stride_mean | 27348729319 | 10734964167 | 1.186 | 1.187 | 0.363 | 0 | 1410542504 | 74684646661 |
| abs_stride_q90 | 96934738447 | 73684910899 | 1.142 | 1.28 | 1.377 | 0 | 1746132828 | 225030954680 |
| object_top1_share | 0.022 | 0.018 | 1.101 | 2.49 | 7.496 | 0 | 0.003 | 0.036 |
| abs_stride_q99 | 207679490267 | 154819312927 | 1.052 | 1.341 | 1.212 | 0 | 14143649626 | 514224999997 |
| object_top10_share | 0.089 | 0.063 | 1.05 | 2.689 | 8.544 | 0 | 0.019 | 0.124 |
| abs_stride_std | 49864332491 | 32792250352 | 1.044 | 1.218 | 0.883 | 0 | 3409250338 | 118723696472 |
| burstiness_cv | 13.605 | 6.699 | 1.028 | 1.559 | 1.739 | 0 | 2.838 | 33.429 |

## Outlier Files

| rel_path | outlier_score | top drivers |
|---|---:|---|
| s3-cache-datasets/cache_dataset_oracleGeneral/2007_msr/msr_proj_0.oracleGeneral.zst | 8.585 | abs_stride_q50 (z=31.878); iat_lag1_autocorr (z=17.217) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2007_msr/msr_prxy_1.oracleGeneral.zst | 6.021 | sample_record_rate (z=74.136); size_bytes (z=20.089) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2007_msr/msr_proj_4.oracleGeneral.zst | 3.263 | iat_std (z=238.856); iat_mean (z=12.74) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2007_msr/msr_usr_1.oracleGeneral.zst | 3.129 | abs_stride_q50 (z=49.43); iat_lag1_autocorr (z=17.257) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2007_msr/msr_proj_1.oracleGeneral.zst | 3.091 | abs_stride_q50 (z=7306.769); abs_stride_mean (z=7.669) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2007_msr/msr_web_2.oracleGeneral.zst | 2.873 | iat_std (z=88.266); iat_mean (z=12.571) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2007_msr/msr_src1_0.oracleGeneral.zst | 2.632 | iat_std (z=197.55); ts_duration (z=12.916) |
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




## Model-Aware Guidance

- Closest learned anchor: tencent_block (distance 1.102)
- Sampling: random-ok
- Regime recipe: single
- Char-file conditioning: yes
- PCF: validated
- Multi-scale critic: promising
- Mixed-type recovery: promising
- Retrieval memory: mixed
- Why: burstiness is materially above the calmer families
- Candidate conditioning additions: object_unique,signed_stride_lag1_autocorr,obj_size_std
