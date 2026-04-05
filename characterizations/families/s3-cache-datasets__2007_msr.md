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

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| oracle_general | 14 | oracle_general |

## Clustering And Regimes

| Item | Value |
|---|---|
| DBSCAN clusters | 1 |
| DBSCAN noise fraction | 0 |
| Ordered PC1 changepoints | 2 |
| PCA variance explained by PC1 | 0.222 |

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

| rel_path | outlier_score |
|---|---:|
| s3-cache-datasets/cache_dataset_oracleGeneral/2007_msr/msr_proj_0.oracleGeneral.zst | 8.585 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2007_msr/msr_prxy_1.oracleGeneral.zst | 6.021 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2007_msr/msr_proj_4.oracleGeneral.zst | 3.263 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2007_msr/msr_usr_1.oracleGeneral.zst | 3.129 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2007_msr/msr_proj_1.oracleGeneral.zst | 3.091 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2007_msr/msr_web_2.oracleGeneral.zst | 2.873 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2007_msr/msr_src1_0.oracleGeneral.zst | 2.632 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2007_msr/msr_prn_0.oracleGeneral.zst | 2.299 |

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
