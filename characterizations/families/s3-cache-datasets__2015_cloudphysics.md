# s3-cache-datasets / 2015_cloudphysics

- Files: 106
- Bytes: 9026863149
- Formats: oracle_general
- Parsers: oracle_general
- ML Use Cases: request_sequence
- Heterogeneity Score: 2.14
- Suggested GAN Modes: 8
- Split By Format: no

## Observations

- Predominantly read-heavy.
- Very weak short-window reuse.
- Ordered feature trajectories show regime boundaries.

## GAN Guidance

- Ordered PC1 changepoints suggest 9 regimes when files are ordered by trace start time.
- Opcode balance is extremely read-skewed; generation should not assume symmetric read/write behavior.
- Strongest feature coupling in this pass: ts_duration vs iat_mean (corr=1).
- A small set of files are strong multivariate outliers; consider holding them out for ablation or separate mode inspection.

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| oracle_general | 106 | oracle_general |

## Clustering And Regimes

| Item | Value |
|---|---|
| DBSCAN clusters | 1 |
| DBSCAN noise fraction | 0.066 |
| Ordered PC1 changepoints | 8 |
| PCA variance explained by PC1 | 0.199 |

## Strongest Correlations

| Metric A | Metric B | Correlation |
|---|---|---:|
| ts_duration | iat_mean | 1 |
| abs_stride_std | abs_stride_q99 | 1 |
| abs_stride_std | abs_stride_q90 | 1 |
| abs_stride_q90 | abs_stride_q99 | 0.999 |
| abs_stride_mean | abs_stride_q90 | 0.999 |
| abs_stride_mean | abs_stride_std | 0.998 |
| abs_stride_mean | abs_stride_q99 | 0.997 |
| forward_seek_ratio | backward_seek_ratio | -0.992 |

## Metrics

| Metric | Mean | Median | CV | Skew | Kurtosis | Missing | Q10 | Q90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| abs_stride_q90 | 612544899 | 47495338 | 7.491 | 10.219 | 104.922 | 0 | 7067617 | 535814232 |
| abs_stride_mean | 194368686 | 12295186 | 7.203 | 10.144 | 103.835 | 0 | 2499717 | 165429073 |
| abs_stride_q50 | 61321000 | 119548 | 7.089 | 9.24 | 89.294 | 0 | 31.5 | 16138880 |
| abs_stride_std | 263005312 | 25883481 | 7.058 | 10.21 | 104.794 | 0 | 6328872 | 324325917 |
| abs_stride_q99 | 937659043 | 95987786 | 6.867 | 10.185 | 104.45 | 0 | 19053323 | 959785868 |
| obj_size_q50 | 25353.66 | 4096 | 4.276 | 7.397 | 58.985 | 0 | 4096 | 32768 |
| iat_std | 2.444 | 0.497 | 3.316 | 6.331 | 44.776 | 0 | 0.088 | 3.69 |
| obj_size_min | 2232.745 | 512 | 2.94 | 8.736 | 84.128 | 0 | 512 | 4096 |
| reuse_ratio | 0.007 | 0.001 | 2.823 | 5.131 | 31.473 | 0 | 0 | 0.013 |
| obj_size_mean | 31757.09 | 11099.31 | 2.416 | 5.822 | 36.499 | 0 | 5168.875 | 58754.88 |
| sample_record_rate | 58.348 | 9.153 | 2.403 | 4.491 | 24.635 | 0 | 1.273 | 157.31 |
| ts_duration | 1241.83 | 448 | 2.359 | 6.831 | 57.049 | 0 | 26.5 | 3217.5 |
| iat_mean | 0.303 | 0.109 | 2.359 | 6.831 | 57.049 | 0 | 0.006 | 0.786 |
| iat_lag1_autocorr | 0.06 | 0.013 | 2.33 | 3.979 | 21.628 | 0 | -0.031 | 0.156 |
| iat_q90 | 0.453 | 0 | 2.242 | 2.94 | 9.034 | 0 | 0 | 1 |
| iat_q99 | 5.006 | 2 | 2.208 | 4.505 | 22.276 | 0 | 0 | 8 |
| obj_size_q90 | 61454.49 | 16384 | 2.072 | 5.286 | 35.654 | 0 | 5632 | 137216 |
| obj_size_std | 41204.26 | 19497.34 | 1.506 | 4.686 | 29.613 | 0 | 7476.934 | 103943.2 |

## Outlier Files

| rel_path | outlier_score |
|---|---:|
| s3-cache-datasets/cache_dataset_oracleGeneral/2015_cloudphysics/w54.oracleGeneral.bin.zst | 100.513 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2015_cloudphysics/w11.oracleGeneral.bin.zst | 56.144 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2015_cloudphysics/w86.oracleGeneral.bin.zst | 42.207 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2015_cloudphysics/w08.oracleGeneral.bin.zst | 14.829 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2015_cloudphysics/w46.oracleGeneral.bin.zst | 6.352 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2015_cloudphysics/w92.oracleGeneral.bin.zst | 6.035 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2015_cloudphysics/w28.oracleGeneral.bin.zst | 5.685 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2015_cloudphysics/w21.oracleGeneral.bin.zst | 4.973 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv | ts_duration |
|---|---|---:|---:|---:|---:|
| s3-cache-datasets/cache_dataset_oracleGeneral/2015_cloudphysics/w30.oracleGeneral.bin.zst | oracle_general | 0 | 0 | 63.984 | 17 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2015_cloudphysics/w46.oracleGeneral.bin.zst | oracle_general | 0 | 0.06 | 43.144 | 3171 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2015_cloudphysics/w15.oracleGeneral.bin.zst | oracle_general | 0 | 0 | 34.096 | 4247 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2015_cloudphysics/w21.oracleGeneral.bin.zst | oracle_general | 0 | 0 | 31.98 | 4 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2015_cloudphysics/w19.oracleGeneral.bin.zst | oracle_general | 0 | 0 | 26.537 | 573 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2015_cloudphysics/w05.oracleGeneral.bin.zst | oracle_general | 0 | 0 | 26.106 | 6 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2015_cloudphysics/w16.oracleGeneral.bin.zst | oracle_general | 0 | 0 | 20.211 | 10 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2015_cloudphysics/w87.oracleGeneral.bin.zst | oracle_general | 0 | 0.002 | 19.268 | 11 |
