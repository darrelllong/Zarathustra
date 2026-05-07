# s3-cache-datasets / 2015_cloudphysics

- Files: 106
- Bytes: 9026863149
- Formats: oracle_general
- Parsers: oracle_general
- ML Use Cases: request_sequence
- Heterogeneity Score: 2.14
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
| Near-constant current conditioning features | write_ratio, iat_q50, opcode_switch_ratio, tenant_unique |
| Recommended candidate additions | object_unique, signed_stride_lag1_autocorr, obj_size_std |
| Highly redundant current pairs | forward_seek_ratio vs backward_seek_ratio (-0.992) |

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| oracle_general | 106 | oracle_general |

## Clustering And Regimes

| Item | Value |
|---|---|
| K-means selected K | 2 |
| Best silhouette K | 2 |
| PCA variance explained by PC1 | 0.199 |
| Block/random distance ratio | 1.02 |
| Sampling recommendation | random_sampling_is_less_problematic |

### K Selection

| K | Within-SS | Silhouette |
|---:|---:|---:|
| 2 | 2579.181 | 0.744 |
| 3 | 2215.786 | 0.16 |
| 4 | 1907.057 | 0.223 |
| 5 | 1682.286 | 0.148 |
| 6 | 1470.369 | 0.177 |
| 7 | 1303.246 | 0.149 |
| 8 | 1181.003 | 0.127 |
| 9 | 1077.978 | 0.146 |
| 10 | 1005.802 | 0.165 |
| 11 | 946.776 | 0.174 |
| 12 | 886.211 | 0.147 |

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
| abs_stride_q90 | 612544899 | 47495338 | 7.491 | N/A | N/A | 0 | 7067617 | 535814232 |
| abs_stride_mean | 194368686 | 12295186 | 7.203 | N/A | N/A | 0 | 2499717 | 165429073 |
| abs_stride_q50 | 61321000 | 119548 | 7.089 | N/A | N/A | 0 | 31.5 | 16138880 |
| abs_stride_std | 263005312 | 25883481 | 7.058 | N/A | N/A | 0 | 6328872 | 324325917 |
| abs_stride_q99 | 937659043 | 95987786 | 6.867 | N/A | N/A | 0 | 19053323 | 959785868 |
| obj_size_q50 | 25353.66 | 4096 | 4.276 | N/A | N/A | 0 | 4096 | 32768 |
| iat_std | 2.444 | 0.497 | 3.316 | N/A | N/A | 0 | 0.088 | 3.69 |
| obj_size_min | 2232.745 | 512 | 2.94 | N/A | N/A | 0 | 512 | 4096 |
| reuse_ratio | 0.007 | 0.001 | 2.823 | N/A | N/A | 0 | 0 | 0.013 |
| obj_size_mean | 31757.09 | 11099.31 | 2.416 | N/A | N/A | 0 | 5168.875 | 58754.88 |
| sample_record_rate | 58.348 | 9.153 | 2.403 | N/A | N/A | 0 | 1.273 | 157.31 |
| ts_duration | 1241.83 | 448 | 2.359 | N/A | N/A | 0 | 26.5 | 3217.5 |
| iat_mean | 0.303 | 0.109 | 2.359 | N/A | N/A | 0 | 0.006 | 0.786 |
| iat_lag1_autocorr | 0.06 | 0.013 | 2.33 | N/A | N/A | 0 | -0.031 | 0.156 |
| iat_q90 | 0.453 | 0 | 2.242 | N/A | N/A | 0 | 0 | 1 |
| iat_q99 | 5.006 | 2 | 2.208 | N/A | N/A | 0 | 0 | 8 |
| obj_size_q90 | 61454.49 | 16384 | 2.072 | N/A | N/A | 0 | 5632 | 137216 |
| obj_size_std | 41204.26 | 19497.34 | 1.506 | N/A | N/A | 0 | 7476.934 | 103943.2 |

## Outlier Files

| rel_path | outlier_score | top drivers |
|---|---:|---|
| s3-cache-datasets/cache_dataset_oracleGeneral/2015_cloudphysics/w54.oracleGeneral.bin.zst | 100.513 | abs_stride_mean (z=100); abs_stride_std (z=100) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2015_cloudphysics/w11.oracleGeneral.bin.zst | 56.144 | obj_size_mean (z=100); obj_size_q90 (z=84) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2015_cloudphysics/w86.oracleGeneral.bin.zst | 42.207 | iat_std (z=100); iat_q99 (z=68.18) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2015_cloudphysics/w08.oracleGeneral.bin.zst | 14.829 | obj_size_mean (z=92.757); obj_size_q90 (z=41.333) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2015_cloudphysics/w46.oracleGeneral.bin.zst | 6.352 | iat_std (z=84.724); reuse_ratio (z=80.667) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2015_cloudphysics/w92.oracleGeneral.bin.zst | 6.035 | ts_duration (z=7.928); iat_mean (z=7.928) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2015_cloudphysics/w28.oracleGeneral.bin.zst | 5.685 | iat_q99 (z=24); iat_mean (z=13.826) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2015_cloudphysics/w21.oracleGeneral.bin.zst | 4.973 | sample_record_rate (z=100); burstiness_cv (z=12.306) |

## Outlier Sensitivity

| N Removed | Metric | Baseline Median | Trimmed Median | Relative Shift |
|---:|---|---:|---:|---:|
| 10 | abs_stride_mean | 12295186 | 12901058 | 0.049 |
| 1 | abs_stride_mean | 12295186 | 11990499 | -0.025 |
| 3 | abs_stride_mean | 12295186 | 11990499 | -0.025 |
| 5 | abs_stride_mean | 12295186 | 11990499 | -0.025 |
| 5 | obj_size_std | 19497.34 | 19047.68 | -0.023 |
| 1 | burstiness_cv | 5.393 | 5.416 | 0.004 |
| 3 | burstiness_cv | 5.393 | 5.416 | 0.004 |
| 5 | burstiness_cv | 5.393 | 5.369 | -0.004 |
| 1 | object_unique | 2784.5 | 2774 | -0.004 |
| 3 | object_unique | 2784.5 | 2795 | 0.004 |

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
