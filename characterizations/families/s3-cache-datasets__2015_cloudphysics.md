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
| DBSCAN clusters | 1 |
| DBSCAN noise fraction | 0.066 |
| Ordered PC1 changepoints | 8 |
| PCA variance explained by PC1 | 0.199 |
| Hurst exponent on ordered PC1 | 0.701 |
| Block/random distance ratio | 1.092 |
| Sampling recommendation | random_sampling_is_less_problematic |

### K Selection

| K | Within-SS | Silhouette |
|---:|---:|---:|
| 2 | 49052887092172562432 | 0.984 |
| 3 | 15368729563236937728 | 0.797 |
| 4 | 12912626014622765056 | 0.58 |
| 5 | 8826949501259444224 | 0.584 |
| 6 | 8379242046393213952 | 0.584 |
| 7 | 7274001190067757056 | 0.569 |
| 8 | 3906510552262976000 | 0.606 |
| 9 | 5354161389259911168 | 0.328 |
| 10 | 5248206810790833152 | 0.337 |
| 11 | 5008914523528914944 | 0.292 |
| 12 | 4844564904266328064 | 0.295 |

## Regime Transition Drivers

| Transition | Driver 1 | Effect | Driver 2 | Effect | Driver 3 | Effect |
|---|---|---:|---|---:|---|---:|
| 1 -> 2 | iat_std | 2.371 | iat_lag1_autocorr | 2.029 | forward_seek_ratio | 1.76 |
| 2 -> 3 | iat_lag1_autocorr | 5.206 | iat_std | 3.414 | tenant_top1_share | 2.892 |
| 3 -> 4 | tenant_top1_share | 2.439 | obj_size_q50 | 1.414 | iat_q90 | 1.414 |
| 4 -> 5 | tenant_top1_share | 2.534 | abs_stride_q90 | 2.336 | signed_stride_lag1_autocorr | 1.427 |
| 5 -> 6 | obj_size_mean | 2.417 | object_unique | 1.996 | signed_stride_lag1_autocorr | 1.552 |
| 6 -> 7 | backward_seek_ratio | 2.06 | forward_seek_ratio | 2.032 | ts_duration | 1.787 |
| 7 -> 8 | abs_stride_q99 | 2.097 | abs_stride_std | 1.87 | abs_stride_mean | 1.736 |
| 8 -> 9 | obj_size_q90 | 3.773 | abs_stride_q50 | 3.299 | obj_size_mean | 2.31 |

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

| rel_path | outlier_score | top drivers |
|---|---:|---|
| s3-cache-datasets/cache_dataset_oracleGeneral/2015_cloudphysics/w54.oracleGeneral.bin.zst | 100.513 | abs_stride_q50 (z=36043.11); abs_stride_mean (z=1501.477) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2015_cloudphysics/w11.oracleGeneral.bin.zst | 56.144 | obj_size_mean (z=102.776); obj_size_q90 (z=84) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2015_cloudphysics/w86.oracleGeneral.bin.zst | 42.207 | iat_std (z=173.5); iat_q99 (z=68.18) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2015_cloudphysics/w08.oracleGeneral.bin.zst | 14.829 | obj_size_mean (z=92.757); obj_size_q90 (z=41.333) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2015_cloudphysics/w46.oracleGeneral.bin.zst | 6.352 | iat_std (z=84.724); reuse_ratio (z=80.667) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2015_cloudphysics/w92.oracleGeneral.bin.zst | 6.035 | ts_duration (z=7.928); iat_mean (z=7.928) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2015_cloudphysics/w28.oracleGeneral.bin.zst | 5.685 | iat_q99 (z=24); iat_mean (z=13.826) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2015_cloudphysics/w21.oracleGeneral.bin.zst | 4.973 | sample_record_rate (z=128.791); burstiness_cv (z=12.306) |

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
