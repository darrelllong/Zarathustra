# s3-cache-datasets / msr

- Files: 36
- Bytes: 8130400717
- Formats: lcs
- Parsers: lcs
- ML Use Cases: request_sequence
- Heterogeneity Score: 1.057
- Suggested GAN Modes: 1
- Split By Format: no

## Observations

- Substantial write pressure across sampled files.
- Very weak short-window reuse.

## GAN Guidance

- Write pressure is material; preserve write bursts and opcode transitions in conditioning.
- Strongest feature coupling in this pass: sample_records vs iat_q99 (corr=-1).
- A small set of files are strong multivariate outliers; consider holding them out for ablation or separate mode inspection.
- Current characterization suggests extra conditioning value from: object_unique, signed_stride_lag1_autocorr.

## Conditioning Audit

| Item | Value |
|---|---|
| Near-constant current conditioning features | iat_q50, obj_size_q50, tenant_unique |
| Recommended candidate additions | object_unique, signed_stride_lag1_autocorr |
| Highly redundant current pairs | none flagged |

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| lcs | 36 | lcs |

## Clustering And Regimes

| Item | Value |
|---|---|
| K-means selected K | 2 |
| Best silhouette K | 2 |
| PCA variance explained by PC1 | 0.369 |
| Block/random distance ratio | 1.25 |
| Sampling recommendation | random_sampling_is_less_problematic |

### K Selection

| K | Within-SS | Silhouette |
|---:|---:|---:|
| 2 | 596.067 | 0.551 |
| 3 | 475.898 | 0.468 |
| 4 | 388.267 | 0.487 |
| 5 | 305.824 | 0.256 |
| 6 | 242.863 | 0.26 |
| 7 | 203.72 | 0.272 |
| 8 | 175.475 | 0.278 |
| 9 | 153.862 | 0.281 |
| 10 | 136.816 | 0.219 |
| 11 | 120.041 | 0.229 |
| 12 | 102.346 | 0.216 |

## Strongest Correlations

| Metric A | Metric B | Correlation |
|---|---|---:|
| sample_records | iat_q99 | -0.997 |
| sample_records | iat_mean | -0.99 |
| iat_mean | iat_q99 | 0.977 |
| ts_duration | iat_q99 | 0.973 |
| sample_records | ts_duration | -0.97 |
| ts_duration | iat_mean | 0.965 |
| iat_zero_ratio | iat_q90 | -0.963 |
| abs_stride_std | abs_stride_q99 | 0.955 |

## Metrics

| Metric | Mean | Median | CV | Skew | Kurtosis | Missing | Q10 | Q90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| abs_stride_q50 | 549.667 | 1 | 4.914 | N/A | N/A | 0 | 1 | 33 |
| iat_q99 | 794.378 | 5 | 4.164 | N/A | N/A | 0 | 0 | 5 |
| iat_mean | 33.528 | 0.313 | 3.881 | N/A | N/A | 0 | 0.017 | 19.719 |
| sample_record_rate | 65.607 | 3.195 | 3.711 | N/A | N/A | 0 | 0.057 | 61.05 |
| reuse_ratio | 0.027 | 0 | 3.391 | N/A | N/A | 0 | 0 | 0.03 |
| ts_duration | 45186.31 | 1282 | 3.083 | N/A | N/A | 0 | 70.5 | 80750 |
| abs_stride_q90 | 3536514 | 238637.9 | 2.606 | N/A | N/A | 0 | 1839 | 9832648 |
| iat_std | 280.306 | 1.379 | 2.544 | N/A | N/A | 0 | 0.151 | 1041.694 |
| iat_q90 | 0.556 | 0 | 2.525 | N/A | N/A | 0 | 0 | 1 |
| size_bytes | 225844464 | 30522060 | 2.078 | N/A | N/A | 0 | 702843 | 1019020626 |
| abs_stride_mean | 1441400 | 505020.5 | 1.891 | N/A | N/A | 0 | 87402.29 | 2862563 |
| iat_lag1_autocorr | 0.056 | 0 | 1.753 | N/A | N/A | 0 | -0.017 | 0.194 |
| opcode_switch_ratio | 0.01 | 0.003 | 1.719 | N/A | N/A | 0 | 0 | 0.027 |
| abs_stride_q99 | 20647060 | 7815726 | 1.433 | N/A | N/A | 0 | 757757.6 | 62338712 |
| abs_stride_std | 4459085 | 2078768 | 1.411 | N/A | N/A | 0 | 355896.6 | 11380507 |
| object_top10_share | 0.133 | 0.066 | 1.285 | N/A | N/A | 0 | 0.009 | 0.275 |
| object_top1_share | 0.028 | 0.013 | 1.143 | N/A | N/A | 0 | 0.001 | 0.063 |
| burstiness_cv | 17.84 | 8.633 | 0.97 | N/A | N/A | 0 | 3.974 | 47.435 |

## Outlier Files

| rel_path | outlier_score | top drivers |
|---|---:|---|
| s3-cache-datasets/cache_dataset_lcs/msr/rsrch_2.csv.lcs.zst | 18.771 | iat_std (z=100); abs_stride_q90 (z=48.684) |
| s3-cache-datasets/cache_dataset_lcs/msr/wdev_3.csv.lcs.zst | 18.228 | ts_duration (z=100); iat_mean (z=100) |
| s3-cache-datasets/cache_dataset_lcs/msr/wdev_1.csv.lcs.zst | 13.692 | ts_duration (z=100); iat_mean (z=100) |
| s3-cache-datasets/cache_dataset_lcs/msr/mds_1.csv.lcs.zst | 12.148 | abs_stride_q90 (z=100); abs_stride_mean (z=37.202) |
| s3-cache-datasets/cache_dataset_lcs/msr/usr_1.csv.lcs.zst | 11.355 | abs_stride_q90 (z=100); size_bytes (z=49.825) |
| s3-cache-datasets/cache_dataset_lcs/msr/prxy_1.csv.lcs.zst | 4.654 | sample_record_rate (z=100); size_bytes (z=62.331) |
| s3-cache-datasets/cache_dataset_lcs/msr/prn_1.csv.lcs.zst | 1.861 | sample_record_rate (z=100); abs_stride_q90 (z=37.48) |
| s3-cache-datasets/cache_dataset_lcs/msr/web_3.csv.lcs.zst | 1.837 | signed_stride_lag1_autocorr (z=4.569); iat_q99 (z=4.53) |

## Outlier Sensitivity

| N Removed | Metric | Baseline Median | Trimmed Median | Relative Shift |
|---:|---|---:|---:|---:|
| 10 | reuse_ratio | 0 | 0.002 | 3.75 |
| 10 | abs_stride_mean | 505020.5 | 302091.3 | -0.402 |
| 5 | abs_stride_mean | 505020.5 | 311927.8 | -0.382 |
| 3 | abs_stride_mean | 505020.5 | 331482.1 | -0.344 |
| 1 | abs_stride_mean | 505020.5 | 447300.5 | -0.114 |
| 3 | object_unique | 2868 | 2977 | 0.038 |
| 1 | object_unique | 2868 | 2928 | 0.021 |
| 5 | object_unique | 2868 | 2928 | 0.021 |
| 1 | burstiness_cv | 8.634 | 8.571 | -0.007 |
| 3 | burstiness_cv | 8.634 | 8.696 | 0.007 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv | ts_duration |
|---|---|---:|---:|---:|---:|
| s3-cache-datasets/cache_dataset_lcs/msr/src2_2.csv.lcs.zst | lcs | 0.609 | 0 | 63.978 | 107641 |
| s3-cache-datasets/cache_dataset_lcs/msr/rsrch_2.csv.lcs.zst | lcs | 1 | 0 | 51.145 | 32160 |
| s3-cache-datasets/cache_dataset_lcs/msr/src2_1.csv.lcs.zst | lcs | 0.983 | 0 | 47.705 | 161209 |
| s3-cache-datasets/cache_dataset_lcs/msr/hm_1.csv.lcs.zst | lcs | 0.249 | 0 | 47.589 | 19066 |
| s3-cache-datasets/cache_dataset_lcs/msr/proj_4.csv.lcs.zst | lcs | 0.821 | 0 | 47.281 | 7939 |
| s3-cache-datasets/cache_dataset_lcs/msr/src1_0.csv.lcs.zst | lcs | 0.821 | 0 | 38.673 | 8036 |
| s3-cache-datasets/cache_dataset_lcs/msr/mds_1.csv.lcs.zst | lcs | 0.124 | 0 | 37.962 | 1296 |
| s3-cache-datasets/cache_dataset_lcs/msr/prxy_1.csv.lcs.zst | lcs | 0.404 | 0.009 | 36.932 | 3 |
