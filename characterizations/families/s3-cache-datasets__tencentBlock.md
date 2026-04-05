# s3-cache-datasets / tencentBlock

- Files: 4995
- Bytes: 817117209599
- Formats: lcs
- Parsers: lcs
- ML Use Cases: request_sequence
- Heterogeneity Score: 1.328
- Suggested GAN Modes: 8
- Split By Format: no

## Observations

- Substantial write pressure across sampled files.
- Very weak short-window reuse.
- Ordered feature trajectories show regime boundaries.

## GAN Guidance

- Ordered PC1 changepoints suggest 17 regimes when files are ordered by trace start time.
- Write pressure is material; preserve write bursts and opcode transitions in conditioning.
- Strongest feature coupling in this pass: iat_std vs iat_q99 (corr=0.98).
- A small set of files are strong multivariate outliers; consider holding them out for ablation or separate mode inspection.

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| lcs | 4995 | lcs |

## Clustering And Regimes

| Item | Value |
|---|---|
| DBSCAN clusters | 4 |
| DBSCAN noise fraction | 0.072 |
| Ordered PC1 changepoints | 16 |
| PCA variance explained by PC1 | 0.287 |

## Strongest Correlations

| Metric A | Metric B | Correlation |
|---|---|---:|
| iat_std | iat_q99 | 0.979 |
| abs_stride_std | abs_stride_q99 | 0.94 |
| iat_mean | iat_std | 0.916 |
| abs_stride_mean | abs_stride_std | 0.913 |
| iat_std | iat_q90 | 0.905 |
| iat_mean | iat_q90 | 0.905 |
| iat_q90 | iat_q99 | 0.877 |
| iat_zero_ratio | backward_seek_ratio | -0.869 |

## Metrics

| Metric | Mean | Median | CV | Skew | Kurtosis | Missing | Q10 | Q90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| size_bytes | 163587029 | 24975086 | 4.895 | 16.465 | 430.767 | 0 | 2585280 | 220043821 |
| ttl_present | 0 | 0 | N/A | N/A | N/A | 0 | 0 | 0 |
| iat_q50 | 30.39 | 0 | 45.752 | 54.147 | 3157.634 | 0.001 | 0 | 0 |
| abs_stride_q50 | 3990.29 | 1 | 40.939 | 56.55 | 3472.372 | 0.001 | 1 | 15 |
| iat_mean | 143.975 | 0.163 | 12.703 | 19.199 | 471.817 | 0.001 | 0.007 | 0.667 |
| iat_q90 | 498.913 | 0 | 11.983 | 14.189 | 236.673 | 0.001 | 0 | 2 |
| iat_q99 | 849.467 | 4 | 11.559 | 17.364 | 414.837 | 0.001 | 0 | 8 |
| iat_std | 288.894 | 0.67 | 11.11 | 16.754 | 382.13 | 0.001 | 0.103 | 2.332 |
| ts_duration | 9823.927 | 666 | 7.336 | 9.462 | 91.189 | 0.001 | 27 | 2685 |
| abs_stride_q90 | 2447795 | 674503 | 2.904 | 10.502 | 149.243 | 0.001 | 2 | 5727156 |
| reuse_ratio | 0.02 | 0.004 | 2.601 | 6.224 | 51.633 | 0.001 | 0 | 0.042 |
| abs_stride_mean | 724530.1 | 393588.3 | 2.477 | 11.791 | 199.339 | 0.001 | 39846.5 | 1251208 |
| abs_stride_q99 | 8994874 | 6483778 | 2.304 | 12.421 | 230.377 | 0.001 | 412146.1 | 13451440 |
| abs_stride_std | 2088251 | 1359910 | 2.241 | 12.728 | 241.983 | 0.001 | 190037.9 | 3387733 |
| object_top1_share | 0.021 | 0.009 | 2.063 | 7.412 | 80.723 | 0.001 | 0.002 | 0.045 |
| opcode_switch_ratio | 0.012 | 0.002 | 2.002 | 6.086 | 71.861 | 0.001 | 0 | 0.035 |
| object_top10_share | 0.117 | 0.064 | 1.288 | 3.091 | 11.765 | 0.001 | 0.014 | 0.26 |
| backward_seek_ratio | 0.087 | 0.078 | 0.691 | 1.004 | 1.807 | 0.001 | 0.014 | 0.168 |

## Outlier Files

| rel_path | outlier_score |
|---|---:|
| s3-cache-datasets/cache_dataset_lcs/tencentBlock/25829.lcs.zst | 555.808 |
| s3-cache-datasets/cache_dataset_lcs/tencentBlock/25497.lcs.zst | 513.913 |
| s3-cache-datasets/cache_dataset_lcs/tencentBlock/11839.lcs.zst | 465.648 |
| s3-cache-datasets/cache_dataset_lcs/tencentBlock/12500.lcs.zst | 461.611 |
| s3-cache-datasets/cache_dataset_lcs/tencentBlock/21369.lcs.zst | 404.428 |
| s3-cache-datasets/cache_dataset_lcs/tencentBlock/25489.lcs.zst | 278.678 |
| s3-cache-datasets/cache_dataset_lcs/tencentBlock/3330.lcs.zst | 273.502 |
| s3-cache-datasets/cache_dataset_lcs/tencentBlock/25800.lcs.zst | 226.331 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv | ts_duration |
|---|---|---:|---:|---:|---:|
| s3-cache-datasets/cache_dataset_lcs/tencentBlock/10370.lcs.zst | lcs | 1 | 0.001 | 63.984 | 2 |
| s3-cache-datasets/cache_dataset_lcs/tencentBlock/1063.lcs.zst | lcs | 0.581 | 0.001 | 63.984 | 1 |
| s3-cache-datasets/cache_dataset_lcs/tencentBlock/1108.lcs.zst | lcs | 0.293 | 0 | 63.984 | 1 |
| s3-cache-datasets/cache_dataset_lcs/tencentBlock/1121.lcs.zst | lcs | 0.993 | 0.001 | 63.984 | 1 |
| s3-cache-datasets/cache_dataset_lcs/tencentBlock/1125.lcs.zst | lcs | 0.229 | 0.006 | 63.984 | 1 |
| s3-cache-datasets/cache_dataset_lcs/tencentBlock/12200.lcs.zst | lcs | 0.022 | 0.016 | 63.984 | 1 |
| s3-cache-datasets/cache_dataset_lcs/tencentBlock/1252.lcs.zst | lcs | 0.996 | 0.003 | 63.984 | 1 |
| s3-cache-datasets/cache_dataset_lcs/tencentBlock/12695.lcs.zst | lcs | 1 | 0.002 | 63.984 | 1 |
