# s3-cache-datasets / alibaba

- Files: 1000
- Bytes: 477220451258
- Formats: lcs
- Parsers: lcs
- ML Use Cases: request_sequence
- Heterogeneity Score: 1.302
- Suggested GAN Modes: 8
- Split By Format: no

## Observations

- Substantial write pressure across sampled files.
- Very weak short-window reuse.
- Ordered feature trajectories show regime boundaries.

## GAN Guidance

- Ordered PC1 changepoints suggest 23 regimes when files are ordered by trace start time.
- Write pressure is material; preserve write bursts and opcode transitions in conditioning.
- Strongest feature coupling in this pass: iat_mean vs iat_q90 (corr=0.99).
- A small set of files are strong multivariate outliers; consider holding them out for ablation or separate mode inspection.

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| lcs | 1000 | lcs |

## Clustering And Regimes

| Item | Value |
|---|---|
| DBSCAN clusters | 1 |
| DBSCAN noise fraction | 0.08 |
| Ordered PC1 changepoints | 22 |
| PCA variance explained by PC1 | 0.274 |

## Strongest Correlations

| Metric A | Metric B | Correlation |
|---|---|---:|
| iat_mean | iat_q90 | 0.994 |
| iat_std | iat_q99 | 0.974 |
| iat_mean | iat_std | 0.973 |
| iat_mean | iat_q99 | 0.962 |
| forward_seek_ratio | backward_seek_ratio | -0.961 |
| iat_std | iat_q90 | 0.953 |
| iat_q90 | iat_q99 | 0.935 |
| abs_stride_std | abs_stride_q99 | 0.918 |

## Metrics

| Metric | Mean | Median | CV | Skew | Kurtosis | Missing | Q10 | Q90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| iat_q90 | 206.572 | 0 | 26.526 | 30.258 | 936.143 | 0 | 0 | 5 |
| iat_mean | 51.374 | 0.141 | 23.537 | 30.003 | 925.89 | 0 | 0 | 2.923 |
| iat_q50 | 0.012 | 0 | 20.736 | 22.119 | 497.544 | 0 | 0 | 0 |
| iat_q99 | 507.048 | 5 | 17.794 | 24.583 | 667.681 | 0 | 0 | 31.254 |
| iat_std | 153.626 | 0.786 | 15.797 | 26.232 | 754.829 | 0 | 0.022 | 12.263 |
| abs_stride_q50 | 4614.367 | 1 | 15.725 | 25.185 | 704.666 | 0 | 1 | 31.1 |
| ts_duration | 21666.54 | 575 | 8.996 | 11.918 | 147.346 | 0 | 2 | 11888.5 |
| reuse_ratio | 0.004 | 0.001 | 6.08 | 17.913 | 378.503 | 0 | 0 | 0.008 |
| size_bytes | 477220451 | 19870765 | 5.412 | 8.993 | 92.779 | 0 | 733126.9 | 363823437 |
| opcode_switch_ratio | 0.003 | 0 | 4.437 | 14.811 | 261.593 | 0 | 0 | 0.005 |
| abs_stride_q90 | 3505808 | 913799.5 | 3.374 | 12.946 | 256.354 | 0 | 1 | 6076779 |
| abs_stride_mean | 995124.4 | 392307 | 2.81 | 9.266 | 116.479 | 0 | 13275.21 | 1814201 |
| abs_stride_q99 | 10165458 | 4553098 | 2.002 | 5.972 | 51.258 | 0 | 41.406 | 23879133 |
| abs_stride_std | 2633708 | 1152817 | 1.98 | 7.298 | 89.022 | 0 | 329212 | 5272177 |
| object_top1_share | 0.015 | 0.008 | 1.653 | 6.485 | 66.578 | 0 | 0.001 | 0.034 |
| object_top10_share | 0.082 | 0.045 | 1.302 | 3.85 | 23.044 | 0 | 0.005 | 0.181 |
| backward_seek_ratio | 0.124 | 0.129 | 0.704 | 0.289 | -0.466 | 0 | 0.006 | 0.227 |
| signed_stride_lag1_autocorr | -0.272 | -0.302 | 0.557 | 0.351 | -0.295 | 0 | -0.44 | 0 |

## Outlier Files

| rel_path | outlier_score |
|---|---:|
| s3-cache-datasets/cache_dataset_lcs/alibaba/809.lcs.zst | 760.456 |
| s3-cache-datasets/cache_dataset_lcs/alibaba/791.lcs.zst | 322.295 |
| s3-cache-datasets/cache_dataset_lcs/alibaba/811.lcs.zst | 114.764 |
| s3-cache-datasets/cache_dataset_lcs/alibaba/893.lcs.zst | 73.588 |
| s3-cache-datasets/cache_dataset_lcs/alibaba/816.lcs.zst | 67.762 |
| s3-cache-datasets/cache_dataset_lcs/alibaba/796.lcs.zst | 45.401 |
| s3-cache-datasets/cache_dataset_lcs/alibaba/4.lcs.zst | 42.333 |
| s3-cache-datasets/cache_dataset_lcs/alibaba/805.lcs.zst | 39.705 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv | ts_duration |
|---|---|---:|---:|---:|---:|
| s3-cache-datasets/cache_dataset_lcs/alibaba/117.lcs.zst | lcs | 0.008 | 0 | 63.984 | 1 |
| s3-cache-datasets/cache_dataset_lcs/alibaba/206.lcs.zst | lcs | 0.011 | 0 | 63.984 | 1 |
| s3-cache-datasets/cache_dataset_lcs/alibaba/4.lcs.zst | lcs | 0.163 | 0 | 63.984 | 1 |
| s3-cache-datasets/cache_dataset_lcs/alibaba/746.lcs.zst | lcs | 1 | 0 | 63.984 | 4 |
| s3-cache-datasets/cache_dataset_lcs/alibaba/801.lcs.zst | lcs | 0.995 | 0 | 63.984 | 5 |
| s3-cache-datasets/cache_dataset_lcs/alibaba/824.lcs.zst | lcs | 0 | 0.101 | 63.984 | 6 |
| s3-cache-datasets/cache_dataset_lcs/alibaba/829.lcs.zst | lcs | 0.038 | 0.001 | 63.984 | 1 |
| s3-cache-datasets/cache_dataset_lcs/alibaba/831.lcs.zst | lcs | 0 | 0.001 | 63.984 | 1 |
