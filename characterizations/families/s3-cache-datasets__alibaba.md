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
- Sequential blocks are much more internally coherent than random file batches; block or curriculum sampling is likely safer than pure iid file sampling.
- Write pressure is material; preserve write bursts and opcode transitions in conditioning.
- Strongest feature coupling in this pass: iat_mean vs iat_q90 (corr=0.99).
- A small set of files are strong multivariate outliers; consider holding them out for ablation or separate mode inspection.
- Current characterization suggests extra conditioning value from: object_unique, signed_stride_lag1_autocorr.

## Conditioning Audit

| Item | Value |
|---|---|
| Near-constant current conditioning features | iat_q50, obj_size_q50, tenant_unique |
| Recommended candidate additions | object_unique, signed_stride_lag1_autocorr |
| Highly redundant current pairs | forward_seek_ratio vs backward_seek_ratio (-0.961) |

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| lcs | 1000 | lcs |

## Clustering And Regimes

| Item | Value |
|---|---|
| K-means selected K | 2 |
| Best silhouette K | 2 |
| DBSCAN clusters | 1 |
| DBSCAN noise fraction | 0.08 |
| Ordered PC1 changepoints | 22 |
| PCA variance explained by PC1 | 0.274 |
| Hurst exponent on ordered PC1 | 0.695 |
| Block/random distance ratio | 0.688 |
| Sampling recommendation | block_sampling_preserves_temporal_coherence |

### K Selection

| K | Within-SS | Silhouette |
|---:|---:|---:|
| 2 | 1115731198562949660672 | 0.977 |
| 3 | 648575049005363888128 | 0.941 |
| 4 | 622952439118409891840 | 0.779 |
| 5 | 504104469511542931456 | 0.791 |
| 6 | 496918499784578826240 | 0.79 |
| 7 | 495712727490412281856 | 0.718 |
| 8 | 495283243302627704832 | 0.667 |
| 9 | 199528512111614427136 | 0.724 |
| 10 | 494794491241867837440 | 0.648 |
| 11 | 198939014722948497408 | 0.612 |
| 12 | 198812720734478630912 | 0.617 |

## Regime Transition Drivers

| Transition | Driver 1 | Effect | Driver 2 | Effect | Driver 3 | Effect |
|---|---|---:|---|---:|---|---:|
| 1 -> 2 | sample_record_rate | 17.794 | burstiness_cv | 16.573 | iat_zero_ratio | 3.589 |
| 2 -> 3 | sample_record_rate | 18.025 | burstiness_cv | 15.382 | object_unique | 3.618 |
| 3 -> 4 | write_ratio | 1.414 | opcode_switch_ratio | 1.414 | reuse_ratio | 1.414 |
| 4 -> 5 | abs_stride_q90 | 1.758 | reuse_ratio | 1.414 | abs_stride_q50 | 1.414 |
| 5 -> 6 | abs_stride_std | 0.837 | forward_seek_ratio | 0.713 | backward_seek_ratio | 0.709 |
| 6 -> 7 | iat_q90 | 2.357 | ts_duration | 2.03 | iat_mean | 1.98 |
| 7 -> 8 | reuse_ratio | 1.414 | write_ratio | 1.414 | opcode_switch_ratio | 1.414 |
| 8 -> 9 | burstiness_cv | 3.055 | write_ratio | 2.493 | iat_zero_ratio | 2.181 |
| 9 -> 10 | sample_record_rate | 28.241 | iat_lag1_autocorr | 7.355 | reuse_ratio | 1.961 |
| 10 -> 11 | size_bytes | 3.884 | signed_stride_lag1_autocorr | 2.661 | write_ratio | 2.002 |
| 11 -> 12 | size_bytes | 2.21 | write_ratio | 1.414 | opcode_switch_ratio | 1.414 |
| 12 -> 13 | burstiness_cv | 6.679 | sample_record_rate | 3.143 | iat_zero_ratio | 2.828 |
| 13 -> 14 | ts_duration | 1.414 | iat_mean | 1.414 | write_ratio | 1.414 |
| 14 -> 15 | abs_stride_q99 | 0.644 | iat_lag1_autocorr | 0.54 | size_bytes | 0.489 |
| 15 -> 16 | backward_seek_ratio | 1.462 | forward_seek_ratio | 1.458 | size_bytes | 1.447 |
| 16 -> 17 | burstiness_cv | 3.115 | abs_stride_std | 1.598 | sample_record_rate | 1.581 |
| 17 -> 18 | sample_record_rate | 4.263 | burstiness_cv | 3.953 | size_bytes | 2.984 |
| 18 -> 19 | sample_record_rate | 2.005 | burstiness_cv | 1.997 | iat_zero_ratio | 1.715 |
| 19 -> 20 | size_bytes | 1.878 | backward_seek_ratio | 1.508 | forward_seek_ratio | 1.507 |
| 20 -> 21 | size_bytes | 3.686 | abs_stride_mean | 2.111 | forward_seek_ratio | 0.971 |
| 21 -> 22 | sample_record_rate | 1245.922 | abs_stride_std | 162.244 | iat_std | 114.903 |
| 22 -> 23 | abs_stride_std | 142.888 | abs_stride_mean | 49.808 | iat_std | 30.177 |

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

| rel_path | outlier_score | top drivers |
|---|---:|---|
| s3-cache-datasets/cache_dataset_lcs/alibaba/809.lcs.zst | 760.456 | iat_mean (z=267650.8); iat_std (z=100006.2) |
| s3-cache-datasets/cache_dataset_lcs/alibaba/791.lcs.zst | 322.295 | abs_stride_q90 (z=292.244); abs_stride_mean (z=177.298) |
| s3-cache-datasets/cache_dataset_lcs/alibaba/811.lcs.zst | 114.764 | iat_mean (z=44878.13); iat_std (z=22241.43) |
| s3-cache-datasets/cache_dataset_lcs/alibaba/893.lcs.zst | 73.588 | opcode_switch_ratio (z=408.5); abs_stride_mean (z=95.709) |
| s3-cache-datasets/cache_dataset_lcs/alibaba/816.lcs.zst | 67.762 | iat_mean (z=27235.56); iat_std (z=22950.63) |
| s3-cache-datasets/cache_dataset_lcs/alibaba/796.lcs.zst | 45.401 | reuse_ratio (z=832); abs_stride_q90 (z=69.654) |
| s3-cache-datasets/cache_dataset_lcs/alibaba/4.lcs.zst | 42.333 | sample_record_rate (z=615.205); opcode_switch_ratio (z=479.5) |
| s3-cache-datasets/cache_dataset_lcs/alibaba/805.lcs.zst | 39.705 | iat_std (z=4755.967); ts_duration (z=4089.091) |

## Outlier Sensitivity

| N Removed | Metric | Baseline Median | Trimmed Median | Relative Shift |
|---:|---|---:|---:|---:|
| 10 | abs_stride_mean | 392307 | 387023.2 | -0.014 |
| 5 | abs_stride_mean | 392307 | 388179 | -0.011 |
| 10 | burstiness_cv | 6.399 | 6.44 | 0.006 |
| 1 | abs_stride_mean | 392307 | 390090.4 | -0.006 |
| 3 | abs_stride_mean | 392307 | 390090.4 | -0.006 |
| 5 | burstiness_cv | 6.399 | 6.429 | 0.005 |
| 10 | object_unique | 2960 | 2971.5 | 0.004 |
| 5 | object_unique | 2960 | 2970 | 0.003 |
| 3 | burstiness_cv | 6.399 | 6.418 | 0.003 |
| 1 | burstiness_cv | 6.399 | 6.408 | 0.001 |

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
