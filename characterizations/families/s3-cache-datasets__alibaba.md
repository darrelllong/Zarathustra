# s3-cache-datasets / alibaba

- Files: 1000
- Bytes: 477220451258
- Formats: lcs
- Parsers: lcs
- ML Use Cases: request_sequence
- Heterogeneity Score: 1.302
- Suggested GAN Modes: 1
- Split By Format: no

## Observations

- Substantial write pressure across sampled files.
- Very weak short-window reuse.

## GAN Guidance

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
| PCA variance explained by PC1 | 0.274 |
| Block/random distance ratio | 0.678 |
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
| 8 | 200736652740751982592 | 0.722 |
| 9 | 199528512111614427136 | 0.724 |
| 10 | 494945741504243236864 | 0.612 |
| 11 | 198939014722948497408 | 0.612 |
| 12 | 198812720734478630912 | 0.617 |

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
| iat_q90 | 206.572 | 0 | 26.526 | N/A | N/A | 0 | 0 | 5 |
| iat_mean | 51.374 | 0.141 | 23.537 | N/A | N/A | 0 | 0 | 2.923 |
| iat_q50 | 0.012 | 0 | 20.736 | N/A | N/A | 0 | 0 | 0 |
| iat_q99 | 507.048 | 5 | 17.794 | N/A | N/A | 0 | 0 | 31.254 |
| iat_std | 153.626 | 0.786 | 15.797 | N/A | N/A | 0 | 0.022 | 12.263 |
| abs_stride_q50 | 4614.367 | 1 | 15.725 | N/A | N/A | 0 | 1 | 31.1 |
| ts_duration | 21666.54 | 575 | 8.996 | N/A | N/A | 0 | 2 | 11888.5 |
| reuse_ratio | 0.004 | 0.001 | 6.08 | N/A | N/A | 0 | 0 | 0.008 |
| size_bytes | 477220451 | 19870765 | 5.412 | N/A | N/A | 0 | 733126.9 | 363823437 |
| opcode_switch_ratio | 0.003 | 0 | 4.437 | N/A | N/A | 0 | 0 | 0.005 |
| abs_stride_q90 | 3505808 | 913799.5 | 3.374 | N/A | N/A | 0 | 1 | 6076779 |
| abs_stride_mean | 995124.4 | 392307 | 2.81 | N/A | N/A | 0 | 13275.21 | 1814201 |
| abs_stride_q99 | 10165458 | 4553098 | 2.002 | N/A | N/A | 0 | 41.406 | 23879133 |
| abs_stride_std | 2633708 | 1152817 | 1.98 | N/A | N/A | 0 | 329212 | 5272177 |
| object_top1_share | 0.015 | 0.008 | 1.653 | N/A | N/A | 0 | 0.001 | 0.034 |
| object_top10_share | 0.082 | 0.045 | 1.302 | N/A | N/A | 0 | 0.005 | 0.181 |
| backward_seek_ratio | 0.124 | 0.129 | 0.704 | N/A | N/A | 0 | 0.006 | 0.227 |
| signed_stride_lag1_autocorr | -0.272 | -0.302 | 0.557 | N/A | N/A | 0 | -0.44 | 0 |

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
