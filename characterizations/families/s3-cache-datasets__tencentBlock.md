# s3-cache-datasets / tencentBlock

- Files: 4995
- Bytes: 817117209599
- Formats: lcs
- Parsers: lcs
- ML Use Cases: request_sequence
- Heterogeneity Score: 1.328
- Suggested GAN Modes: 1
- Split By Format: no

## Observations

- Substantial write pressure across sampled files.
- Very weak short-window reuse.

## GAN Guidance

- Sequential blocks are much more internally coherent than random file batches; block or curriculum sampling is likely safer than pure iid file sampling.
- Write pressure is material; preserve write bursts and opcode transitions in conditioning.
- Strongest feature coupling in this pass: iat_std vs iat_q99 (corr=0.98).
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
| lcs | 4995 | lcs |

## Clustering And Regimes

| Item | Value |
|---|---|
| K-means selected K | 2 |
| Best silhouette K | 2 |
| PCA variance explained by PC1 | 0.287 |
| Block/random distance ratio | 0.83 |
| Sampling recommendation | block_sampling_preserves_temporal_coherence |

### K Selection

| K | Within-SS | Silhouette |
|---:|---:|---:|
| 2 | 1077990332336532357120 | 0.985 |
| 3 | 787620042204492726272 | 0.95 |
| 4 | 380718289478432784384 | 0.9 |
| 5 | 352294003898296696832 | 0.889 |
| 6 | 262720201087332843520 | 0.849 |
| 7 | 252262319040133824512 | 0.767 |
| 8 | 244093063247681748992 | 0.71 |
| 9 | 250795851254761881600 | 0.673 |
| 10 | 242637320322171371520 | 0.674 |
| 11 | 238321780518922256384 | 0.597 |
| 12 | 236854029919949848576 | 0.534 |

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
| size_bytes | 163587029 | 24975086 | 4.895 | N/A | N/A | 0 | 2585280 | 220043821 |
| ttl_present | 0 | 0 | N/A | N/A | N/A | 0 | 0 | 0 |
| iat_q50 | 30.39 | 0 | 45.752 | N/A | N/A | 0.001 | 0 | 0 |
| abs_stride_q50 | 3990.29 | 1 | 40.939 | N/A | N/A | 0.001 | 1 | 15 |
| iat_mean | 143.975 | 0.163 | 12.703 | N/A | N/A | 0.001 | 0.007 | 0.667 |
| iat_q90 | 498.913 | 0 | 11.983 | N/A | N/A | 0.001 | 0 | 2 |
| iat_q99 | 849.467 | 4 | 11.559 | N/A | N/A | 0.001 | 0 | 8 |
| iat_std | 288.894 | 0.67 | 11.11 | N/A | N/A | 0.001 | 0.103 | 2.332 |
| ts_duration | 9823.927 | 666 | 7.336 | N/A | N/A | 0.001 | 27 | 2685 |
| abs_stride_q90 | 2447795 | 674503 | 2.904 | N/A | N/A | 0.001 | 2 | 5727156 |
| reuse_ratio | 0.02 | 0.004 | 2.601 | N/A | N/A | 0.001 | 0 | 0.042 |
| abs_stride_mean | 724530.1 | 393588.3 | 2.477 | N/A | N/A | 0.001 | 39846.5 | 1251208 |
| abs_stride_q99 | 8994874 | 6483778 | 2.304 | N/A | N/A | 0.001 | 412146.1 | 13451440 |
| abs_stride_std | 2088251 | 1359910 | 2.241 | N/A | N/A | 0.001 | 190037.9 | 3387733 |
| object_top1_share | 0.021 | 0.009 | 2.063 | N/A | N/A | 0.001 | 0.002 | 0.045 |
| opcode_switch_ratio | 0.012 | 0.002 | 2.002 | N/A | N/A | 0.001 | 0 | 0.035 |
| object_top10_share | 0.117 | 0.064 | 1.288 | N/A | N/A | 0.001 | 0.014 | 0.26 |
| backward_seek_ratio | 0.087 | 0.078 | 0.691 | N/A | N/A | 0.001 | 0.014 | 0.168 |

## Outlier Files

| rel_path | outlier_score | top drivers |
|---|---:|---|
| s3-cache-datasets/cache_dataset_lcs/tencentBlock/25829.lcs.zst | 555.808 | iat_mean (z=396982.6); iat_std (z=303290) |
| s3-cache-datasets/cache_dataset_lcs/tencentBlock/25497.lcs.zst | 513.913 | iat_mean (z=521134.7); iat_std (z=169408.8) |
| s3-cache-datasets/cache_dataset_lcs/tencentBlock/11839.lcs.zst | 465.648 | abs_stride_std (z=146.902); abs_stride_q99 (z=127.571) |
| s3-cache-datasets/cache_dataset_lcs/tencentBlock/12500.lcs.zst | 461.611 | abs_stride_q90 (z=193.877); abs_stride_mean (z=140.028) |
| s3-cache-datasets/cache_dataset_lcs/tencentBlock/21369.lcs.zst | 404.428 | abs_stride_mean (z=137.012); abs_stride_std (z=126.829) |
| s3-cache-datasets/cache_dataset_lcs/tencentBlock/25489.lcs.zst | 278.678 | iat_mean (z=324153.3); iat_std (z=120310.8) |
| s3-cache-datasets/cache_dataset_lcs/tencentBlock/3330.lcs.zst | 273.502 | abs_stride_q99 (z=127.557); abs_stride_std (z=125.115) |
| s3-cache-datasets/cache_dataset_lcs/tencentBlock/25800.lcs.zst | 226.331 | iat_mean (z=188225.4); iat_std (z=171633.8) |

## Outlier Sensitivity

| N Removed | Metric | Baseline Median | Trimmed Median | Relative Shift |
|---:|---|---:|---:|---:|
| 10 | abs_stride_mean | 393588.3 | 392556.7 | -0.003 |
| 5 | abs_stride_mean | 393588.3 | 392657.1 | -0.002 |
| 10 | burstiness_cv | 4.514 | 4.52 | 0.001 |
| 1 | abs_stride_mean | 393588.3 | 394090.4 | 0.001 |
| 5 | burstiness_cv | 4.514 | 4.519 | 0.001 |
| 3 | abs_stride_mean | 393588.3 | 393173 | -0.001 |
| 3 | burstiness_cv | 4.514 | 4.517 | 0.001 |
| 1 | burstiness_cv | 4.514 | 4.514 | 0 |
| 1 | obj_size_std | 0 | 0 | 0 |
| 1 | reuse_ratio | 0.004 | 0.004 | 0 |

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
