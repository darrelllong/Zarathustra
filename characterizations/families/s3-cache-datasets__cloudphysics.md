# s3-cache-datasets / cloudphysics

- Files: 107
- Bytes: 64319532236
- Formats: lcs, text
- Parsers: generic_text, lcs
- ML Use Cases: request_sequence, structured_table
- Heterogeneity Score: 0.895
- Suggested GAN Modes: 3
- Split By Format: yes

## Observations

- Substantial write pressure across sampled files.
- Very weak short-window reuse.
- Ordered feature trajectories show regime boundaries.

## GAN Guidance

- Family spans multiple encodings; keep format-aware preprocessing and avoid blindly pooling structured-table and request-sequence variants.
- Ordered PC1 changepoints suggest 3 regimes when files are ordered by trace start time.
- Write pressure is material; preserve write bursts and opcode transitions in conditioning.
- Strongest feature coupling in this pass: ts_duration vs iat_mean (corr=1).
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
| lcs | 106 | lcs |
| text | 1 | generic_text |

## Clustering And Regimes

| Item | Value |
|---|---|
| K-means selected K | 3 |
| Best silhouette K | 3 |
| DBSCAN clusters | 1 |
| DBSCAN noise fraction | 0.078 |
| Ordered PC1 changepoints | 2 |
| PCA variance explained by PC1 | 0.282 |
| Hurst exponent on ordered PC1 | 0.662 |
| Block/random distance ratio | 0.853 |
| Sampling recommendation | random_sampling_is_less_problematic |

### K Selection

| K | Within-SS | Silhouette |
|---:|---:|---:|
| 2 | 64198718851476414464 | 0.861 |
| 3 | 17620092236779698176 | 0.868 |
| 4 | 7204102018602645504 | 0.808 |
| 5 | 4876422119149501440 | 0.798 |
| 6 | 4449795845681904640 | 0.616 |
| 7 | 2122115946228761088 | 0.613 |
| 8 | 3991353661505692160 | 0.591 |
| 9 | 1663673762052548608 | 0.589 |
| 10 | 1569006735741307392 | 0.486 |
| 11 | 1167568130608715008 | 0.511 |
| 12 | 1506528322292702464 | 0.496 |

## Regime Transition Drivers

| Transition | Driver 1 | Effect | Driver 2 | Effect | Driver 3 | Effect |
|---|---|---:|---|---:|---|---:|
| 1 -> 2 | object_top1_share | 1.249 | iat_std | 1.219 | object_unique | 1.143 |
| 2 -> 3 | iat_mean | 1.768 | ts_duration | 1.768 | iat_std | 1.695 |

## Strongest Correlations

| Metric A | Metric B | Correlation |
|---|---|---:|
| ts_duration | iat_mean | 1 |
| abs_stride_std | abs_stride_q99 | 0.998 |
| abs_stride_mean | abs_stride_std | 0.962 |
| object_top1_share | object_top10_share | 0.947 |
| abs_stride_mean | abs_stride_q99 | 0.945 |
| abs_stride_mean | abs_stride_q90 | 0.893 |
| forward_seek_ratio | backward_seek_ratio | -0.872 |
| iat_mean | iat_std | 0.863 |

## Metrics

| Metric | Mean | Median | CV | Skew | Kurtosis | Missing | Q10 | Q90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| size_bytes | 601117124 | 142725949 | 1.874 | 2.786 | 7.463 | 0 | 51910176 | 1876453933 |
| sample_records | 4057.729 | 4096 | 0.098 | -10.344 | 107 | 0 | 4096 | 4096 |
| ttl_present | 0 | 0 | N/A | N/A | N/A | 0 | 0 | 0 |
| abs_stride_q50 | 1462665 | 1 | 10.225 | 10.295 | 105.99 | 0.009 | 1 | 3 |
| abs_stride_q99 | 94197002 | 8762276 | 7.194 | 10.185 | 104.441 | 0.009 | 1301775 | 80069101 |
| abs_stride_std | 16807464 | 2071953 | 6.236 | 10.053 | 102.546 | 0.009 | 418594.6 | 16326866 |
| abs_stride_mean | 5150842 | 605757.6 | 4.956 | 8.583 | 79.296 | 0.009 | 111624.7 | 5038042 |
| iat_q90 | 0.17 | 0 | 3.853 | 4.18 | 17.73 | 0.009 | 0 | 0 |
| iat_std | 2.03 | 0.255 | 3.699 | 5.945 | 38.49 | 0.009 | 0.025 | 2.254 |
| abs_stride_q90 | 10291245 | 748239.5 | 3.455 | 5.347 | 31.843 | 0.009 | 1 | 18407638 |
| ts_duration | 675.019 | 146.5 | 2.924 | 6.921 | 57.692 | 0.009 | 2.5 | 1488 |
| iat_mean | 0.165 | 0.036 | 2.924 | 6.921 | 57.692 | 0.009 | 0.001 | 0.363 |
| iat_q99 | 2.654 | 1 | 2.41 | 4.707 | 24.398 | 0.009 | 0 | 5 |
| reuse_ratio | 0.029 | 0.003 | 2.094 | 3.098 | 10.293 | 0.009 | 0 | 0.09 |
| opcode_switch_ratio | 0.014 | 0.004 | 1.96 | 3.907 | 18.32 | 0.009 | 0 | 0.035 |
| object_top1_share | 0.014 | 0.005 | 1.386 | 2.244 | 5.477 | 0.009 | 0 | 0.035 |
| object_top10_share | 0.075 | 0.025 | 1.375 | 2.85 | 11.553 | 0.009 | 0.004 | 0.153 |
| backward_seek_ratio | 0.13 | 0.104 | 0.787 | 1.135 | 1.136 | 0.009 | 0.021 | 0.28 |

## Outlier Files

| rel_path | outlier_score | top drivers |
|---|---:|---|
| s3-cache-datasets/cache_dataset_lcs/cloudphysics/w54_vscsi2.vscsitrace.lcs.zst | 82.267 | abs_stride_q99 (z=1018.93); abs_stride_std (z=805.733) |
| s3-cache-datasets/cache_dataset_lcs/cloudphysics/w86_vscsi1.vscsitrace.lcs.zst | 60.969 | iat_std (z=286.128); ts_duration (z=127.171) |
| s3-cache-datasets/cache_dataset_lcs/cloudphysics/w05_vscsi1.vscsitrace.lcs.zst | 13.053 | abs_stride_q90 (z=204.799); abs_stride_mean (z=187.871) |
| s3-cache-datasets/cache_dataset_lcs/cloudphysics/w93_vscsi2.vscsitrace.lcs.zst | 11.904 | iat_q99 (z=43); ts_duration (z=37.555) |
| s3-cache-datasets/cache_dataset_lcs/cloudphysics/w12_vscsi1.vscsitrace.lcs.zst | 9.903 | abs_stride_q90 (z=57.923); object_top10_share (z=29.409) |
| s3-cache-datasets/cache_dataset_lcs/cloudphysics/w28_vscsi2.vscsitrace.lcs.zst | 7.729 | ts_duration (z=27.683); iat_mean (z=27.683) |
| s3-cache-datasets/cache_dataset_lcs/cloudphysics/w85_vscsi1.vscsitrace.lcs.zst | 7.682 | reuse_ratio (z=25.615); ts_duration (z=19.904) |
| s3-cache-datasets/cache_dataset_lcs/cloudphysics/w55_vscsi2.vscsitrace.lcs.zst | 6.802 | abs_stride_q90 (z=24.693); ts_duration (z=13.94) |

## Outlier Sensitivity

| N Removed | Metric | Baseline Median | Trimmed Median | Relative Shift |
|---:|---|---:|---:|---:|
| 10 | reuse_ratio | 0.003 | 0.003 | -0.192 |
| 1 | reuse_ratio | 0.003 | 0.003 | 0.077 |
| 3 | reuse_ratio | 0.003 | 0.003 | 0.077 |
| 5 | reuse_ratio | 0.003 | 0.003 | -0.077 |
| 10 | abs_stride_mean | 605757.6 | 579278 | -0.044 |
| 10 | object_unique | 3048.5 | 3176 | 0.042 |
| 3 | abs_stride_mean | 605757.6 | 584859.5 | -0.034 |
| 5 | abs_stride_mean | 605757.6 | 584859.5 | -0.034 |
| 5 | object_unique | 3048.5 | 3109 | 0.02 |
| 1 | abs_stride_mean | 605757.6 | 599989.5 | -0.01 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv | ts_duration |
|---|---|---:|---:|---:|---:|
| s3-cache-datasets/cache_dataset_lcs/cloudphysics/w03_vscsi1.vscsitrace.lcs.zst | lcs | 0.378 | 0.004 | 63.984 | 1 |
| s3-cache-datasets/cache_dataset_lcs/cloudphysics/w08_vscsi2.vscsitrace.lcs.zst | lcs | 0 | 0 | 63.984 | 1 |
| s3-cache-datasets/cache_dataset_lcs/cloudphysics/w13_vscsi1.vscsitrace.lcs.zst | lcs | 0 | 0 | 63.984 | 1 |
| s3-cache-datasets/cache_dataset_lcs/cloudphysics/w14_vscsi1.vscsitrace.lcs.zst | lcs | 0.424 | 0.003 | 63.984 | 1 |
| s3-cache-datasets/cache_dataset_lcs/cloudphysics/w22_vscsi2.vscsitrace.lcs.zst | lcs | 0.435 | 0 | 63.984 | 1 |
| s3-cache-datasets/cache_dataset_lcs/cloudphysics/w30_vscsi1.vscsitrace.lcs.zst | lcs | 0.999 | 0 | 63.984 | 17 |
| s3-cache-datasets/cache_dataset_lcs/cloudphysics/w54_vscsi2.vscsitrace.lcs.zst | lcs | 0 | 0 | 63.411 | 893 |
| s3-cache-datasets/cache_dataset_lcs/cloudphysics/w20_vscsi1.vscsitrace.lcs.zst | lcs | 0.042 | 0.004 | 45.238 | 2 |
