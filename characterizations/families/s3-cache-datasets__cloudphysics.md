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

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| lcs | 106 | lcs |
| text | 1 | generic_text |

## Clustering And Regimes

| Item | Value |
|---|---|
| DBSCAN clusters | 1 |
| DBSCAN noise fraction | 0.078 |
| Ordered PC1 changepoints | 2 |
| PCA variance explained by PC1 | 0.282 |

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

| rel_path | outlier_score |
|---|---:|
| s3-cache-datasets/cache_dataset_lcs/cloudphysics/w54_vscsi2.vscsitrace.lcs.zst | 82.267 |
| s3-cache-datasets/cache_dataset_lcs/cloudphysics/w86_vscsi1.vscsitrace.lcs.zst | 60.969 |
| s3-cache-datasets/cache_dataset_lcs/cloudphysics/w05_vscsi1.vscsitrace.lcs.zst | 13.053 |
| s3-cache-datasets/cache_dataset_lcs/cloudphysics/w93_vscsi2.vscsitrace.lcs.zst | 11.904 |
| s3-cache-datasets/cache_dataset_lcs/cloudphysics/w12_vscsi1.vscsitrace.lcs.zst | 9.903 |
| s3-cache-datasets/cache_dataset_lcs/cloudphysics/w28_vscsi2.vscsitrace.lcs.zst | 7.729 |
| s3-cache-datasets/cache_dataset_lcs/cloudphysics/w85_vscsi1.vscsitrace.lcs.zst | 7.682 |
| s3-cache-datasets/cache_dataset_lcs/cloudphysics/w55_vscsi2.vscsitrace.lcs.zst | 6.802 |

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
