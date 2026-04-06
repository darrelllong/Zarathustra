# s3-cache-datasets / 2022_metaStorage

- Files: 5
- Bytes: 616636804
- Formats: oracle_general
- Parsers: oracle_general
- ML Use Cases: request_sequence
- Heterogeneity Score: 0.018
- Suggested GAN Modes: 1
- Split By Format: no

## Observations

- Predominantly read-heavy.

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
| Highly redundant current pairs | reuse_ratio vs backward_seek_ratio (-0.975) |

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| oracle_general | 5 | oracle_general |

## Clustering And Regimes

| Item | Value |
|---|---|
| PCA variance explained by PC1 | 0.389 |

## Strongest Correlations

| Metric A | Metric B | Correlation |
|---|---|---:|
| ts_duration | iat_mean | 1 |
| ts_duration | sample_record_rate | -1 |
| sample_record_rate | iat_mean | -1 |
| object_top1_share | object_top10_share | 0.995 |
| obj_size_q50 | abs_stride_q99 | 0.993 |
| abs_stride_std | abs_stride_q99 | 0.993 |
| reuse_ratio | backward_seek_ratio | -0.975 |
| obj_size_q50 | abs_stride_std | 0.974 |

## Metrics

| Metric | Mean | Median | CV | Skew | Kurtosis | Missing | Q10 | Q90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| iat_lag1_autocorr | -0.006 | -0.007 | 2.497 | 1.263 | 1.661 | 0 | -0.019 | 0.009 |
| object_top1_share | 0.009 | 0.004 | 1.259 | 2.229 | 4.976 | 0 | 0.004 | 0.019 |
| object_top10_share | 0.03 | 0.025 | 0.458 | 2.141 | 4.663 | 0 | 0.022 | 0.042 |
| reuse_ratio | 0.243 | 0.242 | 0.056 | 0.309 | 1.885 | 0 | 0.232 | 0.256 |
| size_bytes | 123327361 | 124294121 | 0.043 | -0.019 | -1.468 | 0 | 117790146 | 128534922 |
| obj_size_mean | 3636634 | 3670309 | 0.043 | -1.022 | 0.619 | 0 | 3469539 | 3772511 |
| abs_stride_q50 | 3462618179223718912 | 3483432334346461696 | 0.037 | -0.554 | 0.921 | 0 | 3332374886531191808 | 3578194699000949248 |
| tenant_top1_share | 0.519 | 0.514 | 0.028 | 0.737 | -1.465 | 0 | 0.507 | 0.535 |
| object_unique | 2415 | 2377 | 0.027 | 1.17 | -0.081 | 0 | 2368 | 2487.8 |
| signed_stride_lag1_autocorr | -0.419 | -0.414 | 0.024 | -1.303 | 0.828 | 0 | -0.43 | -0.411 |
| iat_std | 0.139 | 0.139 | 0.023 | -0.267 | -2.696 | 0 | 0.135 | 0.142 |
| backward_seek_ratio | 0.375 | 0.373 | 0.023 | 0.61 | 1.779 | 0 | 0.368 | 0.383 |
| obj_size_std | 3250958 | 3234544 | 0.022 | 1.315 | 2.665 | 0 | 3193887 | 3323033 |
| obj_size_min | 38.8 | 39 | 0.022 | 0.512 | -0.612 | 0 | 38 | 39.6 |
| abs_stride_q99 | 16394962991533418496 | 16405537515097364480 | 0.019 | 0.183 | 1.217 | 0 | 16102043912136437760 | 16688727231210618880 |
| burstiness_cv | 7.335 | 7.272 | 0.018 | 0.597 | -2.736 | 0 | 7.222 | 7.484 |
| forward_seek_ratio | 0.382 | 0.382 | 0.015 | -1.356 | 2.258 | 0 | 0.376 | 0.387 |
| ts_duration | 77.4 | 77 | 0.015 | 0.405 | -0.178 | 0 | 76.4 | 78.6 |

## Outlier Files

| rel_path | outlier_score | top drivers |
|---|---:|---|
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaStorage/block_traces_1.oracleGeneral.bin.zst | 3.198 | abs_stride_q99 (z=-3.648); obj_size_std (z=-3.595) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaStorage/block_traces_3.oracleGeneral.bin.zst | 3.192 | reuse_ratio (z=7.333); abs_stride_std (z=3.953) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaStorage/block_traces_2.oracleGeneral.bin.zst | 3.19 | object_top1_share (z=51); object_top10_share (z=8.286) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaStorage/block_traces_4.oracleGeneral.bin.zst | 1.284 | object_unique (z=15.333); reuse_ratio (z=-5.75) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaStorage/block_traces_5.oracleGeneral.bin.zst | 1.137 | object_unique (z=7.778); iat_std (z=-1.649) |

## Outlier Sensitivity

| N Removed | Metric | Baseline Median | Trimmed Median | Relative Shift |
|---:|---|---:|---:|---:|
| 3 | object_unique | 2377 | 2481 | 0.044 |
| 3 | reuse_ratio | 0.242 | 0.233 | -0.035 |
| 1 | object_unique | 2377 | 2412 | 0.015 |
| 1 | abs_stride_mean | 4670174504930991104 | 4700168190619619328 | 0.006 |
| 3 | abs_stride_mean | 4670174504930991104 | 4700168190619619328 | 0.006 |
| 1 | burstiness_cv | 7.272 | 7.248 | -0.003 |
| 3 | burstiness_cv | 7.272 | 7.248 | -0.003 |
| 1 | obj_size_std | 3234544 | 3243046 | 0.003 |
| 3 | obj_size_std | 3234544 | 3229564 | -0.002 |
| 1 | reuse_ratio | 0.242 | 0.242 | 0 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv | ts_duration |
|---|---|---:|---:|---:|---:|
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaStorage/block_traces_2.oracleGeneral.bin.zst | oracle_general | 0 | 0.242 | 7.505 | 77 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaStorage/block_traces_1.oracleGeneral.bin.zst | oracle_general | 0 | 0.245 | 7.452 | 78 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaStorage/block_traces_5.oracleGeneral.bin.zst | oracle_general | 0 | 0.242 | 7.272 | 76 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaStorage/block_traces_4.oracleGeneral.bin.zst | oracle_general | 0 | 0.225 | 7.224 | 77 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaStorage/block_traces_3.oracleGeneral.bin.zst | oracle_general | 0 | 0.263 | 7.221 | 79 |
