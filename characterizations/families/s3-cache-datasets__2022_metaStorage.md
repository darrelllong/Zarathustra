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
- Current characterization suggests extra conditioning value from: object_unique, obj_size_std.

## Conditioning Audit

| Item | Value |
|---|---|
| Near-constant current conditioning features | write_ratio, iat_q50, opcode_switch_ratio, tenant_unique |
| Recommended candidate additions | object_unique, obj_size_std |
| Highly redundant current pairs | reuse_ratio vs backward_seek_ratio (-0.975) |

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| oracle_general | 5 | oracle_general |

## Clustering And Regimes

| Item | Value |
|---|---|
| PCA variance explained by PC1 | 0.394 |

## Strongest Correlations

| Metric A | Metric B | Correlation |
|---|---|---:|
| ts_duration | iat_mean | 1 |
| ts_duration | sample_record_rate | -1 |
| sample_record_rate | iat_mean | -1 |
| object_top1_share | object_top10_share | 0.995 |
| reuse_ratio | backward_seek_ratio | -0.975 |
| size_bytes | iat_std | -0.959 |
| obj_size_mean | obj_size_q50 | 0.956 |
| burstiness_cv | tenant_top1_share | 0.951 |

## Metrics

| Metric | Mean | Median | CV | Skew | Kurtosis | Missing | Q10 | Q90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| iat_lag1_autocorr | -0.006 | -0.007 | 2.497 | N/A | N/A | 0 | -0.019 | 0.009 |
| object_top1_share | 0.009 | 0.004 | 1.259 | N/A | N/A | 0 | 0.004 | 0.019 |
| object_top10_share | 0.03 | 0.025 | 0.458 | N/A | N/A | 0 | 0.022 | 0.042 |
| reuse_ratio | 0.243 | 0.242 | 0.056 | N/A | N/A | 0 | 0.232 | 0.256 |
| size_bytes | 123327361 | 124294121 | 0.043 | N/A | N/A | 0 | 117790146 | 128534922 |
| obj_size_mean | 3636634 | 3670309 | 0.043 | N/A | N/A | 0 | 3469539 | 3772511 |
| tenant_top1_share | 0.519 | 0.514 | 0.028 | N/A | N/A | 0 | 0.507 | 0.535 |
| object_unique | 2415 | 2377 | 0.027 | N/A | N/A | 0 | 2368 | 2487.8 |
| iat_std | 0.139 | 0.139 | 0.023 | N/A | N/A | 0 | 0.135 | 0.142 |
| backward_seek_ratio | 0.375 | 0.373 | 0.023 | N/A | N/A | 0 | 0.368 | 0.383 |
| obj_size_std | 3250958 | 3234544 | 0.022 | N/A | N/A | 0 | 3193887 | 3323033 |
| obj_size_min | 38.8 | 39 | 0.022 | N/A | N/A | 0 | 38 | 39.6 |
| burstiness_cv | 7.335 | 7.272 | 0.018 | N/A | N/A | 0 | 7.222 | 7.484 |
| forward_seek_ratio | 0.382 | 0.382 | 0.015 | N/A | N/A | 0 | 0.376 | 0.387 |
| ts_duration | 77.4 | 77 | 0.015 | N/A | N/A | 0 | 76.4 | 78.6 |
| iat_mean | 0.019 | 0.019 | 0.015 | N/A | N/A | 0 | 0.019 | 0.019 |
| sample_record_rate | 52.929 | 53.195 | 0.015 | N/A | N/A | 0 | 52.114 | 53.615 |
| obj_size_q50 | 2107802 | 2109440 | 0.004 | N/A | N/A | 0 | 2100429 | 2114355 |

## Outlier Files

| rel_path | outlier_score | top drivers |
|---|---:|---|
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaStorage/block_traces_1.oracleGeneral.bin.zst | 3.2 | obj_size_std (z=-3.595); burstiness_cv (z=3.564) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaStorage/block_traces_3.oracleGeneral.bin.zst | 3.179 | reuse_ratio (z=7.333); backward_seek_ratio (z=-3.083) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaStorage/block_traces_2.oracleGeneral.bin.zst | 3.131 | object_top1_share (z=51); object_top10_share (z=8.286) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaStorage/block_traces_4.oracleGeneral.bin.zst | 1.387 | object_unique (z=15.333); reuse_ratio (z=-5.75) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaStorage/block_traces_5.oracleGeneral.bin.zst | 1.103 | object_unique (z=7.778); iat_std (z=-1.649) |

## Outlier Sensitivity

| N Removed | Metric | Baseline Median | Trimmed Median | Relative Shift |
|---:|---|---:|---:|---:|
| 3 | object_unique | 2377 | 2481 | 0.044 |
| 3 | reuse_ratio | 0.242 | 0.233 | -0.035 |
| 1 | object_unique | 2377 | 2412 | 0.015 |
| 1 | burstiness_cv | 7.272 | 7.248 | -0.003 |
| 3 | burstiness_cv | 7.272 | 7.248 | -0.003 |
| 1 | obj_size_std | 3234544 | 3243046 | 0.003 |
| 3 | obj_size_std | 3234544 | 3229564 | -0.002 |
| 1 | reuse_ratio | 0.242 | 0.242 | 0 |
| 1 | iat_q50 | 0 | 0 | 0 |
| 3 | iat_q50 | 0 | 0 | 0 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv | ts_duration |
|---|---|---:|---:|---:|---:|
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaStorage/block_traces_2.oracleGeneral.bin.zst | oracle_general | 0 | 0.242 | 7.505 | 77 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaStorage/block_traces_1.oracleGeneral.bin.zst | oracle_general | 0 | 0.245 | 7.452 | 78 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaStorage/block_traces_5.oracleGeneral.bin.zst | oracle_general | 0 | 0.242 | 7.272 | 76 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaStorage/block_traces_4.oracleGeneral.bin.zst | oracle_general | 0 | 0.225 | 7.224 | 77 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaStorage/block_traces_3.oracleGeneral.bin.zst | oracle_general | 0 | 0.263 | 7.221 | 79 |
