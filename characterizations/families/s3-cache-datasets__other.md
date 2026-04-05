# s3-cache-datasets / other

- Files: 2
- Bytes: 386344048303
- Formats: oracle_general
- Parsers: oracle_general
- ML Use Cases: request_sequence
- Heterogeneity Score: 0.09
- Suggested GAN Modes: 1
- Split By Format: no

## Observations

- Predominantly read-heavy.
- Very weak short-window reuse.
- Highly bursty arrivals.

## GAN Guidance

- Opcode balance is extremely read-skewed; generation should not assume symmetric read/write behavior.
- Burstiness is high; inter-arrival and FFT/ACF losses should stay heavily weighted.

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| oracle_general | 2 | oracle_general |

## Clustering And Regimes

| Item | Value |
|---|---|

## Strongest Correlations

| Metric A | Metric B | Correlation |
|---|---|---:|
| N/A | N/A | N/A |

## Metrics

| Metric | Mean | Median | CV | Skew | Kurtosis | Missing | Q10 | Q90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ts_duration | 0.5 | 0.5 | 1.414 | N/A | N/A | 0 | 0.1 | 0.9 |
| iat_mean | 0 | 0 | 1.414 | N/A | N/A | 0 | 0 | 0 |
| iat_std | 0.008 | 0.008 | 1.414 | N/A | N/A | 0 | 0.002 | 0.014 |
| reuse_ratio | 0.025 | 0.025 | 1.414 | N/A | N/A | 0 | 0.005 | 0.046 |
| abs_stride_q50 | 11146414784 | 11146414784 | 1.414 | N/A | N/A | 0 | 2229284083 | 20063545486 |
| abs_stride_q90 | 137285443430 | 137285443430 | 1.413 | N/A | N/A | 0 | 27536631111 | 247034255749 |
| abs_stride_mean | 46735906894 | 46735906894 | 1.412 | N/A | N/A | 0 | 9398702956 | 84073110832 |
| abs_stride_std | 62182548462 | 62182548462 | 1.409 | N/A | N/A | 0 | 12620180132 | 111744916791 |
| abs_stride_q99 | 246191615659 | 246191615659 | 1.408 | N/A | N/A | 0 | 50098133934 | 442285097383 |
| obj_size_q50 | 6144 | 6144 | 0.471 | N/A | N/A | 0 | 4505.6 | 7782.4 |
| backward_seek_ratio | 0.299 | 0.299 | 0.469 | N/A | N/A | 0 | 0.22 | 0.379 |
| size_bytes | 193172024152 | 193172024152 | 0.375 | N/A | N/A | 0 | 152224302461 | 234119745842 |
| signed_stride_lag1_autocorr | -0.429 | -0.429 | 0.248 | N/A | N/A | 0 | -0.489 | -0.369 |
| opcode_switch_ratio | 0.03 | 0.03 | 0.222 | N/A | N/A | 0 | 0.026 | 0.033 |
| forward_seek_ratio | 0.675 | 0.675 | 0.155 | N/A | N/A | 0 | 0.616 | 0.734 |
| obj_size_q90 | 61440 | 61440 | 0.094 | N/A | N/A | 0 | 58163.2 | 64716.8 |
| object_top1_share | 0.008 | 0.008 | 0.086 | N/A | N/A | 0 | 0.008 | 0.008 |
| write_ratio | 0.016 | 0.016 | 0.084 | N/A | N/A | 0 | 0.016 | 0.017 |

## Outlier Files

| rel_path | outlier_score |
|---|---:|
| N/A | N/A |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv | ts_duration |
|---|---|---:|---:|---:|---:|
| s3-cache-datasets/cache_dataset_oracleGeneral/other/tencent_block.oracleGeneral.zst | oracle_general | 0.015 | 0.051 | 63.984 | 1 |
| s3-cache-datasets/cache_dataset_oracleGeneral/other/alibaba_block2020.oracleGeneral.zst | oracle_general | 0.017 | 0 | N/A | 0 |
