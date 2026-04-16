# s3-cache-datasets / 2022_metaCDN

- Files: 3
- Bytes: 2234960001
- Formats: oracle_general
- Parsers: oracle_general
- ML Use Cases: request_sequence
- Heterogeneity Score: 0.195
- Suggested GAN Modes: 1
- Split By Format: no

## Observations

- Predominantly read-heavy.
- Highly bursty arrivals.

## GAN Guidance

- Opcode balance is extremely read-skewed; generation should not assume symmetric read/write behavior.
- Burstiness is high; inter-arrival and FFT/ACF losses should stay heavily weighted.
- Current characterization suggests extra conditioning value from: object_unique, signed_stride_lag1_autocorr, obj_size_std.

## Conditioning Audit

| Item | Value |
|---|---|
| Near-constant current conditioning features | write_ratio, iat_q50, opcode_switch_ratio, tenant_unique |
| Recommended candidate additions | object_unique, signed_stride_lag1_autocorr, obj_size_std |
| Highly redundant current pairs | none flagged |

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| oracle_general | 3 | oracle_general |

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
| obj_size_q99 | 1525854778 | 228342191 | 1.573 | 1.722 | N/A | 0 | 89072317 | 3481642274 |
| obj_size_q90 | 20024696 | 7813858 | 1.117 | 1.724 | N/A | 0 | 6691581 | 38242147 |
| iat_lag1_autocorr | 0.328 | 0.381 | 0.969 | -0.721 | N/A | 0 | 0.066 | 0.57 |
| object_top1_share | 0.112 | 0.112 | 0.915 | -0.004 | N/A | 0 | 0.03 | 0.194 |
| obj_size_min | 2 | 1 | 0.866 | 1.732 | N/A | 0 | 1 | 3.4 |
| iat_std | 0.578 | 0.699 | 0.726 | -1.193 | N/A | 0 | 0.229 | 0.878 |
| object_top10_share | 0.18 | 0.162 | 0.724 | 0.624 | N/A | 0 | 0.08 | 0.288 |
| sample_record_rate | 49.705 | 40.554 | 0.547 | 1.342 | N/A | 0 | 30.71 | 72.362 |
| reuse_ratio | 0.106 | 0.084 | 0.516 | 1.506 | N/A | 0 | 0.069 | 0.152 |
| burstiness_cv | 21.102 | 26.064 | 0.503 | -1.643 | N/A | 0 | 12.337 | 27.884 |
| ts_duration | 99 | 101 | 0.475 | -0.191 | N/A | 0 | 61 | 136.2 |
| iat_mean | 0.024 | 0.025 | 0.475 | -0.191 | N/A | 0 | 0.015 | 0.033 |
| size_bytes | 744986667 | 863872763 | 0.471 | -1.35 | N/A | 0 | 453030194 | 989388701 |
| obj_size_q50 | 54198.5 | 56574.5 | 0.332 | -0.584 | N/A | 0 | 39422.1 | 68024.5 |
| obj_size_mean | 43419057 | 42107931 | 0.32 | 0.42 | N/A | 0 | 32592380 | 54770183 |
| object_unique | 2652.333 | 2736 | 0.195 | -0.708 | N/A | 0 | 2225.6 | 3045.6 |
| obj_size_std | 391311157 | 390096570 | 0.141 | 0.099 | N/A | 0 | 347447216 | 435660933 |
| forward_seek_ratio | 0.447 | 0.453 | 0.071 | -0.701 | N/A | 0 | 0.421 | 0.471 |

## Outlier Files

| rel_path | outlier_score | top drivers |
|---|---:|---|
| N/A | N/A | N/A |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv | ts_duration |
|---|---|---:|---:|---:|---:|
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaCDN/meta_rprn.oracleGeneral.zst | oracle_general | 0 | 0.066 | 28.338 | 101 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaCDN/meta_reag.oracleGeneral.zst | oracle_general | 0 | 0.169 | 26.064 | 145 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaCDN/meta_rnha.oracleGeneral.zst | oracle_general | 0 | 0.084 | 8.905 | 51 |




## Model-Aware Guidance

- Closest learned anchor: tencent_block (distance 7.353)
- Sampling: random-ok
- Regime recipe: single
- Char-file conditioning: yes
- PCF: validated
- Multi-scale critic: promising
- Mixed-type recovery: promising
- Retrieval memory: mixed
- Why: burstiness is materially above the calmer families
- Candidate conditioning additions: object_unique,signed_stride_lag1_autocorr,obj_size_std
