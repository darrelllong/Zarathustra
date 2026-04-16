# s3-cache-datasets / 2023_metaCDN

- Files: 3
- Bytes: 3731742149
- Formats: text_zst
- Parsers: 2023_metaCDN_text
- ML Use Cases: structured_table
- Heterogeneity Score: 0.146
- Suggested GAN Modes: 1
- Split By Format: no

## Observations

- No single dominant behavioral note stood out from the sampled features.

## GAN Guidance

- Family looks comparatively homogeneous in the extracted feature space.

## Conditioning Audit

| Item | Value |
|---|---|
| Near-constant current conditioning features | none flagged |
| Recommended candidate additions | none flagged |
| Highly redundant current pairs | none flagged |

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| text_zst | 3 | 2023_metaCDN_text |

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
| first_numeric_diff_std | 706.857 | 919.727 | 0.796 | -1.459 | N/A | 0 | 239.001 | 1089.566 |
| first_numeric_diff_mean | 28.359 | 34.507 | 0.476 | -1.624 | N/A | 0 | 17.204 | 37.056 |
| size_bytes | 1243914050 | 1442532807 | 0.434 | -1.432 | N/A | 0 | 794844731 | 1613535865 |
| first_numeric_diff_q90 | 32 | 28 | 0.272 | 1.63 | N/A | 0 | 26.4 | 39.2 |
| first_numeric_diff_q50 | 8 | 7 | 0.217 | 1.732 | N/A | 0 | 7 | 9.4 |
| schema_high_cardinality_cols | 7.667 | 8 | 0.075 | -1.732 | N/A | 0 | 7.2 | 8 |
| sample_records | 4096 | 4096 | 0 | N/A | N/A | 0 | 4096 | 4096 |
| schema_column_count | 15 | 15 | 0 | N/A | N/A | 0 | 15 | 15 |
| schema_numeric_cols | 15 | 15 | 0 | N/A | N/A | 0 | 15 | 15 |
| first_numeric_monotone_ratio | 1 | 1 | 0 | N/A | N/A | 0 | 1 | 1 |
| ttl_present | 0 | 0 | N/A | N/A | N/A | 0 | 0 | 0 |
| schema_mixed_cols | 0 | 0 | N/A | N/A | N/A | 0 | 0 | 0 |

## Outlier Files

| rel_path | outlier_score | top drivers |
|---|---:|---|
| N/A | N/A | N/A |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv | ts_duration |
|---|---|---:|---:|---:|---:|
| s3-cache-datasets/cache_dataset_txt/2023_metaCDN/reag0c01_20230315_20230322_0.2000.csv.zst | text_zst | N/A | N/A | N/A | N/A |
| s3-cache-datasets/cache_dataset_txt/2023_metaCDN/rnha0c01_20230315_20230322_0.8000.csv.zst | text_zst | N/A | N/A | N/A | N/A |
| s3-cache-datasets/cache_dataset_txt/2023_metaCDN/rprn0c01_20230315_20230322_0.2000.csv.zst | text_zst | N/A | N/A | N/A | N/A |




## Model-Aware Guidance

- Closest learned anchor: tencent_block (distance 2.483)
- Sampling: random-ok
- Regime recipe: single
- Char-file conditioning: no
- PCF: not-primary
- Multi-scale critic: not-primary
- Mixed-type recovery: not-primary
- Retrieval memory: not-primary
- Why: no single pathological axis dominates this family
