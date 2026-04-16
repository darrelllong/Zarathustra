# s3-cache-datasets / 2020_twr_cdn

- Files: 3
- Bytes: 524831812700
- Formats: parquet, text_zst
- Parsers: 2020_twr_cdn_text, parquet_duckdb
- ML Use Cases: structured_table
- Heterogeneity Score: 0.079
- Suggested GAN Modes: 1
- Split By Format: yes

## Observations

- No single dominant behavioral note stood out from the sampled features.

## GAN Guidance

- Family spans multiple encodings; keep format-aware preprocessing and avoid blindly pooling structured-table and request-sequence variants.

## Conditioning Audit

| Item | Value |
|---|---|
| Near-constant current conditioning features | none flagged |
| Recommended candidate additions | none flagged |
| Highly redundant current pairs | none flagged |

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| parquet | 2 | parquet_duckdb |
| text_zst | 1 | 2020_twr_cdn_text |

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
| schema_mixed_cols | 0.333 | 0 | 1.732 | 1.732 | N/A | 0 | 0 | 0.8 |
| size_bytes | 174943937567 | 231608521285 | 0.854 | -1.461 | N/A | 0 | 50716665623 | 276505376023 |
| schema_numeric_cols | 3.667 | 4 | 0.157 | -1.732 | N/A | 0 | 3.2 | 4 |
| sample_records | 4096 | 4096 | 0 | N/A | N/A | 0 | 4096 | 4096 |
| schema_column_count | 14 | 14 | 0 | N/A | N/A | 0 | 14 | 14 |
| schema_high_cardinality_cols | 8 | 8 | 0 | N/A | N/A | 0 | 8 | 8 |
| ttl_present | 0 | 0 | N/A | N/A | N/A | 0 | 0 | 0 |

## Outlier Files

| rel_path | outlier_score | top drivers |
|---|---:|---|
| N/A | N/A | N/A |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv | ts_duration |
|---|---|---:|---:|---:|---:|
| s3-cache-datasets/cache_dataset_parquet/2020_twr_cdn/2020_twr_cdn.parquet | parquet | N/A | N/A | N/A | N/A |
| s3-cache-datasets/cache_dataset_parquet/2020_twr_cdn/2020_twr_cdn.sample100.parquet | parquet | N/A | N/A | N/A | N/A |
| s3-cache-datasets/cache_dataset_txt/2020_twr_cdn.zst | text_zst | N/A | N/A | N/A | N/A |




## Model-Aware Guidance

- Closest learned anchor: tencent_block (distance 2.573)
- Sampling: split-by-format-first
- Regime recipe: single
- Char-file conditioning: no
- PCF: not-primary
- Multi-scale critic: not-primary
- Mixed-type recovery: not-primary
- Retrieval memory: not-primary
- Why: formats/parsers are mixed
