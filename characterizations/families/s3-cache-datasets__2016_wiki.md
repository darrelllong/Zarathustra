# s3-cache-datasets / 2016_wiki

- Files: 2
- Bytes: 26587125241
- Formats: text, text_zst
- Parsers: 2016_wiki_text
- ML Use Cases: structured_table
- Heterogeneity Score: 0
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
| text | 1 | 2016_wiki_text |
| text_zst | 1 | 2016_wiki_text |

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
| size_bytes | 13293562620 | 13293562620 | 1.404 | N/A | N/A | 0 | 2734882963 | 23852242278 |
| sample_records | 4096 | 4096 | 0 | N/A | N/A | 0 | 4096 | 4096 |
| schema_column_count | 13 | 13 | 0 | N/A | N/A | 0 | 13 | 13 |
| schema_numeric_cols | 3 | 3 | 0 | N/A | N/A | 0 | 3 | 3 |
| schema_high_cardinality_cols | 4 | 4 | 0 | N/A | N/A | 0 | 4 | 4 |
| first_numeric_monotone_ratio | 0.504 | 0.504 | 0 | N/A | N/A | 0 | 0.504 | 0.504 |
| first_numeric_diff_mean | 45778.78 | 45778.78 | 0 | N/A | N/A | 0 | 45778.78 | 45778.78 |
| first_numeric_diff_std | 1823843404 | 1823843404 | 0 | N/A | N/A | 0 | 1823843404 | 1823843404 |
| first_numeric_diff_q50 | 4738961 | 4738961 | 0 | N/A | N/A | 0 | 4738961 | 4738961 |
| first_numeric_diff_q90 | 2442810249 | 2442810249 | 0 | N/A | N/A | 0 | 2442810249 | 2442810249 |
| ttl_present | 0 | 0 | N/A | N/A | N/A | 0 | 0 | 0 |
| schema_mixed_cols | 0 | 0 | N/A | N/A | N/A | 0 | 0 | 0 |

## Outlier Files

| rel_path | outlier_score | top drivers |
|---|---:|---|
| N/A | N/A | N/A |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv | ts_duration |
|---|---|---:|---:|---:|---:|
| s3-cache-datasets/cache_dataset_txt/2016_wiki/wiki/2016/wiki.upload.2016.short | text | N/A | N/A | N/A | N/A |
| s3-cache-datasets/cache_dataset_txt/2016_wiki/wiki/2016/wiki.upload.2016.zst | text_zst | N/A | N/A | N/A | N/A |
