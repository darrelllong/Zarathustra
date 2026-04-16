# s3-cache-datasets / 2023_metaStorage

- Files: 6
- Bytes: 2884049801
- Formats: text, text_zst
- Parsers: 2023_metaStorage_text
- ML Use Cases: structured_table
- Heterogeneity Score: 0
- Suggested GAN Modes: 1
- Split By Format: yes

## Observations

- No single dominant behavioral note stood out from the sampled features.

## GAN Guidance

- Family spans multiple encodings; keep format-aware preprocessing and avoid blindly pooling structured-table and request-sequence variants.
- Strongest feature coupling in this pass: first_numeric_diff_mean vs first_numeric_diff_std (corr=0.73).
- A small set of files are strong multivariate outliers; consider holding them out for ablation or separate mode inspection.

## Conditioning Audit

| Item | Value |
|---|---|
| Near-constant current conditioning features | none flagged |
| Recommended candidate additions | none flagged |
| Highly redundant current pairs | none flagged |

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| text | 1 | 2023_metaStorage_text |
| text_zst | 5 | 2023_metaStorage_text |

## Clustering And Regimes

| Item | Value |
|---|---|
| PCA variance explained by PC1 | 0.654 |

## Strongest Correlations

| Metric A | Metric B | Correlation |
|---|---|---:|
| first_numeric_diff_mean | first_numeric_diff_std | 0.725 |
| size_bytes | first_numeric_diff_std | 0.413 |
| size_bytes | first_numeric_diff_mean | 0.259 |

## Metrics

| Metric | Mean | Median | CV | Skew | Kurtosis | Missing | Q10 | Q90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| size_bytes | 480674967 | 270434506 | 1.095 | 2.447 | 5.99 | 0 | 252962058 | 918628336 |
| first_numeric_diff_std | 0.14 | 0.141 | 0.023 | -0.646 | -2.046 | 0 | 0.136 | 0.143 |
| first_numeric_diff_mean | 0.019 | 0.019 | 0.015 | 0.075 | -1.55 | 0 | 0.019 | 0.019 |
| sample_records | 4096 | 4096 | 0 | N/A | N/A | 0 | 4096 | 4096 |
| schema_column_count | 11 | 11 | 0 | N/A | N/A | 0 | 11 | 11 |
| schema_numeric_cols | 9 | 9 | 0 | N/A | N/A | 0 | 9 | 9 |
| schema_high_cardinality_cols | 4 | 4 | 0 | N/A | N/A | 0 | 4 | 4 |
| first_numeric_monotone_ratio | 1 | 1 | 0 | N/A | N/A | 0 | 1 | 1 |
| ttl_present | 0 | 0 | N/A | N/A | N/A | 0 | 0 | 0 |
| schema_mixed_cols | 0 | 0 | N/A | N/A | N/A | 0 | 0 | 0 |
| first_numeric_diff_q50 | 0 | 0 | N/A | N/A | N/A | 0 | 0 | 0 |
| first_numeric_diff_q90 | 0 | 0 | N/A | N/A | N/A | 0 | 0 | 0 |

## Outlier Files

| rel_path | outlier_score | top drivers |
|---|---:|---|
| s3-cache-datasets/cache_dataset_txt/2023_metaStorage/storage_202312/block_traces_1.csv | 4.166 | size_bytes (z=95.996); first_numeric_diff_std (z=1) |
| s3-cache-datasets/cache_dataset_txt/2023_metaStorage/storage_202312/block_traces_3.csv.zst | 3.796 | first_numeric_diff_mean (z=1.5); first_numeric_diff_std (z=-0.521) |
| s3-cache-datasets/cache_dataset_txt/2023_metaStorage/storage_202312/block_traces_2.csv.zst | 2.202 | size_bytes (z=-1.113); first_numeric_diff_std (z=0.521) |
| s3-cache-datasets/cache_dataset_txt/2023_metaStorage/storage_202312/block_traces_5.csv.zst | 1.63 | first_numeric_diff_std (z=-3.029); first_numeric_diff_mean (z=-1.5) |
| s3-cache-datasets/cache_dataset_txt/2023_metaStorage/storage_202312/block_traces_4.csv.zst | 1.625 | first_numeric_diff_std (z=-3.029); first_numeric_diff_mean (z=-1.5) |
| s3-cache-datasets/cache_dataset_txt/2023_metaStorage/storage_202312/block_traces_1.csv.zst | 1.58 | size_bytes (z=-1.499); first_numeric_diff_std (z=1) |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv | ts_duration |
|---|---|---:|---:|---:|---:|
| s3-cache-datasets/cache_dataset_txt/2023_metaStorage/storage_202312/block_traces_1.csv | text | N/A | N/A | N/A | N/A |
| s3-cache-datasets/cache_dataset_txt/2023_metaStorage/storage_202312/block_traces_1.csv.zst | text_zst | N/A | N/A | N/A | N/A |
| s3-cache-datasets/cache_dataset_txt/2023_metaStorage/storage_202312/block_traces_2.csv.zst | text_zst | N/A | N/A | N/A | N/A |
| s3-cache-datasets/cache_dataset_txt/2023_metaStorage/storage_202312/block_traces_3.csv.zst | text_zst | N/A | N/A | N/A | N/A |
| s3-cache-datasets/cache_dataset_txt/2023_metaStorage/storage_202312/block_traces_4.csv.zst | text_zst | N/A | N/A | N/A | N/A |
| s3-cache-datasets/cache_dataset_txt/2023_metaStorage/storage_202312/block_traces_5.csv.zst | text_zst | N/A | N/A | N/A | N/A |




## Model-Aware Guidance

- Closest learned anchor: tencent_block (distance 2.68)
- Sampling: split-by-format-first
- Regime recipe: single
- Char-file conditioning: no
- PCF: not-primary
- Multi-scale critic: not-primary
- Mixed-type recovery: not-primary
- Retrieval memory: not-primary
- Why: formats/parsers are mixed
