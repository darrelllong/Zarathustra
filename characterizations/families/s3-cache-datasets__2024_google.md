# s3-cache-datasets / 2024_google

- Files: 192
- Bytes: 184455433356
- Formats: parquet, text_zst
- Parsers: generic_text, parquet_duckdb
- ML Use Cases: structured_table
- Heterogeneity Score: 0
- Suggested GAN Modes: 1
- Split By Format: yes

## Observations

- No single dominant behavioral note stood out from the sampled features.

## GAN Guidance

- Family spans multiple encodings; keep format-aware preprocessing and avoid blindly pooling structured-table and request-sequence variants.
- Strongest feature coupling in this pass: size_bytes vs schema_high_cardinality_cols (corr=0.03).
- A small set of files are strong multivariate outliers; consider holding them out for ablation or separate mode inspection.

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| parquet | 6 | parquet_duckdb |
| text_zst | 186 | generic_text |

## Clustering And Regimes

| Item | Value |
|---|---|
| DBSCAN clusters | 2 |
| DBSCAN noise fraction | 0.052 |
| PCA variance explained by PC1 | 0.515 |

## Strongest Correlations

| Metric A | Metric B | Correlation |
|---|---|---:|
| size_bytes | schema_high_cardinality_cols | 0.03 |

## Metrics

| Metric | Mean | Median | CV | Skew | Kurtosis | Missing | Q10 | Q90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| size_bytes | 960705382 | 363288338 | 3.807 | 5.832 | 33.44 | 0 | 210285056 | 402019745 |
| schema_high_cardinality_cols | 8.917 | 9 | 0.063 | -7.824 | 63.529 | 0 | 9 | 9 |
| sample_records | 4096 | 4096 | 0 | N/A | N/A | 0 | 4096 | 4096 |
| schema_column_count | 16 | 16 | 0 | N/A | N/A | 0 | 16 | 16 |
| schema_numeric_cols | 10 | 10 | 0 | N/A | N/A | 0 | 10 | 10 |
| ttl_present | 0 | 0 | N/A | N/A | N/A | 0 | 0 | 0 |
| schema_mixed_cols | 0 | 0 | N/A | N/A | N/A | 0 | 0 | 0 |

## Outlier Files

| rel_path | outlier_score |
|---|---:|
| s3-cache-datasets/cache_dataset_txt/2024_google/cluster2_16TB/20240129.sort.csv.zst | 76.11 |
| s3-cache-datasets/cache_dataset_txt/2024_google/cluster2_16TB/20240219.sort.csv.zst | 76.11 |
| s3-cache-datasets/cache_dataset_parquet/2024_google/cluster3_18TB.sort.parquet | 44.836 |
| s3-cache-datasets/cache_dataset_parquet/2024_google/cluster1_16TB.sort.parquet | 42.69 |
| s3-cache-datasets/cache_dataset_txt/2024_google/cluster3_18TB.sort.csv.zst | 37.545 |
| s3-cache-datasets/cache_dataset_txt/2024_google/cluster1_16TB.sort.csv.zst | 34.2 |
| s3-cache-datasets/cache_dataset_txt/2024_google/cluster2_16TB/20240213.sort.csv.zst | 26.785 |
| s3-cache-datasets/cache_dataset_parquet/2024_google/cluster2_16TB.sort.parquet | 14.392 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv | ts_duration |
|---|---|---:|---:|---:|---:|
| s3-cache-datasets/cache_dataset_parquet/2024_google/cluster1_16TB.sort.parquet | parquet | N/A | N/A | N/A | N/A |
| s3-cache-datasets/cache_dataset_parquet/2024_google/cluster1_16TB.sort.sample100.parquet | parquet | N/A | N/A | N/A | N/A |
| s3-cache-datasets/cache_dataset_parquet/2024_google/cluster2_16TB.sort.parquet | parquet | N/A | N/A | N/A | N/A |
| s3-cache-datasets/cache_dataset_parquet/2024_google/cluster2_16TB.sort.sample100.parquet | parquet | N/A | N/A | N/A | N/A |
| s3-cache-datasets/cache_dataset_parquet/2024_google/cluster3_18TB.sort.parquet | parquet | N/A | N/A | N/A | N/A |
| s3-cache-datasets/cache_dataset_parquet/2024_google/cluster3_18TB.sort.sample100.parquet | parquet | N/A | N/A | N/A | N/A |
| s3-cache-datasets/cache_dataset_txt/2024_google/cluster1_16TB.sort.csv.zst | text_zst | N/A | N/A | N/A | N/A |
| s3-cache-datasets/cache_dataset_txt/2024_google/cluster1_16TB/20240115.sort.csv.zst | text_zst | N/A | N/A | N/A | N/A |
