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

## Conditioning Audit

| Item | Value |
|---|---|
| Near-constant current conditioning features | none flagged |
| Recommended candidate additions | none flagged |
| Highly redundant current pairs | none flagged |

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| parquet | 6 | parquet_duckdb |
| text_zst | 186 | generic_text |

## Clustering And Regimes

| Item | Value |
|---|---|
| K-means selected K | 2 |
| Best silhouette K | 2 |
| DBSCAN clusters | 2 |
| DBSCAN noise fraction | 0.052 |
| PCA variance explained by PC1 | 0.515 |

### K Selection

| K | Within-SS | Silhouette |
|---:|---:|---:|
| 2 | 138097277494976479232 | 0.985 |
| 3 | 137107353725581656064 | 0.814 |
| 4 | 137057435953505189888 | 0.662 |
| 5 | 137040024146018451456 | 0.583 |
| 6 | 137026875258400292864 | 0.542 |
| 7 | 6983958577416497152 | 0.615 |
| 8 | 137016736640469139456 | 0.543 |
| 9 | 6961647027463412736 | 0.548 |
| 10 | 6979172260604485632 | 0.613 |
| 11 | 137010393504188350464 | 0.541 |
| 12 | 137010716208710008832 | 0.546 |

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

| rel_path | outlier_score | top drivers |
|---|---:|---|
| s3-cache-datasets/cache_dataset_txt/2024_google/cluster2_16TB/20240129.sort.csv.zst | 76.11 | schema_high_cardinality_cols (z=-8.872); size_bytes (z=-3.967) |
| s3-cache-datasets/cache_dataset_txt/2024_google/cluster2_16TB/20240219.sort.csv.zst | 76.11 | schema_high_cardinality_cols (z=-8.872); size_bytes (z=-4.93) |
| s3-cache-datasets/cache_dataset_parquet/2024_google/cluster3_18TB.sort.parquet | 44.836 | size_bytes (z=746.137); sample_records (z=0) |
| s3-cache-datasets/cache_dataset_parquet/2024_google/cluster1_16TB.sort.parquet | 42.69 | size_bytes (z=728.5); sample_records (z=0) |
| s3-cache-datasets/cache_dataset_txt/2024_google/cluster3_18TB.sort.csv.zst | 37.545 | size_bytes (z=684.3); sample_records (z=0) |
| s3-cache-datasets/cache_dataset_txt/2024_google/cluster1_16TB.sort.csv.zst | 34.2 | size_bytes (z=653.92); sample_records (z=0) |
| s3-cache-datasets/cache_dataset_txt/2024_google/cluster2_16TB/20240213.sort.csv.zst | 26.785 | schema_high_cardinality_cols (z=-5.323); size_bytes (z=-3.504) |
| s3-cache-datasets/cache_dataset_parquet/2024_google/cluster2_16TB.sort.parquet | 14.392 | size_bytes (z=430.421); sample_records (z=0) |

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
