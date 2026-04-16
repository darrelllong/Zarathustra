# s3-cache-datasets / 2017_docker

- Files: 7
- Bytes: 600585203
- Formats: parquet
- Parsers: parquet_duckdb
- ML Use Cases: structured_table
- Heterogeneity Score: 0
- Suggested GAN Modes: 1
- Split By Format: no

## Observations

- No single dominant behavioral note stood out from the sampled features.

## GAN Guidance

- Strongest feature coupling in this pass: size_bytes vs schema_high_cardinality_cols (corr=0.93).
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
| parquet | 7 | parquet_duckdb |

## Clustering And Regimes

| Item | Value |
|---|---|
| PCA variance explained by PC1 | 0.722 |

## Strongest Correlations

| Metric A | Metric B | Correlation |
|---|---|---:|
| size_bytes | schema_high_cardinality_cols | 0.928 |
| size_bytes | schema_mixed_cols | -0.48 |
| schema_mixed_cols | schema_high_cardinality_cols | -0.258 |

## Metrics

| Metric | Mean | Median | CV | Skew | Kurtosis | Missing | Q10 | Q90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| schema_mixed_cols | 0.286 | 0 | 1.708 | 1.23 | -0.84 | 0 | 0 | 1 |
| size_bytes | 85797886 | 50373819 | 1.287 | 1.94 | 4.081 | 0 | 8647536 | 190931390 |
| schema_high_cardinality_cols | 4.143 | 4 | 0.091 | 2.646 | 7 | 0 | 4 | 4.4 |
| sample_records | 4096 | 4096 | 0 | N/A | N/A | 0 | 4096 | 4096 |
| schema_column_count | 9 | 9 | 0 | N/A | N/A | 0 | 9 | 9 |
| schema_numeric_cols | 4 | 4 | 0 | N/A | N/A | 0 | 4 | 4 |
| ttl_present | 0 | 0 | N/A | N/A | N/A | 0 | 0 | 0 |
| first_numeric_monotone_ratio | 1 | 1 | 0 | N/A | N/A | 0.714 | 1 | 1 |
| first_numeric_diff_mean | 0 | 0 | N/A | N/A | N/A | 0.714 | 0 | 0 |
| first_numeric_diff_std | 0 | 0 | N/A | N/A | N/A | 0.714 | 0 | 0 |
| first_numeric_diff_q50 | 0 | 0 | N/A | N/A | N/A | 0.714 | 0 | 0 |
| first_numeric_diff_q90 | 0 | 0 | N/A | N/A | N/A | 0.714 | 0 | 0 |

## Outlier Files

| rel_path | outlier_score | top drivers |
|---|---:|---|
| s3-cache-datasets/cache_dataset_parquet/2017_docker/dal09.parquet | 5.143 | size_bytes (z=5.972); schema_high_cardinality_cols (z=2.646) |
| s3-cache-datasets/cache_dataset_parquet/2017_docker/syd01.parquet | 3.705 | size_bytes (z=-0.832); sample_records (z=0) |
| s3-cache-datasets/cache_dataset_parquet/2017_docker/lon02.parquet | 2.318 | size_bytes (z=1.242); sample_records (z=0) |
| s3-cache-datasets/cache_dataset_parquet/2017_docker/dev-mon01.parquet | 2.15 | schema_mixed_cols (z=2.049); size_bytes (z=-1) |
| s3-cache-datasets/cache_dataset_parquet/2017_docker/prestage-mon01.parquet | 2.15 | schema_mixed_cols (z=2.049); size_bytes (z=-0.884) |
| s3-cache-datasets/cache_dataset_parquet/2017_docker/stage-dal09.parquet | 1.611 | size_bytes (z=1.03); sample_records (z=0) |
| s3-cache-datasets/cache_dataset_parquet/2017_docker/fra02.parquet | 0.922 | sample_records (z=0); size_bytes (z=0) |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv | ts_duration |
|---|---|---:|---:|---:|---:|
| s3-cache-datasets/cache_dataset_parquet/2017_docker/dal09.parquet | parquet | N/A | N/A | N/A | N/A |
| s3-cache-datasets/cache_dataset_parquet/2017_docker/dev-mon01.parquet | parquet | N/A | N/A | N/A | N/A |
| s3-cache-datasets/cache_dataset_parquet/2017_docker/fra02.parquet | parquet | N/A | N/A | N/A | N/A |
| s3-cache-datasets/cache_dataset_parquet/2017_docker/lon02.parquet | parquet | N/A | N/A | N/A | N/A |
| s3-cache-datasets/cache_dataset_parquet/2017_docker/prestage-mon01.parquet | parquet | N/A | N/A | N/A | N/A |
| s3-cache-datasets/cache_dataset_parquet/2017_docker/stage-dal09.parquet | parquet | N/A | N/A | N/A | N/A |
| s3-cache-datasets/cache_dataset_parquet/2017_docker/syd01.parquet | parquet | N/A | N/A | N/A | N/A |




## Model-Aware Guidance

- Closest learned anchor: tencent_block (distance 2.68)
- Sampling: random-ok
- Regime recipe: single
- Char-file conditioning: no
- PCF: not-primary
- Multi-scale critic: not-primary
- Mixed-type recovery: not-primary
- Retrieval memory: not-primary
- Why: no single pathological axis dominates this family
