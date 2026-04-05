# s3-cache-datasets / 2022_metaKV

- Files: 28
- Bytes: 150823760418
- Formats: oracle_general, text_zst
- Parsers: generic_text, oracle_general
- ML Use Cases: request_sequence, structured_table
- Heterogeneity Score: 0.656
- Suggested GAN Modes: 1
- Split By Format: yes

## Observations

- Predominantly read-heavy.
- High temporal locality / reuse.
- Highly bursty arrivals.

## GAN Guidance

- Family spans multiple encodings; keep format-aware preprocessing and avoid blindly pooling structured-table and request-sequence variants.
- Opcode balance is extremely read-skewed; generation should not assume symmetric read/write behavior.
- Reuse/locality is a major axis here; locality-aware losses and conditioning should matter.
- Burstiness is high; inter-arrival and FFT/ACF losses should stay heavily weighted.
- Strongest feature coupling in this pass: schema_column_count vs schema_numeric_cols (corr=0.99).
- A small set of files are strong multivariate outliers; consider holding them out for ablation or separate mode inspection.

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| oracle_general | 5 | oracle_general |
| text_zst | 23 | generic_text |

## Clustering And Regimes

| Item | Value |
|---|---|
| DBSCAN clusters | 1 |
| DBSCAN noise fraction | 0.044 |
| PCA variance explained by PC1 | 0.518 |

## Strongest Correlations

| Metric A | Metric B | Correlation |
|---|---|---:|
| schema_column_count | schema_numeric_cols | 0.988 |
| first_numeric_diff_std | first_numeric_diff_q90 | 0.936 |
| schema_numeric_cols | first_numeric_monotone_ratio | 0.918 |
| schema_column_count | first_numeric_monotone_ratio | 0.883 |
| schema_column_count | schema_high_cardinality_cols | 0.803 |
| first_numeric_monotone_ratio | first_numeric_diff_std | -0.748 |
| first_numeric_diff_mean | first_numeric_diff_q90 | -0.738 |
| first_numeric_monotone_ratio | first_numeric_diff_q90 | -0.715 |

## Metrics

| Metric | Mean | Median | CV | Skew | Kurtosis | Missing | Q10 | Q90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| size_bytes | 5386562872 | 1508857174 | 1.643 | 3.441 | 12.909 | 0 | 954362283 | 7542216423 |
| sample_records | 4096 | 4096 | 0 | N/A | N/A | 0 | 4096 | 4096 |
| ttl_present | 0 | 0 | N/A | N/A | N/A | 0 | 0 | 0 |
| first_numeric_diff_mean | -216.401 | 0.006 | 4.808 | -3.898 | 17.429 | 0.179 | -700.158 | 140.468 |
| first_numeric_diff_q50 | 0.522 | 0 | 4.796 | 4.796 | 23 | 0.179 | 0 | 0 |
| first_numeric_diff_q90 | 2223170 | 0 | 1.654 | 2.424 | 7.298 | 0.179 | 0 | 5760843 |
| first_numeric_diff_std | 1831022 | 0.079 | 1.58 | 1.5 | 1.038 | 0.179 | 0.072 | 6105965 |
| schema_high_cardinality_cols | 4.826 | 4 | 0.373 | 1.155 | -0.334 | 0.179 | 3.2 | 8 |
| schema_column_count | 7.304 | 8 | 0.272 | 0.222 | -1.534 | 0.179 | 5 | 10 |
| schema_numeric_cols | 6.043 | 7 | 0.271 | -0.076 | -1.764 | 0.179 | 4 | 8 |
| first_numeric_monotone_ratio | 0.815 | 1 | 0.245 | -0.167 | -2.098 | 0.179 | 0.596 | 1 |
| schema_mixed_cols | 0 | 0 | N/A | N/A | N/A | 0.179 | 0 | 0 |
| write_ratio | 0 | 0 | 2.236 | 2.236 | 5 | 0.821 | 0 | 0 |
| opcode_switch_ratio | 0 | 0 | 2.236 | 2.236 | 5 | 0.821 | 0 | 0.001 |
| abs_stride_q99 | 2306924135909836288 | 3308198 | 2.236 | 2.236 | 5 | 0.821 | 1463009 | 6920772407722315776 |
| abs_stride_mean | 84670744032755536 | 160497.9 | 2.236 | 2.236 | 5 | 0.821 | 73271.65 | 254012232097961184 |
| abs_stride_std | 378807968703555968 | 635677.8 | 2.236 | 2.236 | 5 | 0.821 | 366879.8 | 1136423906109274240 |
| abs_stride_q90 | 69.72 | 13.6 | 2.02 | 2.223 | 4.953 | 0.821 | 0 | 198.28 |

## Outlier Files

| rel_path | outlier_score |
|---|---:|
| s3-cache-datasets/cache_dataset_txt/2022_metaKV/kvcache_202312/flat_kvcache_04.csv.zst | 18.717 |
| s3-cache-datasets/cache_dataset_txt/2022_metaKV/202312_kv_traces_all.csv.zst | 12.2 |
| s3-cache-datasets/cache_dataset_txt/2022_metaKV/kvcache_202206/kvcache_traces_4.csv.zst | 9.01 |
| s3-cache-datasets/cache_dataset_txt/2022_metaKV/kvcache_202312/flat_kvcache_02.csv.zst | 2.588 |
| s3-cache-datasets/cache_dataset_txt/2022_metaKV/202401_kv_traces_all_sort.csv.zst | 2.345 |
| s3-cache-datasets/cache_dataset_txt/2022_metaKV/kvcache_202401/kvcache_traces_1.csv.zst | 1.95 |
| s3-cache-datasets/cache_dataset_txt/2022_metaKV/kvcache_202401/kvcache_traces_2.csv.zst | 1.95 |
| s3-cache-datasets/cache_dataset_txt/2022_metaKV/kvcache_202401/kvcache_traces_3.csv.zst | 1.95 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv | ts_duration |
|---|---|---:|---:|---:|---:|
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaKV/meta_kvcache_traces_1.oracleGeneral.bin.zst | oracle_general | 0 | 0.894 | 63.984 | 1 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaKV/202210_kv_traces_all_sort.csv.oracleGeneral.zst | oracle_general | 0 | 0.914 | 21.307 | 9 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaKV/202401_kv_traces_all_sort.csv.oracleGeneral.zst | oracle_general | 0 | 0.931 | 20.211 | 10 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaKV/202206_kv_traces_all.csv.oracleGeneral.zst | oracle_general | 0 | 0.894 | N/A | 0 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaKV/202312_kv_traces_all.csv.oracleGeneral.zst | oracle_general | 0.001 | 0.844 | N/A | 0 |
| s3-cache-datasets/cache_dataset_txt/2022_metaKV/202206_kv_traces_all.csv.zst | text_zst | N/A | N/A | N/A | N/A |
| s3-cache-datasets/cache_dataset_txt/2022_metaKV/202210_kv_traces_all_sort.csv.zst | text_zst | N/A | N/A | N/A | N/A |
| s3-cache-datasets/cache_dataset_txt/2022_metaKV/202312_kv_traces_all.csv.zst | text_zst | N/A | N/A | N/A | N/A |
