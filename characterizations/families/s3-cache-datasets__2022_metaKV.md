# s3-cache-datasets / 2022_metaKV

- Files: 28
- Bytes: 150823760418
- Formats: oracle_general, text_zst
- Parsers: generic_text, oracle_general
- ML Use Cases: request_sequence, structured_table
- Heterogeneity Score: 0.698
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
| oracle_general | 5 | oracle_general |
| text_zst | 23 | generic_text |

## Clustering And Regimes

| Item | Value |
|---|---|
| K-means selected K | 2 |
| Best silhouette K | 2 |
| PCA variance explained by PC1 | 0.518 |

### K Selection

| K | Within-SS | Silhouette |
|---:|---:|---:|
| 2 | 116914040134050611200 | 0.848 |
| 3 | 9855405250063163392 | 0.813 |
| 4 | 8973818133917656064 | 0.791 |
| 5 | 1523060352168586752 | 0.806 |
| 6 | 1509158103059552256 | 0.716 |
| 7 | 613253242332020480 | 0.678 |
| 8 | 256375850446051520 | 0.66 |
| 9 | 611159267163318272 | 0.675 |
| 10 | 255109242363983360 | 0.487 |
| 11 | 610595999650635136 | 0.549 |
| 12 | 1367777674699962 | 0.527 |

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
| size_bytes | 5386562872 | 1508857174 | 1.643 | N/A | N/A | 0 | 954362283 | 7542216423 |
| sample_records | 4096 | 4096 | 0 | N/A | N/A | 0 | 4096 | 4096 |
| ttl_present | 0 | 0 | N/A | N/A | N/A | 0 | 0 | 0 |
| first_numeric_diff_mean | -216.401 | 0.006 | 4.808 | N/A | N/A | 0.179 | -700.158 | 140.468 |
| first_numeric_diff_q50 | 0.522 | 0 | 4.796 | N/A | N/A | 0.179 | 0 | 0 |
| first_numeric_diff_q90 | 2223170 | 0 | 1.654 | N/A | N/A | 0.179 | 0 | 5760843 |
| first_numeric_diff_std | 1831022 | 0.079 | 1.58 | N/A | N/A | 0.179 | 0.072 | 6105965 |
| schema_high_cardinality_cols | 4.826 | 4 | 0.373 | N/A | N/A | 0.179 | 3.2 | 8 |
| schema_column_count | 7.304 | 8 | 0.272 | N/A | N/A | 0.179 | 5 | 10 |
| schema_numeric_cols | 6.043 | 7 | 0.271 | N/A | N/A | 0.179 | 4 | 8 |
| first_numeric_monotone_ratio | 0.815 | 1 | 0.245 | N/A | N/A | 0.179 | 0.596 | 1 |
| schema_mixed_cols | 0 | 0 | N/A | N/A | N/A | 0.179 | 0 | 0 |
| write_ratio | 0 | 0 | 2.236 | N/A | N/A | 0.821 | 0 | 0 |
| opcode_switch_ratio | 0 | 0 | 2.236 | N/A | N/A | 0.821 | 0 | 0.001 |
| ts_duration | 4 | 1 | 1.262 | N/A | N/A | 0.821 | 0 | 9.6 |
| iat_mean | 0.001 | 0 | 1.262 | N/A | N/A | 0.821 | 0 | 0.002 |
| iat_std | 0.022 | 0.016 | 1.089 | N/A | N/A | 0.821 | 0 | 0.048 |
| obj_size_q90 | 1859.8 | 994 | 0.798 | N/A | N/A | 0.821 | 669 | 3479 |

## Outlier Files

| rel_path | outlier_score | top drivers |
|---|---:|---|
| s3-cache-datasets/cache_dataset_txt/2022_metaKV/kvcache_202312/flat_kvcache_04.csv.zst | 18.717 | first_numeric_diff_std (z=974671070); first_numeric_diff_mean (z=-6423502) |
| s3-cache-datasets/cache_dataset_txt/2022_metaKV/202312_kv_traces_all.csv.zst | 12.2 | first_numeric_diff_std (z=628590028); first_numeric_diff_mean (z=461513) |
| s3-cache-datasets/cache_dataset_txt/2022_metaKV/kvcache_202206/kvcache_traces_4.csv.zst | 9.01 | first_numeric_diff_std (z=199118.3); first_numeric_diff_mean (z=10553.33) |
| s3-cache-datasets/cache_dataset_txt/2022_metaKV/kvcache_202312/flat_kvcache_02.csv.zst | 2.588 | first_numeric_diff_std (z=567731696); first_numeric_diff_mean (z=1509536) |
| s3-cache-datasets/cache_dataset_txt/2022_metaKV/202401_kv_traces_all_sort.csv.zst | 2.345 | size_bytes (z=10.247); schema_high_cardinality_cols (z=2.222) |
| s3-cache-datasets/cache_dataset_txt/2022_metaKV/kvcache_202401/kvcache_traces_1.csv.zst | 1.95 | schema_high_cardinality_cols (z=2.222); schema_column_count (z=1) |
| s3-cache-datasets/cache_dataset_txt/2022_metaKV/kvcache_202401/kvcache_traces_2.csv.zst | 1.95 | schema_high_cardinality_cols (z=2.222); schema_column_count (z=1) |
| s3-cache-datasets/cache_dataset_txt/2022_metaKV/kvcache_202401/kvcache_traces_3.csv.zst | 1.95 | schema_high_cardinality_cols (z=2.222); schema_column_count (z=1) |

## Outlier Sensitivity

| N Removed | Metric | Baseline Median | Trimmed Median | Relative Shift |
|---:|---|---:|---:|---:|
| 1 | burstiness_cv | 21.307 | 21.307 | 0 |
| 1 | abs_stride_mean | 160497.9 | 160497.9 | 0 |
| 1 | obj_size_std | 1731.789 | 1731.789 | 0 |
| 1 | reuse_ratio | 0.894 | 0.894 | 0 |
| 1 | iat_q50 | 0 | 0 | 0 |
| 1 | object_unique | 325 | 325 | 0 |
| 3 | burstiness_cv | 21.307 | 21.307 | 0 |
| 3 | abs_stride_mean | 160497.9 | 160497.9 | 0 |
| 3 | obj_size_std | 1731.789 | 1731.789 | 0 |
| 3 | reuse_ratio | 0.894 | 0.894 | 0 |

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
