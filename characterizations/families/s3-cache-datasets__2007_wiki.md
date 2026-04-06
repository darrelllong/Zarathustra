# s3-cache-datasets / 2007_wiki

- Files: 4
- Bytes: 39128779019
- Formats: text_zst
- Parsers: wiki_hash_text
- ML Use Cases: request_sequence
- Heterogeneity Score: 0.286
- Suggested GAN Modes: 1
- Split By Format: no

## Observations

- Very weak short-window reuse.
- Highly bursty arrivals.

## GAN Guidance

- Burstiness is high; inter-arrival and FFT/ACF losses should stay heavily weighted.
- Strongest feature coupling in this pass: ts_duration vs iat_zero_ratio (corr=-1).
- A small set of files are strong multivariate outliers; consider holding them out for ablation or separate mode inspection.
- Current characterization suggests extra conditioning value from: object_unique, signed_stride_lag1_autocorr.

## Conditioning Audit

| Item | Value |
|---|---|
| Near-constant current conditioning features | iat_q50 |
| Recommended candidate additions | object_unique, signed_stride_lag1_autocorr |
| Highly redundant current pairs | none flagged |

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| text_zst | 4 | wiki_hash_text |

## Clustering And Regimes

| Item | Value |
|---|---|
| PCA variance explained by PC1 | 0.774 |

## Strongest Correlations

| Metric A | Metric B | Correlation |
|---|---|---:|
| ts_duration | iat_zero_ratio | -1 |
| ts_duration | iat_mean | 1 |
| iat_zero_ratio | iat_mean | -1 |
| iat_q99 | schema_high_cardinality_cols | 1 |
| ts_duration | iat_lag1_autocorr | -1 |
| iat_zero_ratio | iat_lag1_autocorr | 1 |
| iat_lag1_autocorr | iat_mean | -1 |
| iat_lag1_autocorr | iat_q99 | -0.997 |

## Metrics

| Metric | Mean | Median | CV | Skew | Kurtosis | Missing | Q10 | Q90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| iat_q99 | 0.25 | 0 | 2 | 2 | 4 | 0 | 0 | 0.7 |
| size_bytes | 9782194755 | 1303873646 | 1.822 | 1.98 | 3.928 | 0 | 111301084 | 26235745313 |
| iat_lag1_autocorr | -0.011 | -0.002 | 1.771 | -1.966 | 3.879 | 0 | -0.029 | 0 |
| ts_duration | 44 | 7.5 | 1.763 | 1.963 | 3.869 | 0 | 1 | 116.2 |
| iat_mean | 0.011 | 0.002 | 1.763 | 1.963 | 3.869 | 0 | 0 | 0.028 |
| iat_std | 0.071 | 0.037 | 1.191 | 1.685 | 2.73 | 0 | 0.016 | 0.153 |
| sample_record_rate | 2127.543 | 2194.286 | 1.07 | -0.012 | -5.931 | 0 | 105.691 | 4096 |
| burstiness_cv | 37.5 | 40.529 | 0.826 | -0.131 | -5.25 | 0 | 8.593 | 63.984 |
| schema_high_cardinality_cols | 1.25 | 1 | 0.4 | 2 | 4 | 0 | 1 | 1.7 |
| schema_numeric_cols | 1.75 | 2 | 0.286 | -2 | 4 | 0 | 1.3 | 2 |
| iat_zero_ratio | 0.989 | 0.998 | 0.019 | -1.963 | 3.869 | 0 | 0.972 | 1 |
| sample_records | 4096 | 4096 | 0 | N/A | N/A | 0 | 4096 | 4096 |
| schema_column_count | 2 | 2 | 0 | N/A | N/A | 0 | 2 | 2 |
| iat_min | 0 | 0 | N/A | N/A | N/A | 0 | 0 | 0 |
| iat_q50 | 0 | 0 | N/A | N/A | N/A | 0 | 0 | 0 |
| iat_q90 | 0 | 0 | N/A | N/A | N/A | 0 | 0 | 0 |
| ttl_present | 0 | 0 | N/A | N/A | N/A | 0 | 0 | 0 |
| schema_mixed_cols | 0 | 0 | N/A | N/A | N/A | 0 | 0 | 0 |

## Outlier Files

| rel_path | outlier_score | top drivers |
|---|---:|---|
| s3-cache-datasets/cache_dataset_txt/2007_wiki/wiki/2007/wiki.2007.sort.hash.sample100.csv.zst | 2.25 | iat_lag1_autocorr (z=-24.369); iat_zero_ratio (z=-23.462) |
| s3-cache-datasets/cache_dataset_txt/2007_wiki/wiki/2007/wiki.2007.sort.hash.sample10.csv.zst | 2.25 | sample_record_rate (z=-1); ts_duration (z=1) |
| s3-cache-datasets/cache_dataset_txt/2007_wiki/wiki/2007/wiki.2007.sort.raw.csv.zst | 2.25 | schema_numeric_cols (z=-2); size_bytes (z=-1.08) |
| s3-cache-datasets/cache_dataset_txt/2007_wiki/wiki/2007/wiki.2007.sort.hash.csv.zst | 2.25 | size_bytes (z=30.432); ts_duration (z=-1) |

## Outlier Sensitivity

| N Removed | Metric | Baseline Median | Trimmed Median | Relative Shift |
|---:|---|---:|---:|---:|
| 3 | reuse_ratio | 0.027 | 0.002 | -0.908 |
| 1 | burstiness_cv | 40.529 | 63.984 | 0.579 |
| 3 | burstiness_cv | 40.529 | 63.984 | 0.579 |
| 1 | reuse_ratio | 0.027 | 0.015 | -0.454 |
| 3 | object_unique | 1738 | 2397 | 0.379 |
| 1 | object_unique | 1738 | 2067.5 | 0.19 |
| 3 | abs_stride_mean | 5772565070961766400 | 6344717976505384960 | 0.099 |
| 1 | abs_stride_mean | 5772565070961766400 | 6058641523733575680 | 0.05 |
| 1 | iat_q50 | 0 | 0 | 0 |
| 3 | iat_q50 | 0 | 0 | 0 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv | ts_duration |
|---|---|---:|---:|---:|---:|
| s3-cache-datasets/cache_dataset_txt/2007_wiki/wiki/2007/wiki.2007.sort.hash.csv.zst | text_zst | N/A | 0.002 | 63.984 | 1 |
| s3-cache-datasets/cache_dataset_txt/2007_wiki/wiki/2007/wiki.2007.sort.raw.csv.zst | text_zst | N/A | N/A | 63.984 | 1 |
| s3-cache-datasets/cache_dataset_txt/2007_wiki/wiki/2007/wiki.2007.sort.hash.sample10.csv.zst | text_zst | N/A | 0.027 | 17.073 | 14 |
| s3-cache-datasets/cache_dataset_txt/2007_wiki/wiki/2007/wiki.2007.sort.hash.sample100.csv.zst | text_zst | N/A | 0.029 | 4.959 | 160 |
