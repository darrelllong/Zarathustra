# s3-cache-datasets / cache_trace_twitter_memcache

- Files: 226
- Bytes: 4179512478708
- Formats: oracle_general, parquet, text_zst
- Parsers: generic_text, oracle_general, parquet_duckdb
- ML Use Cases: request_sequence, structured_table
- Heterogeneity Score: 1.351
- Suggested GAN Modes: 1
- Split By Format: yes

## Observations

- Predominantly read-heavy.
- Highly bursty arrivals.

## GAN Guidance

- Family spans multiple encodings; keep format-aware preprocessing and avoid blindly pooling structured-table and request-sequence variants.
- Opcode balance is extremely read-skewed; generation should not assume symmetric read/write behavior.
- Burstiness is high; inter-arrival and FFT/ACF losses should stay heavily weighted.
- Strongest feature coupling in this pass: ts_duration vs iat_mean (corr=1).
- A small set of files are strong multivariate outliers; consider holding them out for ablation or separate mode inspection.
- Current characterization suggests extra conditioning value from: object_unique, obj_size_std.

## Conditioning Audit

| Item | Value |
|---|---|
| Near-constant current conditioning features | write_ratio, iat_q50, opcode_switch_ratio, tenant_unique |
| Recommended candidate additions | object_unique, obj_size_std |
| Highly redundant current pairs | write_ratio vs opcode_switch_ratio (0.999); reuse_ratio vs backward_seek_ratio (-0.997); reuse_ratio vs forward_seek_ratio (-0.997); forward_seek_ratio vs backward_seek_ratio (0.989) |

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| oracle_general | 162 | oracle_general |
| parquet | 10 | parquet_duckdb |
| text_zst | 54 | generic_text |

## Clustering And Regimes

| Item | Value |
|---|---|
| K-means selected K | 2 |
| Best silhouette K | 2 |
| PCA variance explained by PC1 | 0.287 |
| Block/random distance ratio | 0.908 |
| Sampling recommendation | random_sampling_is_less_problematic |

### K Selection

| K | Within-SS | Silhouette |
|---:|---:|---:|
| 2 | 3192.002 | 0.505 |
| 3 | 2700.722 | 0.256 |
| 4 | 2210.975 | 0.259 |
| 5 | 1942.597 | 0.223 |
| 6 | 1702.57 | 0.232 |
| 7 | 1513.748 | 0.245 |
| 8 | 1386.511 | 0.222 |
| 9 | 1259.492 | 0.22 |
| 10 | 1157.744 | 0.225 |
| 11 | 1122.796 | 0.22 |
| 12 | 1037.949 | 0.231 |

## Strongest Correlations

| Metric A | Metric B | Correlation |
|---|---|---:|
| ts_duration | iat_mean | 1 |
| write_ratio | opcode_switch_ratio | 0.999 |
| reuse_ratio | backward_seek_ratio | -0.997 |
| reuse_ratio | forward_seek_ratio | -0.997 |
| obj_size_mean | obj_size_q90 | 0.991 |
| forward_seek_ratio | backward_seek_ratio | 0.989 |
| iat_zero_ratio | iat_mean | -0.976 |
| ts_duration | iat_zero_ratio | -0.976 |

## Metrics

| Metric | Mean | Median | CV | Skew | Kurtosis | Missing | Q10 | Q90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| size_bytes | 18493418047 | 4746918094 | 1.404 | N/A | N/A | 0 | 152719836 | 54095259983 |
| sample_records | 4096 | 4096 | 0 | N/A | N/A | 0 | 4096 | 4096 |
| ttl_present | 0 | 0 | N/A | N/A | N/A | 0 | 0 | 0 |
| iat_q90 | 0.056 | 0 | 4.597 | N/A | N/A | 0.283 | 0 | 0 |
| write_ratio | 0.001 | 0 | 4.364 | N/A | N/A | 0.283 | 0 | 0.001 |
| opcode_switch_ratio | 0.001 | 0 | 4.304 | N/A | N/A | 0.283 | 0 | 0.002 |
| obj_size_q50 | 1058.377 | 122 | 3.606 | N/A | N/A | 0.283 | 44 | 896.8 |
| obj_size_q90 | 5639.886 | 504 | 3.02 | N/A | N/A | 0.283 | 68 | 6232.4 |
| obj_size_mean | 2101.888 | 348.215 | 2.884 | N/A | N/A | 0.283 | 56.306 | 2568.437 |
| ts_duration | 105.136 | 13 | 2.727 | N/A | N/A | 0.283 | 1 | 276.3 |
| iat_mean | 0.026 | 0.003 | 2.727 | N/A | N/A | 0.283 | 0 | 0.067 |
| obj_size_std | 2856.522 | 235.567 | 2.407 | N/A | N/A | 0.283 | 15.84 | 8672.57 |
| obj_size_q99 | 11014.6 | 978.225 | 2.22 | N/A | N/A | 0.283 | 86.5 | 36085.4 |
| object_top1_share | 0.044 | 0.014 | 1.853 | N/A | N/A | 0.283 | 0.001 | 0.119 |
| reuse_ratio | 0.104 | 0.033 | 1.607 | N/A | N/A | 0.283 | 0 | 0.308 |
| iat_q99 | 0.383 | 0 | 1.594 | N/A | N/A | 0.283 | 0 | 1 |
| iat_std | 0.103 | 0.056 | 1.298 | N/A | N/A | 0.283 | 0.016 | 0.254 |
| object_top10_share | 0.131 | 0.07 | 1.26 | N/A | N/A | 0.283 | 0.005 | 0.367 |

## Outlier Files

| rel_path | outlier_score | top drivers |
|---|---:|---|
| s3-cache-datasets/cache_trace_twitter_memcache/oracleGeneral/sample100/cluster10.oracleGeneral.sample100.zst | 49.764 | ts_duration (z=100); iat_zero_ratio (z=-100) |
| s3-cache-datasets/cache_trace_twitter_memcache/oracleGeneral/cluster50.oracleGeneral.zst | 31.96 | obj_size_mean (z=100); obj_size_std (z=100) |
| s3-cache-datasets/cache_trace_twitter_memcache/oracleGeneral/sample10/cluster50.oracleGeneral.sample10.zst | 30.549 | obj_size_mean (z=100); obj_size_std (z=100) |
| s3-cache-datasets/cache_trace_twitter_memcache/oracleGeneral/sample100/cluster50.oracleGeneral.sample100.zst | 28.475 | ts_duration (z=100); iat_zero_ratio (z=-100) |
| s3-cache-datasets/cache_trace_twitter_memcache/oracleGeneral/sample100/cluster26.oracleGeneral.sample100.zst | 17.941 | ts_duration (z=100); iat_zero_ratio (z=-100) |
| s3-cache-datasets/cache_trace_twitter_memcache/oracleGeneral/sample10/cluster25.oracleGeneral.sample10.zst | 16.393 | object_top1_share (z=42.086); object_top10_share (z=13.802) |
| s3-cache-datasets/cache_trace_twitter_memcache/oracleGeneral/sample100/cluster45.oracleGeneral.sample100.zst | 11.527 | iat_mean (z=87.833); ts_duration (z=87.833) |
| s3-cache-datasets/cache_trace_twitter_memcache/oracleGeneral/sample10/cluster49.oracleGeneral.sample10.zst | 11.407 | obj_size_q50 (z=100); obj_size_q90 (z=100) |

## Outlier Sensitivity

| N Removed | Metric | Baseline Median | Trimmed Median | Relative Shift |
|---:|---|---:|---:|---:|
| 10 | reuse_ratio | 0.033 | 0.03 | -0.096 |
| 1 | reuse_ratio | 0.033 | 0.032 | -0.037 |
| 5 | reuse_ratio | 0.033 | 0.032 | -0.037 |
| 3 | reuse_ratio | 0.033 | 0.034 | 0.037 |
| 5 | obj_size_std | 235.567 | 228.209 | -0.031 |
| 3 | object_unique | 2049 | 1991 | -0.028 |
| 10 | obj_size_std | 235.567 | 231.254 | -0.018 |
| 1 | obj_size_std | 235.567 | 236.836 | 0.005 |
| 3 | obj_size_std | 235.567 | 234.299 | -0.005 |
| 10 | object_unique | 2049 | 2050.5 | 0.001 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv | ts_duration |
|---|---|---:|---:|---:|---:|
| s3-cache-datasets/cache_trace_twitter_memcache/oracleGeneral/cluster11.oracleGeneral.zst | oracle_general | 0 | 0.007 | 63.984 | 1 |
| s3-cache-datasets/cache_trace_twitter_memcache/oracleGeneral/cluster12.oracleGeneral.zst | oracle_general | 0 | 0 | 63.984 | 1 |
| s3-cache-datasets/cache_trace_twitter_memcache/oracleGeneral/cluster14.oracleGeneral.zst | oracle_general | 0 | 0.009 | 63.984 | 1 |
| s3-cache-datasets/cache_trace_twitter_memcache/oracleGeneral/cluster15.oracleGeneral.zst | oracle_general | 0 | 0 | 63.984 | 1 |
| s3-cache-datasets/cache_trace_twitter_memcache/oracleGeneral/cluster16.oracleGeneral.zst | oracle_general | 0 | 0.021 | 63.984 | 1 |
| s3-cache-datasets/cache_trace_twitter_memcache/oracleGeneral/cluster21.oracleGeneral.zst | oracle_general | 0 | 0 | 63.984 | 1 |
| s3-cache-datasets/cache_trace_twitter_memcache/oracleGeneral/cluster23.oracleGeneral.zst | oracle_general | 0 | 0.027 | 63.984 | 1 |
| s3-cache-datasets/cache_trace_twitter_memcache/oracleGeneral/cluster25.oracleGeneral.zst | oracle_general | 0 | 0.22 | 63.984 | 1 |
