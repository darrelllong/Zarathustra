# s3-cache-datasets / cache_trace_twitter_memcache

- Files: 226
- Bytes: 4179512478708
- Formats: oracle_general, parquet, text_zst
- Parsers: generic_text, oracle_general, parquet_duckdb
- ML Use Cases: request_sequence, structured_table
- Heterogeneity Score: 0.991
- Suggested GAN Modes: 6
- Split By Format: yes

## Observations

- Predominantly read-heavy.
- Highly bursty arrivals.
- Ordered feature trajectories show regime boundaries.

## GAN Guidance

- Family spans multiple encodings; keep format-aware preprocessing and avoid blindly pooling structured-table and request-sequence variants.
- Ordered PC1 changepoints suggest 6 regimes when files are ordered by trace start time.
- Sequential blocks are much more internally coherent than random file batches; block or curriculum sampling is likely safer than pure iid file sampling.
- Opcode balance is extremely read-skewed; generation should not assume symmetric read/write behavior.
- Burstiness is high; inter-arrival and FFT/ACF losses should stay heavily weighted.
- Strongest feature coupling in this pass: ts_duration vs iat_mean (corr=1).
- A small set of files are strong multivariate outliers; consider holding them out for ablation or separate mode inspection.
- Current characterization suggests extra conditioning value from: object_unique, signed_stride_lag1_autocorr, obj_size_std.

## Conditioning Audit

| Item | Value |
|---|---|
| Near-constant current conditioning features | write_ratio, iat_q50, opcode_switch_ratio, tenant_unique |
| Recommended candidate additions | object_unique, signed_stride_lag1_autocorr, obj_size_std |
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
| DBSCAN clusters | 1 |
| DBSCAN noise fraction | 0.046 |
| Ordered PC1 changepoints | 5 |
| PCA variance explained by PC1 | 0.301 |
| Hurst exponent on ordered PC1 | 0.754 |
| Block/random distance ratio | 0.796 |
| Sampling recommendation | block_sampling_preserves_temporal_coherence |

### K Selection

| K | Within-SS | Silhouette |
|---:|---:|---:|
| 2 | 362433149221556813023659370426247675904 | 0.75 |
| 3 | 230661324012885212000584779795498270720 | 0.654 |
| 4 | 159424726336036321898670417895101562880 | 0.544 |
| 5 | 129418390509840183833688668006091063296 | 0.481 |
| 6 | 98627952872992457686281104823114792960 | 0.549 |
| 7 | 94923206264952209068403969167630270464 | 0.477 |
| 8 | 68389440741088705290761486051114483712 | 0.464 |
| 9 | 86756845455010307626549531242154426368 | 0.253 |
| 10 | 77046349388830155182197475330628780032 | 0.286 |
| 11 | 73748398385013079408522047003912306688 | 0.273 |
| 12 | 51178017838513009708017277652348960768 | 0.267 |

## Regime Transition Drivers

| Transition | Driver 1 | Effect | Driver 2 | Effect | Driver 3 | Effect |
|---|---|---:|---|---:|---|---:|
| 1 -> 2 | size_bytes | 2.186 | iat_std | 2.026 | burstiness_cv | 1.841 |
| 2 -> 3 | tenant_top1_share | 1.111 | obj_size_q50 | 1.082 | backward_seek_ratio | 0.854 |
| 3 -> 4 | size_bytes | 1.517 | obj_size_mean | 1.418 | obj_size_q90 | 1.407 |
| 4 -> 5 | opcode_switch_ratio | 2.357 | write_ratio | 2.357 | obj_size_mean | 2.163 |
| 5 -> 6 | size_bytes | 3.237 | opcode_switch_ratio | 2.357 | write_ratio | 2.357 |

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
| size_bytes | 18493418047 | 4746918094 | 1.404 | 1.884 | 3.651 | 0 | 152719836 | 54095259983 |
| sample_records | 4096 | 4096 | 0 | N/A | N/A | 0 | 4096 | 4096 |
| ttl_present | 0 | 0 | N/A | N/A | N/A | 0 | 0 | 0 |
| iat_q90 | 0.056 | 0 | 4.597 | 4.993 | 27.055 | 0.283 | 0 | 0 |
| write_ratio | 0.001 | 0 | 4.364 | 6.837 | 51.231 | 0.283 | 0 | 0.001 |
| opcode_switch_ratio | 0.001 | 0 | 4.304 | 6.763 | 50.27 | 0.283 | 0 | 0.002 |
| obj_size_q50 | 1058.377 | 122 | 3.606 | 5.222 | 29.041 | 0.283 | 44 | 896.8 |
| obj_size_q90 | 5639.886 | 504 | 3.02 | 4.32 | 20.097 | 0.283 | 68 | 6232.4 |
| obj_size_mean | 2101.888 | 348.215 | 2.884 | 4.208 | 18.401 | 0.283 | 56.306 | 2568.437 |
| ts_duration | 105.136 | 13 | 2.727 | 5.345 | 32.886 | 0.283 | 1 | 276.3 |
| iat_mean | 0.026 | 0.003 | 2.727 | 5.345 | 32.886 | 0.283 | 0 | 0.067 |
| obj_size_std | 2856.522 | 235.567 | 2.407 | 3.649 | 14.413 | 0.283 | 15.84 | 8672.57 |
| obj_size_q99 | 11014.6 | 978.225 | 2.22 | 3.017 | 8.866 | 0.283 | 86.5 | 36085.4 |
| object_top1_share | 0.044 | 0.014 | 1.853 | 3.567 | 15.397 | 0.283 | 0.001 | 0.119 |
| reuse_ratio | 0.104 | 0.033 | 1.607 | 2.547 | 6.987 | 0.283 | 0 | 0.308 |
| iat_q99 | 0.383 | 0 | 1.594 | 2.191 | 8.073 | 0.283 | 0 | 1 |
| iat_std | 0.103 | 0.056 | 1.298 | 3.583 | 19.012 | 0.283 | 0.016 | 0.254 |
| object_top10_share | 0.131 | 0.07 | 1.26 | 2.251 | 6.398 | 0.283 | 0.005 | 0.367 |

## Outlier Files

| rel_path | outlier_score | top drivers |
|---|---:|---|
| s3-cache-datasets/cache_trace_twitter_memcache/oracleGeneral/sample100/cluster10.oracleGeneral.sample100.zst | 50.804 | iat_mean (z=193.5); ts_duration (z=193.5) |
| s3-cache-datasets/cache_trace_twitter_memcache/oracleGeneral/cluster50.oracleGeneral.zst | 31.97 | obj_size_q50 (z=371.833); obj_size_q90 (z=263.834) |
| s3-cache-datasets/cache_trace_twitter_memcache/oracleGeneral/sample100/cluster35.oracleGeneral.sample100.zst | 31.07 | abs_stride_q90 (z=-44.902); abs_stride_std (z=-33.139) |
| s3-cache-datasets/cache_trace_twitter_memcache/oracleGeneral/sample10/cluster50.oracleGeneral.sample10.zst | 30.589 | obj_size_q50 (z=339.08); obj_size_q90 (z=264.116) |
| s3-cache-datasets/cache_trace_twitter_memcache/oracleGeneral/sample100/cluster50.oracleGeneral.sample100.zst | 27.951 | obj_size_q90 (z=204.09); obj_size_std (z=158.084) |
| s3-cache-datasets/cache_trace_twitter_memcache/oracleGeneral/sample100/cluster26.oracleGeneral.sample100.zst | 17.622 | iat_mean (z=136.917); ts_duration (z=136.917) |
| s3-cache-datasets/cache_trace_twitter_memcache/oracleGeneral/sample100/cluster2.oracleGeneral.sample100.zst | 16.65 | obj_size_q90 (z=24.211); reuse_ratio (z=22.316) |
| s3-cache-datasets/cache_trace_twitter_memcache/oracleGeneral/sample10/cluster35.oracleGeneral.sample10.zst | 14.186 | abs_stride_q90 (z=-24.128); reuse_ratio (z=22.853) |

## Outlier Sensitivity

| N Removed | Metric | Baseline Median | Trimmed Median | Relative Shift |
|---:|---|---:|---:|---:|
| 10 | reuse_ratio | 0.033 | 0.028 | -0.169 |
| 10 | obj_size_std | 235.567 | 216.323 | -0.082 |
| 1 | reuse_ratio | 0.033 | 0.032 | -0.037 |
| 3 | reuse_ratio | 0.033 | 0.032 | -0.037 |
| 5 | reuse_ratio | 0.033 | 0.032 | -0.037 |
| 10 | burstiness_cv | 16.492 | 17.073 | 0.035 |
| 5 | obj_size_std | 235.567 | 228.209 | -0.031 |
| 1 | obj_size_std | 235.567 | 236.836 | 0.005 |
| 3 | obj_size_std | 235.567 | 234.299 | -0.005 |
| 10 | abs_stride_mean | 5930313889305265152 | 5944100828898565120 | 0.002 |

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
