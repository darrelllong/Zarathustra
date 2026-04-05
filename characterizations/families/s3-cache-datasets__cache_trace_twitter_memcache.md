# s3-cache-datasets / cache_trace_twitter_memcache

- Files: 226
- Bytes: 4179512478708
- Formats: oracle_general, parquet, text_zst
- Parsers: generic_text, oracle_general, parquet_duckdb

## Observations

- Predominantly read-heavy.
- Highly bursty arrivals.

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| oracle_general | 162 | oracle_general |
| parquet | 10 | parquet_duckdb |
| text_zst | 54 | generic_text |

## Metrics

| Metric | Mean | Median | Min | Max | SD |
|---|---:|---:|---:|---:|---:|
| write_ratio | 0.001 | 0 | 0 | 0.022 | 0.003 |
| reuse_ratio | 0.104 | 0.033 | 0 | 0.914 | 0.168 |
| burstiness_cv | 23.015 | 16.492 | 1.447 | 63.984 | 19.941 |
| iat_q50 | 0 | 0 | 0 | 0 | 0 |
| iat_q90 | 0.056 | 0 | 0 | 2 | 0.255 |
| obj_size_q50 | 1058.377 | 122 | 10 | 28009.5 | 3816.054 |
| obj_size_q90 | 5639.886 | 504 | 22 | 109452 | 17029.92 |
| tenant_unique | 1.926 | 2 | 1 | 2 | 0.263 |
| opcode_switch_ratio | 0.001 | 0 | 0 | 0.041 | 0.005 |
| iat_lag1_autocorr | -0.021 | -0.004 | -0.269 | 0.034 | 0.04 |
| forward_seek_ratio | 0.448 | 0.483 | 0.043 | 0.513 | 0.084 |
| backward_seek_ratio | 0.448 | 0.483 | 0.043 | 0.51 | 0.084 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv |
|---|---|---:|---:|---:|
| s3-cache-datasets/cache_trace_twitter_memcache/oracleGeneral/cluster11.oracleGeneral.zst | oracle_general | 0 | 0.007 | 63.984 |
| s3-cache-datasets/cache_trace_twitter_memcache/oracleGeneral/cluster12.oracleGeneral.zst | oracle_general | 0 | 0 | 63.984 |
| s3-cache-datasets/cache_trace_twitter_memcache/oracleGeneral/cluster14.oracleGeneral.zst | oracle_general | 0 | 0.009 | 63.984 |
| s3-cache-datasets/cache_trace_twitter_memcache/oracleGeneral/cluster15.oracleGeneral.zst | oracle_general | 0 | 0 | 63.984 |
| s3-cache-datasets/cache_trace_twitter_memcache/oracleGeneral/cluster16.oracleGeneral.zst | oracle_general | 0 | 0.021 | 63.984 |
