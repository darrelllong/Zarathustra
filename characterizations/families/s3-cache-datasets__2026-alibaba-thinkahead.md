# s3-cache-datasets / 2026-alibaba-thinkahead

- Files: 45
- Bytes: 15366738380
- Formats: oracle_general
- Parsers: oracle_general

## Observations

- Predominantly read-heavy.
- Very weak short-window reuse.
- Highly bursty arrivals.

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| oracle_general | 45 | oracle_general |

## Metrics

| Metric | Mean | Median | Min | Max | SD |
|---|---:|---:|---:|---:|---:|
| write_ratio | 0 | 0 | 0 | 0 | 0 |
| reuse_ratio | 0.027 | 0.027 | 0 | 0.139 | 0.028 |
| burstiness_cv | 29.031 | 26.106 | 12.069 | 63.952 | 11.71 |
| iat_q50 | 0 | 0 | 0 | 0 | 0 |
| iat_q90 | 0 | 0 | 0 | 0 | 0 |
| obj_size_q50 | 1 | 1 | 1 | 1 | 0 |
| obj_size_q90 | 1 | 1 | 1 | 1 | 0 |
| tenant_unique | 1.978 | 2 | 1 | 2 | 0.149 |
| opcode_switch_ratio | 0 | 0 | 0 | 0 | 0 |
| iat_lag1_autocorr | 0.022 | -0.001 | -0.007 | 0.281 | 0.07 |
| forward_seek_ratio | 0.701 | 0.68 | 0.55 | 0.999 | 0.095 |
| backward_seek_ratio | 0.272 | 0.306 | 0.001 | 0.383 | 0.09 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv |
|---|---|---:|---:|---:|
| s3-cache-datasets/cache_dataset_oracleGeneral/2026-alibaba-thinkahead/c4.oracleGeneral.zst | oracle_general | 0 | 0.002 | 63.952 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2026-alibaba-thinkahead/c15.oracleGeneral.zst | oracle_general | 0 | 0 | 63.522 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2026-alibaba-thinkahead/c21.oracleGeneral.zst | oracle_general | 0 | 0.004 | 60.426 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2026-alibaba-thinkahead/c29.oracleGeneral.zst | oracle_general | 0 | 0.041 | 42.436 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2026-alibaba-thinkahead/c44.oracleGeneral.zst | oracle_general | 0 | 0.035 | 42.436 |
