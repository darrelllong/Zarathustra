# s3-cache-datasets / 2015_cloudphysics

- Files: 106
- Bytes: 9026863149
- Formats: oracle_general
- Parsers: oracle_general

## Observations

- Predominantly read-heavy.
- Very weak short-window reuse.

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| oracle_general | 106 | oracle_general |

## Metrics

| Metric | Mean | Median | Min | Max | SD |
|---|---:|---:|---:|---:|---:|
| write_ratio | 0 | 0 | 0 | 0 | 0 |
| reuse_ratio | 0.007 | 0.001 | 0 | 0.148 | 0.019 |
| burstiness_cv | 8.758 | 5.393 | 1.884 | 63.984 | 8.947 |
| iat_q50 | 0 | 0 | 0 | 0 | 0 |
| iat_q90 | 0.453 | 0 | 0 | 5 | 1.015 |
| obj_size_q50 | 25353.66 | 4096 | 512 | 966656 | 108409 |
| obj_size_q90 | 61454.49 | 16384 | 512 | 1048576 | 127363 |
| tenant_unique | 1.962 | 2 | 1 | 2 | 0.191 |
| opcode_switch_ratio | 0 | 0 | 0 | 0 | 0 |
| iat_lag1_autocorr | 0.06 | 0.013 | -0.108 | 0.99 | 0.139 |
| forward_seek_ratio | 0.652 | 0.627 | 0.012 | 0.95 | 0.154 |
| backward_seek_ratio | 0.341 | 0.361 | 0.05 | 0.988 | 0.154 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv |
|---|---|---:|---:|---:|
| s3-cache-datasets/cache_dataset_oracleGeneral/2015_cloudphysics/w30.oracleGeneral.bin.zst | oracle_general | 0 | 0 | 63.984 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2015_cloudphysics/w46.oracleGeneral.bin.zst | oracle_general | 0 | 0.06 | 43.144 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2015_cloudphysics/w15.oracleGeneral.bin.zst | oracle_general | 0 | 0 | 34.096 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2015_cloudphysics/w21.oracleGeneral.bin.zst | oracle_general | 0 | 0 | 31.98 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2015_cloudphysics/w19.oracleGeneral.bin.zst | oracle_general | 0 | 0 | 26.537 |
