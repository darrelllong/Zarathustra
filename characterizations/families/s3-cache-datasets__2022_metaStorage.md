# s3-cache-datasets / 2022_metaStorage

- Files: 5
- Bytes: 616636804
- Formats: oracle_general
- Parsers: oracle_general

## Observations

- Predominantly read-heavy.

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| oracle_general | 5 | oracle_general |

## Metrics

| Metric | Mean | Median | Min | Max | SD |
|---|---:|---:|---:|---:|---:|
| write_ratio | 0 | 0 | 0 | 0 | 0 |
| reuse_ratio | 0.243 | 0.242 | 0.225 | 0.263 | 0.014 |
| burstiness_cv | 7.335 | 7.272 | 7.221 | 7.505 | 0.134 |
| iat_q50 | 0 | 0 | 0 | 0 | 0 |
| iat_q90 | 0 | 0 | 0 | 0 | 0 |
| obj_size_q50 | 2107802 | 2109440 | 2097152 | 2117632 | 7440.754 |
| obj_size_q90 | 8388608 | 8388608 | 8388608 | 8388608 | 0 |
| tenant_unique | 2 | 2 | 2 | 2 | 0 |
| opcode_switch_ratio | 0 | 0 | 0 | 0 | 0 |
| iat_lag1_autocorr | -0.006 | -0.007 | -0.019 | 0.019 | 0.015 |
| forward_seek_ratio | 0.382 | 0.382 | 0.373 | 0.388 | 0.006 |
| backward_seek_ratio | 0.375 | 0.373 | 0.364 | 0.388 | 0.008 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv |
|---|---|---:|---:|---:|
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaStorage/block_traces_2.oracleGeneral.bin.zst | oracle_general | 0 | 0.242 | 7.505 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaStorage/block_traces_1.oracleGeneral.bin.zst | oracle_general | 0 | 0.245 | 7.452 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaStorage/block_traces_5.oracleGeneral.bin.zst | oracle_general | 0 | 0.242 | 7.272 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaStorage/block_traces_4.oracleGeneral.bin.zst | oracle_general | 0 | 0.225 | 7.224 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaStorage/block_traces_3.oracleGeneral.bin.zst | oracle_general | 0 | 0.263 | 7.221 |
