# s3-cache-datasets / other

- Files: 2
- Bytes: 386344048303
- Formats: oracle_general
- Parsers: oracle_general

## Observations

- Predominantly read-heavy.
- Very weak short-window reuse.
- Highly bursty arrivals.

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| oracle_general | 2 | oracle_general |

## Metrics

| Metric | Mean | Median | Min | Max | SD |
|---|---:|---:|---:|---:|---:|
| write_ratio | 0.016 | 0.016 | 0.015 | 0.017 | 0.001 |
| reuse_ratio | 0.025 | 0.025 | 0 | 0.051 | 0.036 |
| burstiness_cv | 63.984 | 63.984 | 63.984 | 63.984 | 0 |
| iat_q50 | 0 | 0 | 0 | 0 | 0 |
| iat_q90 | 0 | 0 | 0 | 0 | 0 |
| obj_size_q50 | 6144 | 6144 | 4096 | 8192 | 2896.309 |
| obj_size_q90 | 61440 | 61440 | 57344 | 65536 | 5792.619 |
| tenant_unique | 2 | 2 | 2 | 2 | 0 |
| opcode_switch_ratio | 0.03 | 0.03 | 0.025 | 0.034 | 0.007 |
| iat_lag1_autocorr | 0 | 0 | 0 | 0 | 0 |
| forward_seek_ratio | 0.675 | 0.675 | 0.601 | 0.749 | 0.105 |
| backward_seek_ratio | 0.299 | 0.299 | 0.2 | 0.399 | 0.141 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv |
|---|---|---:|---:|---:|
| s3-cache-datasets/cache_dataset_oracleGeneral/other/tencent_block.oracleGeneral.zst | oracle_general | 0.015 | 0.051 | 63.984 |
| s3-cache-datasets/cache_dataset_oracleGeneral/other/alibaba_block2020.oracleGeneral.zst | oracle_general | 0.017 | 0 | N/A |
