# s3-cache-datasets / 2020_tencentBlock

- Files: 4996
- Bytes: 300469429369
- Formats: oracle_general, text_zst
- Parsers: generic_text, oracle_general

## Observations

- Predominantly read-heavy.
- Very weak short-window reuse.

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| oracle_general | 4995 | oracle_general |
| text_zst | 1 | generic_text |

## Metrics

| Metric | Mean | Median | Min | Max | SD |
|---|---:|---:|---:|---:|---:|
| write_ratio | 0 | 0 | 0 | 0 | 0 |
| reuse_ratio | 0.039 | 0.01 | 0 | 1 | 0.09 |
| burstiness_cv | 4.144 | 2.958 | 0 | 63.963 | 4.199 |
| iat_q50 | 181.187 | 0 | 0 | 258597 | 5088.752 |
| iat_q90 | 668.777 | 1 | 0 | 337312.2 | 8741.207 |
| obj_size_q50 | 13172.49 | 4096 | 512 | 524288 | 56210.1 |
| obj_size_q90 | 46053.73 | 16384 | 512 | 524288 | 103381.9 |
| tenant_unique | 1.984 | 2 | 1 | 2 | 0.124 |
| opcode_switch_ratio | 0 | 0 | 0 | 0 | 0 |
| iat_lag1_autocorr | -0.048 | -0.043 | -1 | 0.552 | 0.109 |
| forward_seek_ratio | 0.788 | 0.794 | 0 | 1 | 0.108 |
| backward_seek_ratio | 0.173 | 0.176 | 0 | 0.509 | 0.084 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv |
|---|---|---:|---:|---:|
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_tencentBlock/1M/tencentBlock_25959.oracleGeneral.zst | oracle_general | 0 | 0.171 | 63.963 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_tencentBlock/1M/tencentBlock_25878.oracleGeneral.zst | oracle_general | 0 | 0.063 | 63.177 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_tencentBlock/100K/tencentBlock_25820.oracleGeneral.zst | oracle_general | 0 | 0.174 | 56.519 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_tencentBlock/100K/tencentBlock_25815.oracleGeneral.zst | oracle_general | 0 | 0.15 | 54.844 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_tencentBlock/1M/tencentBlock_25788.oracleGeneral.zst | oracle_general | 0 | 0.041 | 42.804 |
