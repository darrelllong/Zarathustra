# s3-cache-datasets / 2007_msr

- Files: 14
- Bytes: 1637389703
- Formats: oracle_general
- Parsers: oracle_general

## Observations

- Predominantly read-heavy.
- Very weak short-window reuse.

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| oracle_general | 14 | oracle_general |

## Metrics

| Metric | Mean | Median | Min | Max | SD |
|---|---:|---:|---:|---:|---:|
| write_ratio | 0 | 0 | 0 | 0 | 0 |
| reuse_ratio | 0.024 | 0.004 | 0 | 0.118 | 0.037 |
| burstiness_cv | 13.605 | 6.699 | 2.313 | 47.245 | 13.985 |
| iat_q50 | 0 | 0 | 0 | 0 | 0 |
| iat_q90 | 0.357 | 0 | 0 | 2 | 0.633 |
| obj_size_q50 | 8704 | 4096 | 512 | 65536 | 16521.27 |
| obj_size_q90 | 39789.71 | 43008 | 4096 | 65536 | 25271.34 |
| tenant_unique | 1.929 | 2 | 1 | 2 | 0.267 |
| opcode_switch_ratio | 0 | 0 | 0 | 0 | 0 |
| iat_lag1_autocorr | 0.068 | -0.001 | -0.065 | 0.269 | 0.112 |
| forward_seek_ratio | 0.535 | 0.495 | 0.46 | 0.795 | 0.098 |
| backward_seek_ratio | 0.441 | 0.484 | 0.2 | 0.513 | 0.094 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv |
|---|---|---:|---:|---:|
| s3-cache-datasets/cache_dataset_oracleGeneral/2007_msr/msr_proj_4.oracleGeneral.zst | oracle_general | 0 | 0.001 | 47.245 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2007_msr/msr_src1_0.oracleGeneral.zst | oracle_general | 0 | 0 | 38.625 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2007_msr/msr_prxy_1.oracleGeneral.zst | oracle_general | 0 | 0.075 | 21.307 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2007_msr/msr_web_2.oracleGeneral.zst | oracle_general | 0 | 0 | 17.825 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2007_msr/msr_proj_2.oracleGeneral.zst | oracle_general | 0 | 0 | 16.87 |
