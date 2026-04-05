# s3-cache-datasets / 2022_metaCDN

- Files: 3
- Bytes: 2234960001
- Formats: oracle_general
- Parsers: oracle_general

## Observations

- Predominantly read-heavy.
- Highly bursty arrivals.

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| oracle_general | 3 | oracle_general |

## Metrics

| Metric | Mean | Median | Min | Max | SD |
|---|---:|---:|---:|---:|---:|
| write_ratio | 0 | 0 | 0 | 0 | 0 |
| reuse_ratio | 0.106 | 0.084 | 0.066 | 0.169 | 0.055 |
| burstiness_cv | 21.102 | 26.064 | 8.905 | 28.338 | 10.624 |
| iat_q50 | 0 | 0 | 0 | 0 | 0 |
| iat_q90 | 0 | 0 | 0 | 0 | 0 |
| obj_size_q50 | 54198.5 | 56574.5 | 35134 | 70887 | 17994.53 |
| obj_size_q90 | 20024696 | 7813858 | 6411012 | 45849219 | 22375689 |
| tenant_unique | 2 | 2 | 2 | 2 | 0 |
| opcode_switch_ratio | 0 | 0 | 0 | 0 | 0 |
| iat_lag1_autocorr | 0.328 | 0.381 | -0.013 | 0.617 | 0.318 |
| forward_seek_ratio | 0.447 | 0.453 | 0.414 | 0.476 | 0.032 |
| backward_seek_ratio | 0.446 | 0.458 | 0.418 | 0.463 | 0.025 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv |
|---|---|---:|---:|---:|
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaCDN/meta_rprn.oracleGeneral.zst | oracle_general | 0 | 0.066 | 28.338 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaCDN/meta_reag.oracleGeneral.zst | oracle_general | 0 | 0.169 | 26.064 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaCDN/meta_rnha.oracleGeneral.zst | oracle_general | 0 | 0.084 | 8.905 |
