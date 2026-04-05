# s3-cache-datasets / 2020_twitter

- Files: 54
- Bytes: 152248401229
- Formats: oracle_general
- Parsers: oracle_general

## Observations

- Predominantly read-heavy.
- Highly bursty arrivals.

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| oracle_general | 54 | oracle_general |

## Metrics

| Metric | Mean | Median | Min | Max | SD |
|---|---:|---:|---:|---:|---:|
| write_ratio | 0 | 0 | 0 | 0 | 0 |
| reuse_ratio | 0.099 | 0.037 | 0 | 0.792 | 0.163 |
| burstiness_cv | 20.019 | 18.446 | 4.835 | 45.238 | 9.785 |
| iat_q50 | 0 | 0 | 0 | 0 | 0 |
| iat_q90 | 0 | 0 | 0 | 0 | 0 |
| obj_size_q50 | 1213.102 | 124 | 19 | 25553 | 4259.361 |
| obj_size_q90 | 5572.667 | 506 | 22 | 109452 | 17834.83 |
| tenant_unique | 1.907 | 2 | 1 | 2 | 0.293 |
| opcode_switch_ratio | 0 | 0 | 0 | 0 | 0 |
| iat_lag1_autocorr | -0.006 | -0.003 | -0.043 | 0 | 0.01 |
| forward_seek_ratio | 0.45 | 0.483 | 0.104 | 0.508 | 0.082 |
| backward_seek_ratio | 0.451 | 0.479 | 0.104 | 0.51 | 0.082 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv |
|---|---|---:|---:|---:|
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_twitter/cluster18.oracleGeneral.sample10.zst | oracle_general | 0 | 0.07 | 45.238 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_twitter/cluster52.oracleGeneral.sample10.zst | oracle_general | 0 | 0.053 | 45.238 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_twitter/cluster1.oracleGeneral.sample10.zst | oracle_general | 0 | 0.043 | 36.932 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_twitter/cluster43.oracleGeneral.sample10.zst | oracle_general | 0 | 0 | 36.932 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_twitter/cluster25.oracleGeneral.sample10.zst | oracle_general | 0 | 0.425 | 31.98 |
