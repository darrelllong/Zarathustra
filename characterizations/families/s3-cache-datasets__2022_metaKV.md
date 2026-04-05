# s3-cache-datasets / 2022_metaKV

- Files: 28
- Bytes: 150823760418
- Formats: oracle_general, text_zst
- Parsers: generic_text, oracle_general

## Observations

- Predominantly read-heavy.
- High temporal locality / reuse.
- Highly bursty arrivals.

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| oracle_general | 5 | oracle_general |
| text_zst | 23 | generic_text |

## Metrics

| Metric | Mean | Median | Min | Max | SD |
|---|---:|---:|---:|---:|---:|
| write_ratio | 0 | 0 | 0 | 0.001 | 0 |
| reuse_ratio | 0.895 | 0.894 | 0.844 | 0.931 | 0.033 |
| burstiness_cv | 35.168 | 21.307 | 20.211 | 63.984 | 24.962 |
| iat_q50 | 0 | 0 | 0 | 0 | 0 |
| iat_q90 | 0 | 0 | 0 | 0 | 0 |
| obj_size_q50 | 144.8 | 196 | 17 | 226 | 99.377 |
| obj_size_q90 | 1859.8 | 994 | 651 | 3479 | 1483.986 |
| tenant_unique | 2 | 2 | 2 | 2 | 0 |
| opcode_switch_ratio | 0 | 0 | 0 | 0.001 | 0.001 |
| iat_lag1_autocorr | -0.002 | -0.002 | -0.002 | 0 | 0.001 |
| forward_seek_ratio | 0.063 | 0.056 | 0.035 | 0.084 | 0.021 |
| backward_seek_ratio | 0.042 | 0.05 | 0.002 | 0.074 | 0.026 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv |
|---|---|---:|---:|---:|
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaKV/meta_kvcache_traces_1.oracleGeneral.bin.zst | oracle_general | 0 | 0.894 | 63.984 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaKV/202210_kv_traces_all_sort.csv.oracleGeneral.zst | oracle_general | 0 | 0.914 | 21.307 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaKV/202401_kv_traces_all_sort.csv.oracleGeneral.zst | oracle_general | 0 | 0.931 | 20.211 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaKV/202206_kv_traces_all.csv.oracleGeneral.zst | oracle_general | 0 | 0.894 | N/A |
| s3-cache-datasets/cache_dataset_oracleGeneral/2022_metaKV/202312_kv_traces_all.csv.oracleGeneral.zst | oracle_general | 0.001 | 0.844 | N/A |
