# s3-cache-datasets / 2018_tencentPhoto

- Files: 4
- Bytes: 163051085109
- Formats: oracle_general, text_zst
- Parsers: generic_text, oracle_general

## Observations

- Predominantly read-heavy.
- Very weak short-window reuse.
- Highly bursty arrivals.

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| oracle_general | 2 | oracle_general |
| text_zst | 2 | generic_text |

## Metrics

| Metric | Mean | Median | Min | Max | SD |
|---|---:|---:|---:|---:|---:|
| write_ratio | 0 | 0 | 0 | 0 | 0 |
| reuse_ratio | 0.009 | 0.009 | 0.008 | 0.009 | 0.001 |
| burstiness_cv | 28.601 | 28.601 | 28.601 | 28.601 | 0 |
| iat_q50 | 0 | 0 | 0 | 0 | 0 |
| iat_q90 | 0 | 0 | 0 | 0 | 0 |
| obj_size_q50 | 14442.5 | 14442.5 | 14204.5 | 14680.5 | 336.583 |
| obj_size_q90 | 62235.5 | 62235.5 | 62025.5 | 62445.5 | 296.985 |
| tenant_unique | 2 | 2 | 2 | 2 | 0 |
| opcode_switch_ratio | 0 | 0 | 0 | 0 | 0 |
| iat_lag1_autocorr | -0.001 | -0.001 | -0.001 | -0.001 | 0 |
| forward_seek_ratio | 0.493 | 0.493 | 0.493 | 0.494 | 0.001 |
| backward_seek_ratio | 0.498 | 0.498 | 0.498 | 0.498 | 0 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv |
|---|---|---:|---:|---:|
| s3-cache-datasets/cache_dataset_oracleGeneral/2018_tencentPhoto/tencent_photo1.oracleGeneral.zst | oracle_general | 0 | 0.008 | 28.601 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2018_tencentPhoto/tencent_photo2.oracleGeneral.zst | oracle_general | 0 | 0.009 | 28.601 |
| s3-cache-datasets/cache_dataset_txt/2018_tencentPhoto/tencentPhoto1.sort.zst | text_zst | N/A | N/A | N/A |
| s3-cache-datasets/cache_dataset_txt/2018_tencentPhoto/tencentPhoto2.sort.zst | text_zst | N/A | N/A | N/A |
