# s3-cache-datasets / 2019_wiki

- Files: 6
- Bytes: 79475295162
- Formats: oracle_general, text, text_zst
- Parsers: generic_text, oracle_general

## Observations

- Predominantly read-heavy.
- Very weak short-window reuse.
- Highly bursty arrivals.

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| oracle_general | 3 | oracle_general |
| text | 1 | generic_text |
| text_zst | 2 | generic_text |

## Metrics

| Metric | Mean | Median | Min | Max | SD |
|---|---:|---:|---:|---:|---:|
| write_ratio | 0 | 0 | 0 | 0 | 0 |
| reuse_ratio | 0.001 | 0.001 | 0 | 0.003 | 0.001 |
| burstiness_cv | 28.439 | 28.439 | 11.64 | 45.238 | 23.757 |
| iat_q50 | 0 | 0 | 0 | 0 | 0 |
| iat_q90 | 0 | 0 | 0 | 0 | 0 |
| obj_size_q50 | 12503.33 | 10272.5 | 4496 | 22741.5 | 9325.075 |
| obj_size_q90 | 54673.33 | 52224 | 29499 | 82297 | 26484.08 |
| tenant_unique | 2 | 2 | 2 | 2 | 0 |
| opcode_switch_ratio | 0 | 0 | 0 | 0 | 0 |
| iat_lag1_autocorr | -0.004 | -0.004 | -0.007 | 0 | 0.005 |
| forward_seek_ratio | 0.501 | 0.501 | 0.498 | 0.503 | 0.003 |
| backward_seek_ratio | 0.498 | 0.496 | 0.496 | 0.502 | 0.004 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv |
|---|---|---:|---:|---:|
| s3-cache-datasets/cache_dataset_oracleGeneral/2019_wiki/wiki_2019u.oracleGeneral.zst | oracle_general | 0 | 0 | 45.238 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2019_wiki/wiki_2019t.oracleGeneral.zst | oracle_general | 0 | 0.001 | 11.64 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2019_wiki/wiki_2016u.oracleGeneral.zst | oracle_general | 0 | 0.003 | N/A |
| s3-cache-datasets/cache_dataset_txt/2019_wiki/wiki/2019/wiki.upload.2019.short | text | N/A | N/A | N/A |
| s3-cache-datasets/cache_dataset_txt/2019_wiki/wiki/2019/wiki.txt.2019.zst | text_zst | N/A | N/A | N/A |
