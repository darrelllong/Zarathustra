# s3-cache-datasets / 2017_systor

- Files: 6
- Bytes: 34993572898
- Formats: text_zst
- Parsers: systor_text

## Observations

- Very weak short-window reuse.

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| text_zst | 6 | systor_text |

## Metrics

| Metric | Mean | Median | Min | Max | SD |
|---|---:|---:|---:|---:|---:|
| write_ratio | 0.469 | 0.479 | 0.181 | 0.836 | 0.265 |
| reuse_ratio | 0.002 | 0.001 | 0 | 0.006 | 0.002 |
| burstiness_cv | 11.68 | 2.943 | 1.307 | 57.931 | 22.678 |
| iat_q50 | 0.001 | 0.001 | 0 | 0.001 | 0 |
| iat_q90 | 0.014 | 0.014 | 0.006 | 0.022 | 0.007 |
| obj_size_q50 | 5461.333 | 4096 | 4096 | 8192 | 2115.165 |
| obj_size_q90 | 60074.67 | 32768 | 16384 | 131072 | 55480.27 |
| tenant_unique | 1 | 1 | 1 | 1 | 0 |
| opcode_switch_ratio | 0.118 | 0.125 | 0.075 | 0.164 | 0.035 |
| iat_lag1_autocorr | 0.075 | 0.089 | 0 | 0.137 | 0.052 |
| forward_seek_ratio | 0.598 | 0.602 | 0.544 | 0.657 | 0.045 |
| backward_seek_ratio | 0.4 | 0.398 | 0.339 | 0.455 | 0.047 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv |
|---|---|---:|---:|---:|
| s3-cache-datasets/cache_dataset_txt/2017_systor/2016_LUN3.csv.sort.zst | text_zst | 0.836 | 0.004 | 57.931 |
| s3-cache-datasets/cache_dataset_txt/2017_systor/2016_LUN6.csv.sort.zst | text_zst | 0.6 | 0.001 | 3.593 |
| s3-cache-datasets/cache_dataset_txt/2017_systor/2016_LUN4.csv.sort.zst | text_zst | 0.359 | 0.006 | 3.185 |
| s3-cache-datasets/cache_dataset_txt/2017_systor/2016_LUN1.csv.sort.zst | text_zst | 0.642 | 0 | 2.702 |
| s3-cache-datasets/cache_dataset_txt/2017_systor/2016_LUN0.csv.sort.zst | text_zst | 0.198 | 0.001 | 1.361 |
