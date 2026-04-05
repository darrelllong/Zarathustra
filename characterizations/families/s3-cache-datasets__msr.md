# s3-cache-datasets / msr

- Files: 36
- Bytes: 8130400717
- Formats: lcs
- Parsers: lcs

## Observations

- Substantial write pressure across sampled files.
- Very weak short-window reuse.

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| lcs | 36 | lcs |

## Metrics

| Metric | Mean | Median | Min | Max | SD |
|---|---:|---:|---:|---:|---:|
| write_ratio | 0.771 | 0.906 | 0.01 | 1 | 0.299 |
| reuse_ratio | 0.027 | 0 | 0 | 0.505 | 0.092 |
| burstiness_cv | 17.84 | 8.633 | 3.227 | 63.978 | 17.306 |
| iat_q50 | 0 | 0 | 0 | 0 | 0 |
| iat_q90 | 0.556 | 0 | 0 | 5 | 1.403 |
| obj_size_q50 | 4096 | 4096 | 4096 | 4096 | 0 |
| obj_size_q90 | 4096 | 4096 | 4096 | 4096 | 0 |
| tenant_unique | 1 | 1 | 1 | 1 | 0 |
| opcode_switch_ratio | 0.01 | 0.003 | 0 | 0.093 | 0.018 |
| iat_lag1_autocorr | 0.056 | 0 | -0.051 | 0.344 | 0.098 |
| forward_seek_ratio | 0.747 | 0.755 | 0.373 | 0.981 | 0.149 |
| backward_seek_ratio | 0.226 | 0.207 | 0.019 | 0.627 | 0.135 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv |
|---|---|---:|---:|---:|
| s3-cache-datasets/cache_dataset_lcs/msr/src2_2.csv.lcs.zst | lcs | 0.609 | 0 | 63.978 |
| s3-cache-datasets/cache_dataset_lcs/msr/rsrch_2.csv.lcs.zst | lcs | 1 | 0 | 51.145 |
| s3-cache-datasets/cache_dataset_lcs/msr/src2_1.csv.lcs.zst | lcs | 0.983 | 0 | 47.705 |
| s3-cache-datasets/cache_dataset_lcs/msr/hm_1.csv.lcs.zst | lcs | 0.249 | 0 | 47.589 |
| s3-cache-datasets/cache_dataset_lcs/msr/proj_4.csv.lcs.zst | lcs | 0.821 | 0 | 47.281 |
