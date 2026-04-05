# s3-cache-datasets / tencentBlock

- Files: 4995
- Bytes: 817117209599
- Formats: lcs
- Parsers: lcs

## Observations

- Substantial write pressure across sampled files.
- Very weak short-window reuse.

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| lcs | 4995 | lcs |

## Metrics

| Metric | Mean | Median | Min | Max | SD |
|---|---:|---:|---:|---:|---:|
| write_ratio | 0.82 | 0.98 | 0 | 1 | 0.282 |
| reuse_ratio | 0.02 | 0.004 | 0 | 0.766 | 0.052 |
| burstiness_cv | 8.305 | 4.514 | 0.731 | 63.984 | 11.032 |
| iat_q50 | 30.39 | 0 | 0 | 86410 | 1390.401 |
| iat_q90 | 498.913 | 0 | 0 | 160203 | 5978.513 |
| obj_size_q50 | 4096 | 4096 | 4096 | 4096 | 0 |
| obj_size_q90 | 4096 | 4096 | 4096 | 4096 | 0 |
| tenant_unique | 1 | 1 | 1 | 1 | 0 |
| opcode_switch_ratio | 0.012 | 0.002 | 0 | 0.5 | 0.023 |
| iat_lag1_autocorr | 0.001 | -0.005 | -1 | 0.651 | 0.08 |
| forward_seek_ratio | 0.893 | 0.909 | 0.204 | 1 | 0.084 |
| backward_seek_ratio | 0.087 | 0.078 | 0 | 0.5 | 0.06 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv |
|---|---|---:|---:|---:|
| s3-cache-datasets/cache_dataset_lcs/tencentBlock/10370.lcs.zst | lcs | 1 | 0.001 | 63.984 |
| s3-cache-datasets/cache_dataset_lcs/tencentBlock/1063.lcs.zst | lcs | 0.581 | 0.001 | 63.984 |
| s3-cache-datasets/cache_dataset_lcs/tencentBlock/1108.lcs.zst | lcs | 0.293 | 0 | 63.984 |
| s3-cache-datasets/cache_dataset_lcs/tencentBlock/1121.lcs.zst | lcs | 0.993 | 0.001 | 63.984 |
| s3-cache-datasets/cache_dataset_lcs/tencentBlock/1125.lcs.zst | lcs | 0.229 | 0.006 | 63.984 |
