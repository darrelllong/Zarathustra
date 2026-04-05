# s3-cache-datasets / alibaba

- Files: 1000
- Bytes: 477220451258
- Formats: lcs
- Parsers: lcs

## Observations

- Substantial write pressure across sampled files.
- Very weak short-window reuse.

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| lcs | 1000 | lcs |

## Metrics

| Metric | Mean | Median | Min | Max | SD |
|---|---:|---:|---:|---:|---:|
| write_ratio | 0.762 | 0.976 | 0 | 1 | 0.355 |
| reuse_ratio | 0.004 | 0.001 | 0 | 0.61 | 0.025 |
| burstiness_cv | 14.921 | 6.399 | 1.072 | 63.984 | 18.249 |
| iat_q50 | 0.012 | 0 | 0 | 6 | 0.249 |
| iat_q90 | 206.572 | 0 | 0 | 170437.8 | 5479.433 |
| obj_size_q50 | 4096 | 4096 | 4096 | 4096 | 0 |
| obj_size_q90 | 4096 | 4096 | 4096 | 4096 | 0 |
| tenant_unique | 1 | 1 | 1 | 1 | 0 |
| opcode_switch_ratio | 0.003 | 0 | 0 | 0.235 | 0.012 |
| iat_lag1_autocorr | 0.022 | 0 | -0.577 | 0.769 | 0.13 |
| forward_seek_ratio | 0.872 | 0.869 | 0.307 | 0.998 | 0.091 |
| backward_seek_ratio | 0.124 | 0.129 | 0.002 | 0.471 | 0.087 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv |
|---|---|---:|---:|---:|
| s3-cache-datasets/cache_dataset_lcs/alibaba/117.lcs.zst | lcs | 0.008 | 0 | 63.984 |
| s3-cache-datasets/cache_dataset_lcs/alibaba/206.lcs.zst | lcs | 0.011 | 0 | 63.984 |
| s3-cache-datasets/cache_dataset_lcs/alibaba/4.lcs.zst | lcs | 0.163 | 0 | 63.984 |
| s3-cache-datasets/cache_dataset_lcs/alibaba/746.lcs.zst | lcs | 1 | 0 | 63.984 |
| s3-cache-datasets/cache_dataset_lcs/alibaba/801.lcs.zst | lcs | 0.995 | 0 | 63.984 |
