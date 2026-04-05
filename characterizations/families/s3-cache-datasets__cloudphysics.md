# s3-cache-datasets / cloudphysics

- Files: 107
- Bytes: 64319532236
- Formats: lcs, text
- Parsers: generic_text, lcs

## Observations

- Substantial write pressure across sampled files.
- Very weak short-window reuse.

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| lcs | 106 | lcs |
| text | 1 | generic_text |

## Metrics

| Metric | Mean | Median | Min | Max | SD |
|---|---:|---:|---:|---:|---:|
| write_ratio | 0.679 | 0.852 | 0 | 1 | 0.364 |
| reuse_ratio | 0.029 | 0.003 | 0 | 0.326 | 0.06 |
| burstiness_cv | 16.426 | 8.924 | 2.421 | 63.984 | 16.48 |
| iat_q50 | 0 | 0 | 0 | 0 | 0 |
| iat_q90 | 0.17 | 0 | 0 | 4 | 0.654 |
| obj_size_q50 | 4096 | 4096 | 4096 | 4096 | 0 |
| obj_size_q90 | 4096 | 4096 | 4096 | 4096 | 0 |
| tenant_unique | 1 | 1 | 1 | 1 | 0 |
| opcode_switch_ratio | 0.014 | 0.004 | 0 | 0.181 | 0.027 |
| iat_lag1_autocorr | 0.046 | 0.004 | -0.068 | 0.62 | 0.097 |
| forward_seek_ratio | 0.841 | 0.867 | 0.505 | 0.999 | 0.122 |
| backward_seek_ratio | 0.13 | 0.104 | 0.001 | 0.495 | 0.102 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv |
|---|---|---:|---:|---:|
| s3-cache-datasets/cache_dataset_lcs/cloudphysics/w03_vscsi1.vscsitrace.lcs.zst | lcs | 0.378 | 0.004 | 63.984 |
| s3-cache-datasets/cache_dataset_lcs/cloudphysics/w08_vscsi2.vscsitrace.lcs.zst | lcs | 0 | 0 | 63.984 |
| s3-cache-datasets/cache_dataset_lcs/cloudphysics/w13_vscsi1.vscsitrace.lcs.zst | lcs | 0 | 0 | 63.984 |
| s3-cache-datasets/cache_dataset_lcs/cloudphysics/w14_vscsi1.vscsitrace.lcs.zst | lcs | 0.424 | 0.003 | 63.984 |
| s3-cache-datasets/cache_dataset_lcs/cloudphysics/w22_vscsi2.vscsitrace.lcs.zst | lcs | 0.435 | 0 | 63.984 |
