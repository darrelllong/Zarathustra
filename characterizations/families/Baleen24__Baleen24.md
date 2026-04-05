# Baleen24 / Baleen24

- Files: 374
- Bytes: 31376065904
- Formats: baleen24
- Parsers: baleen24

## Observations

- Substantial write pressure across sampled files.

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| baleen24 | 374 | baleen24 |

## Metrics

| Metric | Mean | Median | Min | Max | SD |
|---|---:|---:|---:|---:|---:|
| write_ratio | 0.641 | 0.646 | 0.175 | 1 | 0.168 |
| reuse_ratio | 0.368 | 0.311 | 0.077 | 1 | 0.21 |
| burstiness_cv | 1.52 | 1.415 | 1.011 | 4.278 | 0.357 |
| iat_q50 | 1.025 | 1 | 0 | 3.9 | 0.981 |
| iat_q90 | 6.723 | 6 | 0.003 | 81 | 9.01 |
| obj_size_q50 | 1124316 | 1048576 | 262144 | 3191316 | 497918.8 |
| obj_size_q90 | 7733403 | 8388608 | 524288 | 8519680 | 2039052 |
| tenant_unique | 120.971 | 28 | 1 | 587 | 155.307 |
| opcode_switch_ratio | 0.288 | 0.304 | 0 | 0.426 | 0.088 |
| iat_lag1_autocorr | 0.087 | 0.103 | -0.152 | 0.376 | 0.099 |
| forward_seek_ratio | 0.313 | 0.336 | 0 | 0.486 | 0.106 |
| backward_seek_ratio | 0.319 | 0.354 | 0 | 0.467 | 0.108 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv |
|---|---|---:|---:|---:|
| Baleen24/extracted/storage_0.1_10/storage/201910/Region2/full_0.2_0.1.trace | baleen24 | 0.844 | 0.854 | 4.278 |
| Baleen24/extracted/storage_10/storage/201910/Region2/full_0.2_0.1.trace | baleen24 | 0.844 | 0.854 | 4.278 |
| Baleen24/extracted/storage/storage/201910/Region2/full_0.2_0.1.trace | baleen24 | 0.844 | 0.854 | 4.278 |
| Baleen24/extracted/storage_10/storage/202110/Region4/full_3_1.trace | baleen24 | 0.666 | 0.201 | 2.398 |
| Baleen24/extracted/storage/storage/202110/Region4/full_3_1.trace | baleen24 | 0.666 | 0.201 | 2.398 |
