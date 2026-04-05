# alibaba / alibaba

- Files: 1000
- Bytes: 100765044913
- Formats: oracle_general
- Parsers: oracle_general

## Observations

- Predominantly read-heavy.
- Very weak short-window reuse.

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| oracle_general | 1000 | oracle_general |

## Metrics

| Metric | Mean | Median | Min | Max | SD |
|---|---:|---:|---:|---:|---:|
| write_ratio | 0 | 0 | 0 | 0 | 0 |
| reuse_ratio | 0.004 | 0 | 0 | 0.683 | 0.029 |
| burstiness_cv | 8.116 | 3.833 | 0.822 | 63.984 | 10.36 |
| iat_q50 | 0.099 | 0 | 0 | 30 | 1.157 |
| iat_q90 | 226.35 | 1 | 0 | 173030.6 | 5608.065 |
| obj_size_q50 | 41164.24 | 4096 | 1024 | 524288 | 113291.3 |
| obj_size_q90 | 80101.91 | 16384 | 1024 | 524288 | 155944.4 |
| tenant_unique | 1.997 | 2 | 1 | 2 | 0.055 |
| opcode_switch_ratio | 0 | 0 | 0 | 0 | 0 |
| iat_lag1_autocorr | -0.022 | -0.009 | -1 | 0.77 | 0.167 |
| forward_seek_ratio | 0.691 | 0.678 | 0.22 | 0.978 | 0.11 |
| backward_seek_ratio | 0.305 | 0.32 | 0.021 | 0.661 | 0.107 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv |
|---|---|---:|---:|---:|
| alibaba/alibabaBlock_4.oracleGeneral.zst | oracle_general | 0 | 0 | 63.984 |
| alibaba/alibabaBlock_810.oracleGeneral.zst | oracle_general | 0 | 0 | 63.899 |
| alibaba/1M/alibabaBlock_808.oracleGeneral.zst | oracle_general | 0 | 0 | 54.095 |
| alibaba/10K/alibabaBlock_821.oracleGeneral.zst | oracle_general | 0 | 0.012 | 51.321 |
| alibaba/10K/alibabaBlock_822.oracleGeneral.zst | oracle_general | 0 | 0 | 45.625 |
