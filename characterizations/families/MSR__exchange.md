# MSR / exchange

- Files: 96
- Bytes: 1479416826
- Formats: exchange_etw
- Parsers: exchange_etw

## Observations

- Substantial write pressure across sampled files.
- Very weak short-window reuse.

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| exchange_etw | 96 | exchange_etw |

## Metrics

| Metric | Mean | Median | Min | Max | SD |
|---|---:|---:|---:|---:|---:|
| write_ratio | 0.747 | 0.786 | 0.001 | 0.93 | 0.157 |
| reuse_ratio | 0.003 | 0.003 | 0 | 0.018 | 0.003 |
| burstiness_cv | 4.415 | 4.495 | 1.123 | 7.508 | 1.381 |
| iat_q50 | 130.573 | 115.5 | 47 | 307 | 50.35 |
| iat_q90 | 3467.21 | 3484.3 | 290 | 6563.8 | 1345.811 |
| obj_size_q50 | 7021.333 | 8192 | 4096 | 8192 | 1841.831 |
| obj_size_q90 | 38592 | 32768 | 8192 | 65536 | 18293.41 |
| tenant_unique | 8.99 | 9 | 7 | 10 | 0.229 |
| opcode_switch_ratio | 0.09 | 0.089 | 0.002 | 0.177 | 0.036 |
| iat_lag1_autocorr | 0.154 | 0.124 | -0.128 | 0.625 | 0.123 |
| forward_seek_ratio | 0.634 | 0.638 | 0.495 | 0.743 | 0.042 |
| backward_seek_ratio | 0.362 | 0.356 | 0.253 | 0.505 | 0.041 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv |
|---|---|---:|---:|---:|
| MSR/Exchange-Server-Traces/Exchange/Exchange.12-13-2007.12-44-AM.trace.csv.gz | exchange_etw | 0.93 | 0.011 | 7.508 |
| MSR/Exchange-Server-Traces/Exchange/Exchange.12-13-2007.06-17-AM.trace.csv.gz | exchange_etw | 0.911 | 0.002 | 6.974 |
| MSR/Exchange-Server-Traces/Exchange/Exchange.12-13-2007.02-30-AM.trace.csv.gz | exchange_etw | 0.915 | 0.007 | 6.964 |
| MSR/Exchange-Server-Traces/Exchange/Exchange.12-12-2007.09-12-PM.trace.csv.gz | exchange_etw | 0.844 | 0.008 | 6.282 |
| MSR/Exchange-Server-Traces/Exchange/Exchange.12-13-2007.02-45-AM.trace.csv.gz | exchange_etw | 0.923 | 0.01 | 6.277 |
