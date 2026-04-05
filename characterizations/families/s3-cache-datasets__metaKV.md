# s3-cache-datasets / metaKV

- Files: 11
- Bytes: 33279238718
- Formats: lcs
- Parsers: lcs

## Observations

- Predominantly read-heavy.
- High temporal locality / reuse.
- Highly bursty arrivals.

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| lcs | 11 | lcs |

## Metrics

| Metric | Mean | Median | Min | Max | SD |
|---|---:|---:|---:|---:|---:|
| write_ratio | 0.069 | 0.044 | 0.019 | 0.153 | 0.053 |
| reuse_ratio | 0.759 | 0.755 | 0.507 | 0.93 | 0.144 |
| burstiness_cv | 11.863 | 10.118 | 4.479 | 21.307 | 7.314 |
| iat_q50 | 0 | 0 | 0 | 0 | 0 |
| iat_q90 | 0 | 0 | 0 | 0 | 0 |
| obj_size_q50 | 105.909 | 59 | 17 | 235 | 79.939 |
| obj_size_q90 | 876.273 | 568 | 203 | 3479 | 916.654 |
| tenant_unique | 20.909 | 1 | 1 | 103 | 37.668 |
| opcode_switch_ratio | 0.101 | 0.074 | 0.027 | 0.248 | 0.074 |
| iat_lag1_autocorr | 0.005 | -0.001 | -0.033 | 0.052 | 0.028 |
| forward_seek_ratio | 0.187 | 0.167 | 0.054 | 0.445 | 0.134 |
| backward_seek_ratio | 0.054 | 0.048 | 0.002 | 0.13 | 0.049 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv |
|---|---|---:|---:|---:|
| s3-cache-datasets/cache_dataset_lcs/metaKV/202210_kv_traces_all_sort.csv.lcs.zst | lcs | 0.021 | 0.914 | 21.307 |
| s3-cache-datasets/cache_dataset_lcs/metaKV/202401_kv_traces_all_sort.csv.lcs.zst | lcs | 0.019 | 0.93 | 20.211 |
| s3-cache-datasets/cache_dataset_lcs/metaKV/202210_kv_traces_all_sort.csv.lcs.sample0.1.zst | lcs | 0.044 | 0.805 | 10.333 |
| s3-cache-datasets/cache_dataset_lcs/metaKV/202401_kv_traces_all_sort.csv.lcs.sample0.1.zst | lcs | 0.153 | 0.755 | 9.904 |
| s3-cache-datasets/cache_dataset_lcs/metaKV/202401_kv_traces_all_sort.csv.lcs.sample0.01.zst | lcs | 0.153 | 0.535 | 4.944 |
