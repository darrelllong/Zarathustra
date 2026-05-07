# s3-cache-datasets / metaKV

- Files: 11
- Bytes: 33279238718
- Formats: lcs
- Parsers: lcs
- ML Use Cases: request_sequence
- Heterogeneity Score: 0.768
- Suggested GAN Modes: 1
- Split By Format: no

## Observations

- Predominantly read-heavy.
- High temporal locality / reuse.
- Highly bursty arrivals.

## GAN Guidance

- Opcode balance is extremely read-skewed; generation should not assume symmetric read/write behavior.
- Reuse/locality is a major axis here; locality-aware losses and conditioning should matter.
- Burstiness is high; inter-arrival and FFT/ACF losses should stay heavily weighted.
- Strongest feature coupling in this pass: ts_duration vs iat_mean (corr=1).
- Current characterization suggests extra conditioning value from: obj_size_std.

## Conditioning Audit

| Item | Value |
|---|---|
| Near-constant current conditioning features | iat_q50 |
| Recommended candidate additions | obj_size_std |
| Highly redundant current pairs | none flagged |

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| lcs | 11 | lcs |

## Clustering And Regimes

| Item | Value |
|---|---|

## Strongest Correlations

| Metric A | Metric B | Correlation |
|---|---|---:|
| ts_duration | iat_mean | 1 |
| lcs_version | feature_field_count | 1 |
| ts_duration | iat_zero_ratio | -1 |
| iat_zero_ratio | iat_mean | -1 |
| burstiness_cv | signed_stride_lag1_autocorr | 0.999 |
| sample_record_rate | signed_stride_lag1_autocorr | 0.993 |
| sample_record_rate | burstiness_cv | 0.988 |
| reuse_ratio | object_unique | -0.987 |

## Metrics

| Metric | Mean | Median | CV | Skew | Kurtosis | Missing | Q10 | Q90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tenant_unique | 20.909 | 1 | 1.802 | N/A | N/A | 0 | 1 | 85 |
| ts_duration | 46.909 | 9 | 1.721 | N/A | N/A | 0 | 0 | 204 |
| iat_mean | 0.011 | 0.002 | 1.721 | N/A | N/A | 0 | 0 | 0.05 |
| iat_q99 | 0.273 | 0 | 1.713 | N/A | N/A | 0 | 0 | 1 |
| feature_field_count | 0.273 | 0 | 1.713 | N/A | N/A | 0 | 0 | 1 |
| iat_std | 0.071 | 0.047 | 1.302 | N/A | N/A | 0 | 0 | 0.223 |
| size_bytes | 3025385338 | 780415344 | 1.272 | N/A | N/A | 0 | 69150661 | 9138448556 |
| obj_size_min | 22.909 | 8 | 1.115 | N/A | N/A | 0 | 8 | 63 |
| obj_size_q90 | 876.273 | 568 | 1.046 | N/A | N/A | 0 | 309 | 1216 |
| backward_seek_ratio | 0.054 | 0.048 | 0.916 | N/A | N/A | 0 | 0.002 | 0.123 |
| write_ratio | 0.069 | 0.044 | 0.768 | N/A | N/A | 0 | 0.021 | 0.153 |
| obj_size_q50 | 105.909 | 59 | 0.755 | N/A | N/A | 0 | 40 | 226 |
| opcode_switch_ratio | 0.101 | 0.074 | 0.736 | N/A | N/A | 0 | 0.033 | 0.204 |
| forward_seek_ratio | 0.187 | 0.167 | 0.716 | N/A | N/A | 0 | 0.067 | 0.417 |
| obj_size_std | 3960.184 | 3118.714 | 0.625 | N/A | N/A | 0 | 1742.492 | 7355.84 |
| object_top1_share | 0.246 | 0.206 | 0.545 | N/A | N/A | 0 | 0.119 | 0.441 |
| object_unique | 604.273 | 680 | 0.471 | N/A | N/A | 0 | 284 | 1043 |
| obj_size_mean | 548.904 | 590.693 | 0.389 | N/A | N/A | 0 | 306.218 | 749.087 |

## Outlier Files

| rel_path | outlier_score | top drivers |
|---|---:|---|
| N/A | N/A | N/A |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv | ts_duration |
|---|---|---:|---:|---:|---:|
| s3-cache-datasets/cache_dataset_lcs/metaKV/202210_kv_traces_all_sort.csv.lcs.zst | lcs | 0.021 | 0.914 | 21.307 | 9 |
| s3-cache-datasets/cache_dataset_lcs/metaKV/202401_kv_traces_all_sort.csv.lcs.zst | lcs | 0.019 | 0.93 | 20.211 | 10 |
| s3-cache-datasets/cache_dataset_lcs/metaKV/202210_kv_traces_all_sort.csv.lcs.sample0.1.zst | lcs | 0.044 | 0.805 | 10.333 | 38 |
| s3-cache-datasets/cache_dataset_lcs/metaKV/202401_kv_traces_all_sort.csv.lcs.sample0.1.zst | lcs | 0.153 | 0.755 | 9.904 | 45 |
| s3-cache-datasets/cache_dataset_lcs/metaKV/202401_kv_traces_all_sort.csv.lcs.sample0.01.zst | lcs | 0.153 | 0.535 | 4.944 | 210 |
| s3-cache-datasets/cache_dataset_lcs/metaKV/202210_kv_traces_all_sort.csv.lcs.sample0.01.zst | lcs | 0.125 | 0.507 | 4.479 | 204 |
| s3-cache-datasets/cache_dataset_lcs/metaKV/202206_kv_traces_all.csv.lcs.sample0.01.zst | lcs | 0.093 | 0.694 | N/A | 0 |
| s3-cache-datasets/cache_dataset_lcs/metaKV/202206_kv_traces_all.csv.lcs.sample0.1.zst | lcs | 0.066 | 0.73 | N/A | 0 |
