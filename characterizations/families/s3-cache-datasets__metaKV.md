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
- A small set of files are strong multivariate outliers; consider holding them out for ablation or separate mode inspection.

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| lcs | 11 | lcs |

## Clustering And Regimes

| Item | Value |
|---|---|
| PCA variance explained by PC1 | 0.565 |

## Strongest Correlations

| Metric A | Metric B | Correlation |
|---|---|---:|
| ts_duration | iat_mean | 1 |
| lcs_version | feature_field_count | 1 |
| ts_duration | iat_zero_ratio | -1 |
| iat_zero_ratio | iat_mean | -1 |
| abs_stride_q99 | tenant_top10_share | -0.998 |
| abs_stride_mean | abs_stride_q90 | 0.997 |
| sample_record_rate | burstiness_cv | 0.988 |
| reuse_ratio | object_unique | -0.987 |

## Metrics

| Metric | Mean | Median | CV | Skew | Kurtosis | Missing | Q10 | Q90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| abs_stride_q90 | 356872358451408064 | 961600 | 2.802 | 3.138 | 10.034 | 0 | 0 | 603801665740767488 |
| abs_stride_mean | 158384272065069664 | 434482.8 | 2.552 | 3.015 | 9.34 | 0 | 67149.01 | 321507278532667136 |
| abs_stride_q99 | 2369991040115161600 | 4041960 | 2.262 | 2.47 | 5.914 | 0 | 1756175 | 7936426556739888128 |
| abs_stride_std | 527912687150487488 | 1062088 | 1.988 | 2.135 | 4.275 | 0 | 392167.6 | 1690009152147212288 |
| tenant_unique | 20.909 | 1 | 1.802 | 1.728 | 1.622 | 0 | 1 | 85 |
| ts_duration | 46.909 | 9 | 1.721 | 1.764 | 1.639 | 0 | 0 | 204 |
| iat_mean | 0.011 | 0.002 | 1.721 | 1.764 | 1.639 | 0 | 0 | 0.05 |
| iat_q99 | 0.273 | 0 | 1.713 | 1.189 | -0.764 | 0 | 0 | 1 |
| feature_field_count | 0.273 | 0 | 1.713 | 1.189 | -0.764 | 0 | 0 | 1 |
| iat_std | 0.071 | 0.047 | 1.302 | 1.243 | 0.389 | 0 | 0 | 0.223 |
| size_bytes | 3025385338 | 780415344 | 1.272 | 0.95 | -1.066 | 0 | 69150661 | 9138448556 |
| obj_size_min | 22.909 | 8 | 1.115 | 1.19 | -0.762 | 0 | 8 | 63 |
| obj_size_q90 | 876.273 | 568 | 1.046 | 2.68 | 7.865 | 0 | 309 | 1216 |
| backward_seek_ratio | 0.054 | 0.048 | 0.916 | 0.623 | -1.151 | 0 | 0.002 | 0.123 |
| signed_stride_lag1_autocorr | -0.136 | -0.109 | 0.863 | -0.015 | -1.817 | 0 | -0.274 | -0.014 |
| write_ratio | 0.069 | 0.044 | 0.768 | 0.734 | -1.179 | 0 | 0.021 | 0.153 |
| obj_size_q50 | 105.909 | 59 | 0.755 | 0.715 | -1.161 | 0 | 40 | 226 |
| opcode_switch_ratio | 0.101 | 0.074 | 0.736 | 0.901 | -0.124 | 0 | 0.033 | 0.204 |

## Outlier Files

| rel_path | outlier_score |
|---|---:|
| s3-cache-datasets/cache_dataset_lcs/metaKV/202401_kv_traces_all_sort.csv.lcs.sample0.01.zst | 3.062 |
| s3-cache-datasets/cache_dataset_lcs/metaKV/202401_kv_traces_all_sort.csv.lcs.sample0.1.zst | 2.885 |
| s3-cache-datasets/cache_dataset_lcs/metaKV/202210_kv_traces_all_sort.csv.lcs.sample0.01.zst | 2.87 |
| s3-cache-datasets/cache_dataset_lcs/metaKV/202401_kv_traces_all_sort.csv.lcs.zst | 2.805 |
| s3-cache-datasets/cache_dataset_lcs/metaKV/202210_kv_traces_all_sort.csv.lcs.sample0.1.zst | 1.897 |
| s3-cache-datasets/cache_dataset_lcs/metaKV/202210_kv_traces_all_sort.csv.lcs.zst | 1.482 |

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
