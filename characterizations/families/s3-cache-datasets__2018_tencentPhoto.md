# s3-cache-datasets / 2018_tencentPhoto

- Files: 4
- Bytes: 163051085109
- Formats: oracle_general, text_zst
- Parsers: generic_text, oracle_general
- ML Use Cases: request_sequence, structured_table
- Heterogeneity Score: 0.001
- Suggested GAN Modes: 1
- Split By Format: yes

## Observations

- Predominantly read-heavy.
- Very weak short-window reuse.
- Highly bursty arrivals.
- Much of the family is only partially represented in numeric features; interpret structured-table metrics separately.

## GAN Guidance

- Family spans multiple encodings; keep format-aware preprocessing and avoid blindly pooling structured-table and request-sequence variants.
- Opcode balance is extremely read-skewed; generation should not assume symmetric read/write behavior.
- Burstiness is high; inter-arrival and FFT/ACF losses should stay heavily weighted.

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| oracle_general | 2 | oracle_general |
| text_zst | 2 | generic_text |

## Clustering And Regimes

| Item | Value |
|---|---|

## Strongest Correlations

| Metric A | Metric B | Correlation |
|---|---|---:|
| N/A | N/A | N/A |

## Metrics

| Metric | Mean | Median | CV | Skew | Kurtosis | Missing | Q10 | Q90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| size_bytes | 40762771277 | 36066086455 | 0.42 | 1.471 | 2.692 | 0 | 28398878774 | 56884011638 |
| sample_records | 4096 | 4096 | 0 | N/A | N/A | 0 | 4096 | 4096 |
| ttl_present | 0 | 0 | N/A | N/A | N/A | 0 | 0 | 0 |
| first_numeric_diff_std | 0.338 | 0.338 | 1.268 | N/A | N/A | 0.5 | 0.096 | 0.581 |
| first_numeric_diff_mean | 0.006 | 0.006 | 1.131 | N/A | N/A | 0.5 | 0.002 | 0.01 |
| obj_size_min | 487 | 487 | 0.357 | N/A | N/A | 0.5 | 388.6 | 585.4 |
| object_top10_share | 0.01 | 0.01 | 0.207 | N/A | N/A | 0.5 | 0.009 | 0.011 |
| obj_size_q99 | 214041.1 | 214041.1 | 0.157 | N/A | N/A | 0.5 | 195073.7 | 233008.6 |
| obj_size_std | 136093.2 | 136093.2 | 0.134 | N/A | N/A | 0.5 | 125777.1 | 146409.3 |
| object_top1_share | 0.001 | 0.001 | 0.129 | N/A | N/A | 0.5 | 0.001 | 0.001 |
| reuse_ratio | 0.009 | 0.009 | 0.06 | N/A | N/A | 0.5 | 0.008 | 0.009 |
| obj_size_mean | 32770.83 | 32770.83 | 0.055 | N/A | N/A | 0.5 | 31743.54 | 33798.13 |
| obj_size_q50 | 14442.5 | 14442.5 | 0.023 | N/A | N/A | 0.5 | 14252.1 | 14632.9 |
| tenant_top1_share | 0.83 | 0.83 | 0.019 | N/A | N/A | 0.5 | 0.822 | 0.839 |
| abs_stride_q50 | 5316512646079543296 | 5316512646079543296 | 0.012 | N/A | N/A | 0.5 | 5279055717908032512 | 5353969574251055104 |
| signed_stride_lag1_autocorr | -0.49 | -0.49 | 0.011 | N/A | N/A | 0.5 | -0.493 | -0.487 |
| abs_stride_std | 4403435667058213376 | 4403435667058213376 | 0.005 | N/A | N/A | 0.5 | 4391190834353107456 | 4415680499763319296 |
| obj_size_q90 | 62235.5 | 62235.5 | 0.005 | N/A | N/A | 0.5 | 62067.5 | 62403.5 |

## Outlier Files

| rel_path | outlier_score |
|---|---:|
| N/A | N/A |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv | ts_duration |
|---|---|---:|---:|---:|---:|
| s3-cache-datasets/cache_dataset_oracleGeneral/2018_tencentPhoto/tencent_photo1.oracleGeneral.zst | oracle_general | 0 | 0.008 | 28.601 | 5 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2018_tencentPhoto/tencent_photo2.oracleGeneral.zst | oracle_general | 0 | 0.009 | 28.601 | 5 |
| s3-cache-datasets/cache_dataset_txt/2018_tencentPhoto/tencentPhoto1.sort.zst | text_zst | N/A | N/A | N/A | N/A |
| s3-cache-datasets/cache_dataset_txt/2018_tencentPhoto/tencentPhoto2.sort.zst | text_zst | N/A | N/A | N/A | N/A |
