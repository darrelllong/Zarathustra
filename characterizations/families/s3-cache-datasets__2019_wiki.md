# s3-cache-datasets / 2019_wiki

- Files: 6
- Bytes: 79475295162
- Formats: oracle_general, text, text_zst
- Parsers: generic_text, oracle_general
- ML Use Cases: request_sequence, structured_table
- Heterogeneity Score: 0.323
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
- Current characterization suggests extra conditioning value from: object_unique, signed_stride_lag1_autocorr, obj_size_std.

## Conditioning Audit

| Item | Value |
|---|---|
| Near-constant current conditioning features | write_ratio, iat_q50, opcode_switch_ratio, tenant_unique |
| Recommended candidate additions | object_unique, signed_stride_lag1_autocorr, obj_size_std |
| Highly redundant current pairs | none flagged |

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| oracle_general | 3 | oracle_general |
| text | 1 | generic_text |
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
| size_bytes | 13245882527 | 12912860594 | 0.989 | 0.024 | -3.19 | 0 | 1018485170 | 25806301816 |
| sample_records | 4096 | 4096 | 0 | N/A | N/A | 0 | 4096 | 4096 |
| ttl_present | 0 | 0 | N/A | N/A | N/A | 0 | 0 | 0 |
| obj_size_min | 1425.667 | 95 | 1.628 | 1.732 | N/A | 0.5 | 79.8 | 3303.8 |
| ts_duration | 10.667 | 2 | 1.572 | 1.704 | N/A | 0.5 | 0.4 | 24.4 |
| iat_mean | 0.003 | 0 | 1.572 | 1.704 | N/A | 0.5 | 0 | 0.006 |
| first_numeric_diff_mean | 0.003 | 0 | 1.426 | 1.732 | N/A | 0.5 | 0 | 0.006 |
| iat_std | 0.036 | 0.022 | 1.237 | 1.259 | N/A | 0.5 | 0.004 | 0.073 |
| reuse_ratio | 0.001 | 0.001 | 0.972 | -0.271 | N/A | 0.5 | 0 | 0.002 |
| abs_stride_q50 | 12297829380359000064 | 18446744069588279296 | 0.866 | -1.732 | N/A | 0.5 | 3689348815380571648 | 18446744069645729792 |
| first_numeric_diff_std | 0.043 | 0.022 | 0.845 | 1.732 | N/A | 0.5 | 0.022 | 0.073 |
| object_top10_share | 0.059 | 0.045 | 0.791 | 1.255 | N/A | 0.5 | 0.026 | 0.098 |
| obj_size_std | 261801.1 | 355813.9 | 0.756 | -1.656 | N/A | 0.5 | 98754.97 | 387242.1 |
| obj_size_q50 | 12503.33 | 10272.5 | 0.746 | 1.015 | N/A | 0.5 | 5651.3 | 20247.7 |
| object_top1_share | 0.017 | 0.023 | 0.706 | -1.712 | N/A | 0.5 | 0.007 | 0.024 |
| obj_size_q99 | 377983.3 | 485855.1 | 0.496 | -1.732 | N/A | 0.5 | 226259.5 | 486558.3 |
| obj_size_q90 | 54673.33 | 52224 | 0.484 | 0.413 | N/A | 0.5 | 34044 | 76282.4 |
| obj_size_mean | 39870.65 | 36764.62 | 0.161 | 1.668 | N/A | 0.5 | 35831.77 | 45151.94 |

## Outlier Files

| rel_path | outlier_score | top drivers |
|---|---:|---|
| N/A | N/A | N/A |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv | ts_duration |
|---|---|---:|---:|---:|---:|
| s3-cache-datasets/cache_dataset_oracleGeneral/2019_wiki/wiki_2019u.oracleGeneral.zst | oracle_general | 0 | 0 | 45.238 | 2 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2019_wiki/wiki_2019t.oracleGeneral.zst | oracle_general | 0 | 0.001 | 11.64 | 30 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2019_wiki/wiki_2016u.oracleGeneral.zst | oracle_general | 0 | 0.003 | N/A | 0 |
| s3-cache-datasets/cache_dataset_txt/2019_wiki/wiki/2019/wiki.upload.2019.short | text | N/A | N/A | N/A | N/A |
| s3-cache-datasets/cache_dataset_txt/2019_wiki/wiki/2019/wiki.txt.2019.zst | text_zst | N/A | N/A | N/A | N/A |
| s3-cache-datasets/cache_dataset_txt/2019_wiki/wiki/2019/wiki.upload.2019.zst | text_zst | N/A | N/A | N/A | N/A |
