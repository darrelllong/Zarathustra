# alibaba / alibaba

- Files: 1000
- Bytes: 100765044913
- Formats: oracle_general
- Parsers: oracle_general
- ML Use Cases: request_sequence
- Heterogeneity Score: 2.043
- Suggested GAN Modes: 8
- Split By Format: no

## Observations

- Predominantly read-heavy.
- Very weak short-window reuse.
- Ordered feature trajectories show regime boundaries.

## GAN Guidance

- Ordered PC1 changepoints suggest 22 regimes when files are ordered by trace start time.
- Opcode balance is extremely read-skewed; generation should not assume symmetric read/write behavior.
- Strongest feature coupling in this pass: iat_mean vs iat_q90 (corr=1).
- A small set of files are strong multivariate outliers; consider holding them out for ablation or separate mode inspection.

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| oracle_general | 1000 | oracle_general |

## Clustering And Regimes

| Item | Value |
|---|---|
| DBSCAN clusters | 2 |
| DBSCAN noise fraction | 0.071 |
| Ordered PC1 changepoints | 21 |
| PCA variance explained by PC1 | 0.219 |

## Strongest Correlations

| Metric A | Metric B | Correlation |
|---|---|---:|
| iat_mean | iat_q90 | 0.995 |
| iat_std | iat_q99 | 0.973 |
| iat_mean | iat_std | 0.97 |
| obj_size_mean | obj_size_q50 | 0.966 |
| forward_seek_ratio | backward_seek_ratio | -0.966 |
| iat_mean | iat_q99 | 0.965 |
| obj_size_std | obj_size_q90 | 0.955 |
| obj_size_mean | obj_size_q90 | 0.95 |

## Metrics

| Metric | Mean | Median | CV | Skew | Kurtosis | Missing | Q10 | Q90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| size_bytes | 100765045 | 12080977 | 6.815 | 20.572 | 522.213 | 0 | 128091.3 | 118116398 |
| sample_records | 4047.027 | 4096 | 0.1 | -8.6 | 74.67 | 0 | 4096 | 4096 |
| ttl_present | 0 | 0 | N/A | N/A | N/A | 0 | 0 | 0 |
| iat_q90 | 226.35 | 1 | 24.776 | 29.681 | 907.81 | 0.001 | 0 | 6 |
| iat_mean | 74.891 | 0.463 | 21.924 | 29.048 | 879.935 | 0.001 | 0.004 | 4.549 |
| iat_q99 | 554.557 | 5 | 16.552 | 23.849 | 635.737 | 0.001 | 0 | 51.792 |
| iat_std | 190.031 | 1.372 | 14.37 | 24.951 | 696.9 | 0.001 | 0.1 | 16.722 |
| iat_q50 | 0.099 | 0 | 11.737 | 19.884 | 471.361 | 0.001 | 0 | 0 |
| ts_duration | 28117.43 | 1888 | 7.839 | 10.919 | 121.995 | 0.001 | 14 | 18630 |
| iat_lag1_autocorr | -0.022 | -0.009 | 7.684 | 0.44 | 6.971 | 0.001 | -0.171 | 0.1 |
| reuse_ratio | 0.004 | 0 | 6.507 | 18.34 | 389.289 | 0.001 | 0 | 0.007 |
| abs_stride_q50 | 4144475437 | 5390336 | 4.844 | 9.191 | 102.256 | 0.001 | 17203.2 | 4701389619 |
| sample_record_rate | 90.765 | 2.158 | 3.495 | 5.65 | 41.244 | 0.001 | 0.22 | 276.968 |
| obj_size_q50 | 41164.24 | 4096 | 2.752 | 3.093 | 8.109 | 0.001 | 4096 | 98201.6 |
| abs_stride_q99 | 68473354998 | 24607121900 | 2.343 | 11.234 | 190.425 | 0.001 | 13470522442 | 169392239018 |
| abs_stride_q90 | 30101766113 | 11156923187 | 2.339 | 6.555 | 66.933 | 0.001 | 14465434 | 65308499804 |
| abs_stride_mean | 10682837839 | 3774048681 | 2.291 | 5.708 | 43.504 | 0.001 | 1018801892 | 24291638633 |
| abs_stride_std | 17059814945 | 6141127514 | 2.155 | 7.647 | 87.423 | 0.001 | 3093798238 | 35656136912 |

## Outlier Files

| rel_path | outlier_score |
|---|---:|
| alibaba/1K/alibabaBlock_809.oracleGeneral.zst | 675.801 |
| alibaba/alibabaBlock_207.oracleGeneral.zst | 102.5 |
| alibaba/1K/alibabaBlock_811.oracleGeneral.zst | 99.492 |
| alibaba/1K/alibabaBlock_816.oracleGeneral.zst | 82.432 |
| alibaba/alibabaBlock_791.oracleGeneral.zst | 60.827 |
| alibaba/alibabaBlock_369.oracleGeneral.zst | 50.437 |
| alibaba/alibabaBlock_771.oracleGeneral.zst | 41.62 |
| alibaba/10K/alibabaBlock_805.oracleGeneral.zst | 29.965 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv | ts_duration |
|---|---|---:|---:|---:|---:|
| alibaba/alibabaBlock_4.oracleGeneral.zst | oracle_general | 0 | 0 | 63.984 | 1 |
| alibaba/alibabaBlock_810.oracleGeneral.zst | oracle_general | 0 | 0 | 63.899 | 55970 |
| alibaba/1M/alibabaBlock_808.oracleGeneral.zst | oracle_general | 0 | 0 | 54.095 | 211798 |
| alibaba/10K/alibabaBlock_821.oracleGeneral.zst | oracle_general | 0 | 0.012 | 51.321 | 606253 |
| alibaba/10K/alibabaBlock_822.oracleGeneral.zst | oracle_general | 0 | 0 | 45.625 | 956754 |
| alibaba/100K/alibabaBlock_831.oracleGeneral.zst | oracle_general | 0 | 0.001 | 45.238 | 2 |
| alibaba/100K/alibabaBlock_838.oracleGeneral.zst | oracle_general | 0 | 0.001 | 45.238 | 2 |
| alibaba/100K/alibabaBlock_839.oracleGeneral.zst | oracle_general | 0 | 0.001 | 45.238 | 2 |
