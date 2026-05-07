# alibaba / alibaba

- Files: 1000
- Bytes: 100765044913
- Formats: oracle_general
- Parsers: oracle_general
- ML Use Cases: request_sequence
- Heterogeneity Score: 2.043
- Suggested GAN Modes: 1
- Split By Format: no

## Observations

- Predominantly read-heavy.
- Very weak short-window reuse.

## GAN Guidance

- Sequential blocks are much more internally coherent than random file batches; block or curriculum sampling is likely safer than pure iid file sampling.
- Opcode balance is extremely read-skewed; generation should not assume symmetric read/write behavior.
- Strongest feature coupling in this pass: iat_mean vs iat_q90 (corr=1).
- A small set of files are strong multivariate outliers; consider holding them out for ablation or separate mode inspection.
- Current characterization suggests extra conditioning value from: object_unique, signed_stride_lag1_autocorr, obj_size_std.

## Conditioning Audit

| Item | Value |
|---|---|
| Near-constant current conditioning features | write_ratio, iat_q50, opcode_switch_ratio, tenant_unique |
| Recommended candidate additions | object_unique, signed_stride_lag1_autocorr, obj_size_std |
| Highly redundant current pairs | forward_seek_ratio vs backward_seek_ratio (-0.966) |

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| oracle_general | 1000 | oracle_general |

## Clustering And Regimes

| Item | Value |
|---|---|
| K-means selected K | 2 |
| Best silhouette K | 2 |
| PCA variance explained by PC1 | 0.219 |
| Block/random distance ratio | 0.545 |
| Sampling recommendation | block_sampling_preserves_temporal_coherence |

### K Selection

| K | Within-SS | Silhouette |
|---:|---:|---:|
| 2 | 17366892647652245882535936 | 0.881 |
| 3 | 9438996643215217885446144 | 0.797 |
| 4 | 5420951043535495896236032 | 0.78 |
| 5 | 4135037713399320680595456 | 0.775 |
| 6 | 4225221846738098855084032 | 0.695 |
| 7 | 3289008419059508355006464 | 0.666 |
| 8 | 3152888840553245471408128 | 0.632 |
| 9 | 3095231218888217009848320 | 0.434 |
| 10 | 3052330031702617618382848 | 0.437 |
| 11 | 3016717209072445257940992 | 0.465 |
| 12 | 2803767208156508209545216 | 0.341 |

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
| size_bytes | 100765045 | 12080977 | 6.815 | N/A | N/A | 0 | 128091.3 | 118116398 |
| sample_records | 4047.027 | 4096 | 0.1 | N/A | N/A | 0 | 4096 | 4096 |
| ttl_present | 0 | 0 | N/A | N/A | N/A | 0 | 0 | 0 |
| iat_q90 | 226.35 | 1 | 24.776 | N/A | N/A | 0.001 | 0 | 6 |
| iat_mean | 74.891 | 0.463 | 21.924 | N/A | N/A | 0.001 | 0.004 | 4.549 |
| iat_q99 | 554.557 | 5 | 16.552 | N/A | N/A | 0.001 | 0 | 51.792 |
| iat_std | 190.031 | 1.372 | 14.37 | N/A | N/A | 0.001 | 0.1 | 16.722 |
| iat_q50 | 0.099 | 0 | 11.737 | N/A | N/A | 0.001 | 0 | 0 |
| ts_duration | 28117.43 | 1888 | 7.839 | N/A | N/A | 0.001 | 14 | 18630 |
| iat_lag1_autocorr | -0.022 | -0.009 | 7.684 | N/A | N/A | 0.001 | -0.171 | 0.1 |
| reuse_ratio | 0.004 | 0 | 6.507 | N/A | N/A | 0.001 | 0 | 0.007 |
| abs_stride_q50 | 4144475437 | 5390336 | 4.844 | N/A | N/A | 0.001 | 17203.2 | 4701389619 |
| sample_record_rate | 90.765 | 2.158 | 3.495 | N/A | N/A | 0.001 | 0.22 | 276.968 |
| obj_size_q50 | 41164.24 | 4096 | 2.752 | N/A | N/A | 0.001 | 4096 | 98201.6 |
| abs_stride_q99 | 68473354998 | 24607121900 | 2.343 | N/A | N/A | 0.001 | 13470522442 | 169392239018 |
| abs_stride_q90 | 30101766113 | 11156923187 | 2.339 | N/A | N/A | 0.001 | 14465434 | 65308499804 |
| abs_stride_mean | 10682837839 | 3774048681 | 2.291 | N/A | N/A | 0.001 | 1018801892 | 24291638633 |
| abs_stride_std | 17059814945 | 6141127514 | 2.155 | N/A | N/A | 0.001 | 3093798238 | 35656136912 |

## Outlier Files

| rel_path | outlier_score | top drivers |
|---|---:|---|
| alibaba/1K/alibabaBlock_809.oracleGeneral.zst | 675.801 | iat_q90 (z=173029.6); iat_mean (z=109502.7) |
| alibaba/alibabaBlock_207.oracleGeneral.zst | 102.5 | abs_stride_q99 (z=343.168); abs_stride_std (z=230.713) |
| alibaba/1K/alibabaBlock_811.oracleGeneral.zst | 99.492 | iat_q90 (z=37947.8); iat_mean (z=23906.63) |
| alibaba/1K/alibabaBlock_816.oracleGeneral.zst | 82.432 | iat_std (z=18020.07); iat_q99 (z=16200.08) |
| alibaba/alibabaBlock_791.oracleGeneral.zst | 60.827 | abs_stride_std (z=159.681); abs_stride_mean (z=133.707) |
| alibaba/alibabaBlock_369.oracleGeneral.zst | 50.437 | abs_stride_q99 (z=202.187); abs_stride_std (z=152.851) |
| alibaba/alibabaBlock_771.oracleGeneral.zst | 41.62 | abs_stride_q50 (z=49912.17); abs_stride_mean (z=101.651) |
| alibaba/10K/alibabaBlock_805.oracleGeneral.zst | 29.965 | abs_stride_q50 (z=33193.55); iat_q99 (z=4175.6) |

## Outlier Sensitivity

| N Removed | Metric | Baseline Median | Trimmed Median | Relative Shift |
|---:|---|---:|---:|---:|
| 10 | abs_stride_mean | 3774048681 | 3742613921 | -0.008 |
| 5 | abs_stride_mean | 3774048681 | 3758016943 | -0.004 |
| 1 | abs_stride_mean | 3774048681 | 3770559664 | -0.001 |
| 3 | abs_stride_mean | 3774048681 | 3770559664 | -0.001 |
| 1 | object_unique | 2182 | 2183 | 0 |
| 3 | object_unique | 2182 | 2183 | 0 |
| 5 | object_unique | 2182 | 2183 | 0 |
| 10 | burstiness_cv | 3.833 | 3.834 | 0 |
| 5 | burstiness_cv | 3.833 | 3.834 | 0 |
| 3 | burstiness_cv | 3.833 | 3.833 | 0 |

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
