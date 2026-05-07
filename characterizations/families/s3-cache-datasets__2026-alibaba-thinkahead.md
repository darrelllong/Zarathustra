# s3-cache-datasets / 2026-alibaba-thinkahead

- Files: 45
- Bytes: 15366738380
- Formats: oracle_general
- Parsers: oracle_general
- ML Use Cases: request_sequence
- Heterogeneity Score: 0.425
- Suggested GAN Modes: 1
- Split By Format: no

## Observations

- Predominantly read-heavy.
- Very weak short-window reuse.
- Highly bursty arrivals.

## GAN Guidance

- Opcode balance is extremely read-skewed; generation should not assume symmetric read/write behavior.
- Burstiness is high; inter-arrival and FFT/ACF losses should stay heavily weighted.
- Strongest feature coupling in this pass: ts_duration vs iat_mean (corr=1).
- A small set of files are strong multivariate outliers; consider holding them out for ablation or separate mode inspection.
- Current characterization suggests extra conditioning value from: object_unique, signed_stride_lag1_autocorr.

## Conditioning Audit

| Item | Value |
|---|---|
| Near-constant current conditioning features | write_ratio, iat_q50, obj_size_q50, opcode_switch_ratio, tenant_unique |
| Recommended candidate additions | object_unique, signed_stride_lag1_autocorr |
| Highly redundant current pairs | forward_seek_ratio vs backward_seek_ratio (-0.955) |

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| oracle_general | 45 | oracle_general |

## Clustering And Regimes

| Item | Value |
|---|---|
| K-means selected K | 2 |
| Best silhouette K | 2 |
| PCA variance explained by PC1 | 0.238 |
| Block/random distance ratio | 1.039 |
| Sampling recommendation | random_sampling_is_less_problematic |

### K Selection

| K | Within-SS | Silhouette |
|---:|---:|---:|
| 2 | 106776690978223506325504 | 0.938 |
| 3 | 35431607872077827145728 | 0.87 |
| 4 | 19829828452379996979200 | 0.882 |
| 5 | 5302177802903527358464 | 0.907 |
| 6 | 5251942494705387831296 | 0.823 |
| 7 | 5203954043661199605760 | 0.848 |
| 8 | 5143464622124497371136 | 0.808 |
| 9 | 5202446310692827955200 | 0.557 |
| 10 | 5141956889156126769152 | 0.516 |
| 11 | 5141930915724050563072 | 0.588 |
| 12 | 5141912746800728506368 | 0.598 |

## Strongest Correlations

| Metric A | Metric B | Correlation |
|---|---|---:|
| ts_duration | iat_mean | 1 |
| ts_duration | iat_std | 1 |
| iat_mean | iat_std | 1 |
| abs_stride_mean | abs_stride_q90 | 0.998 |
| abs_stride_mean | abs_stride_q50 | 0.974 |
| abs_stride_q90 | abs_stride_q99 | 0.968 |
| abs_stride_mean | abs_stride_q99 | 0.966 |
| abs_stride_q50 | abs_stride_q90 | 0.963 |

## Metrics

| Metric | Mean | Median | CV | Skew | Kurtosis | Missing | Q10 | Q90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| abs_stride_q50 | 4444443745 | 176 | 6.708 | N/A | N/A | 0 | 16 | 3012.6 |
| iat_std | 8.521 | 0.052 | 5.464 | N/A | N/A | 0 | 0.034 | 0.096 |
| ts_duration | 553.2 | 9 | 5.389 | N/A | N/A | 0 | 4.4 | 27.2 |
| iat_mean | 0.135 | 0.002 | 5.389 | N/A | N/A | 0 | 0.001 | 0.007 |
| abs_stride_q90 | 15557719215 | 1822152 | 4.933 | N/A | N/A | 0 | 911090.2 | 4697574 |
| abs_stride_mean | 6066488377 | 674742.4 | 4.93 | N/A | N/A | 0 | 395213.7 | 2364311534 |
| abs_stride_q99 | 24462338736 | 8277304 | 3.913 | N/A | N/A | 0 | 4699519 | 60082168770 |
| size_bytes | 341483075 | 6440531 | 3.535 | N/A | N/A | 0 | 901735.8 | 573440809 |
| abs_stride_std | 10718826025 | 2202018 | 3.33 | N/A | N/A | 0 | 1097926 | 20869063973 |
| iat_lag1_autocorr | 0.022 | -0.001 | 3.216 | N/A | N/A | 0 | -0.002 | 0.072 |
| reuse_ratio | 0.027 | 0.027 | 1.053 | N/A | N/A | 0 | 0.001 | 0.062 |
| object_top10_share | 0.028 | 0.021 | 0.823 | N/A | N/A | 0 | 0.014 | 0.061 |
| object_top1_share | 0.006 | 0.004 | 0.755 | N/A | N/A | 0 | 0.002 | 0.011 |
| sample_record_rate | 536.649 | 455.111 | 0.628 | N/A | N/A | 0 | 150.787 | 942.08 |
| signed_stride_lag1_autocorr | -0.357 | -0.393 | 0.425 | N/A | N/A | 0 | -0.491 | -0.139 |
| burstiness_cv | 29.031 | 26.106 | 0.403 | N/A | N/A | 0 | 18.244 | 40.814 |
| backward_seek_ratio | 0.272 | 0.306 | 0.33 | N/A | N/A | 0 | 0.16 | 0.371 |
| object_unique | 3492.911 | 3614 | 0.203 | N/A | N/A | 0 | 3250.8 | 3955.4 |

## Outlier Files

| rel_path | outlier_score | top drivers |
|---|---:|---|
| s3-cache-datasets/cache_dataset_oracleGeneral/2026-alibaba-thinkahead/c17.oracleGeneral.zst | 36.679 | abs_stride_q50 (z=1249998189); abs_stride_mean (z=631226.5) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2026-alibaba-thinkahead/c4.oracleGeneral.zst | 35.349 | abs_stride_std (z=109335.4); iat_std (z=29316.91) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2026-alibaba-thinkahead/c25.oracleGeneral.zst | 15.343 | iat_lag1_autocorr (z=420.801); object_top10_share (z=17.238) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2026-alibaba-thinkahead/c15.oracleGeneral.zst | 12.976 | iat_std (z=5715.798); abs_stride_std (z=5389.91) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2026-alibaba-thinkahead/c14.oracleGeneral.zst | 10.336 | object_top10_share (z=16.857); object_unique (z=-12.05) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2026-alibaba-thinkahead/c3.oracleGeneral.zst | 2.192 | iat_mean (z=5.667); ts_duration (z=5.667) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2026-alibaba-thinkahead/c26.oracleGeneral.zst | 1.982 | abs_stride_q99 (z=55.42); abs_stride_std (z=38.067) |
| s3-cache-datasets/cache_dataset_oracleGeneral/2026-alibaba-thinkahead/c21.oracleGeneral.zst | 1.958 | abs_stride_q90 (z=114583.3); abs_stride_mean (z=89459.59) |

## Outlier Sensitivity

| N Removed | Metric | Baseline Median | Trimmed Median | Relative Shift |
|---:|---|---:|---:|---:|
| 3 | burstiness_cv | 26.106 | 25.446 | -0.025 |
| 5 | burstiness_cv | 26.106 | 25.446 | -0.025 |
| 10 | burstiness_cv | 26.106 | 25.617 | -0.019 |
| 10 | reuse_ratio | 0.027 | 0.027 | 0.018 |
| 10 | abs_stride_mean | 674742.4 | 663604.4 | -0.016 |
| 1 | burstiness_cv | 26.106 | 25.861 | -0.009 |
| 1 | reuse_ratio | 0.027 | 0.027 | 0.009 |
| 3 | reuse_ratio | 0.027 | 0.027 | 0.009 |
| 5 | reuse_ratio | 0.027 | 0.027 | 0.009 |
| 1 | abs_stride_mean | 674742.4 | 669173.4 | -0.008 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv | ts_duration |
|---|---|---:|---:|---:|---:|
| s3-cache-datasets/cache_dataset_oracleGeneral/2026-alibaba-thinkahead/c4.oracleGeneral.zst | oracle_general | 0 | 0.002 | 63.952 | 19723 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2026-alibaba-thinkahead/c15.oracleGeneral.zst | oracle_general | 0 | 0 | 63.522 | 3874 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2026-alibaba-thinkahead/c21.oracleGeneral.zst | oracle_general | 0 | 0.004 | 60.426 | 880 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2026-alibaba-thinkahead/c29.oracleGeneral.zst | oracle_general | 0 | 0.041 | 42.436 | 5 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2026-alibaba-thinkahead/c44.oracleGeneral.zst | oracle_general | 0 | 0.035 | 42.436 | 5 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2026-alibaba-thinkahead/c37.oracleGeneral.zst | oracle_general | 0 | 0.04 | 38.382 | 5 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2026-alibaba-thinkahead/c12.oracleGeneral.zst | oracle_general | 0 | 0.002 | 36.932 | 3 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2026-alibaba-thinkahead/c13.oracleGeneral.zst | oracle_general | 0 | 0.001 | 36.932 | 3 |
