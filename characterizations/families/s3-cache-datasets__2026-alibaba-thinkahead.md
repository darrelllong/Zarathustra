# s3-cache-datasets / 2026-alibaba-thinkahead

- Files: 45
- Bytes: 15366738380
- Formats: oracle_general
- Parsers: oracle_general
- ML Use Cases: request_sequence
- Heterogeneity Score: 0.425
- Suggested GAN Modes: 8
- Split By Format: no

## Observations

- Predominantly read-heavy.
- Very weak short-window reuse.
- Highly bursty arrivals.
- Ordered feature trajectories show regime boundaries.

## GAN Guidance

- Ordered PC1 changepoints suggest 8 regimes when files are ordered by trace start time.
- Opcode balance is extremely read-skewed; generation should not assume symmetric read/write behavior.
- Burstiness is high; inter-arrival and FFT/ACF losses should stay heavily weighted.
- Strongest feature coupling in this pass: ts_duration vs iat_mean (corr=1).
- A small set of files are strong multivariate outliers; consider holding them out for ablation or separate mode inspection.

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| oracle_general | 45 | oracle_general |

## Clustering And Regimes

| Item | Value |
|---|---|
| DBSCAN clusters | 1 |
| DBSCAN noise fraction | 0.089 |
| Ordered PC1 changepoints | 7 |
| PCA variance explained by PC1 | 0.238 |

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
| abs_stride_q50 | 4444443745 | 176 | 6.708 | 6.708 | 45 | 0 | 16 | 3012.6 |
| iat_std | 8.521 | 0.052 | 5.464 | 6.356 | 41.42 | 0 | 0.034 | 0.096 |
| ts_duration | 553.2 | 9 | 5.389 | 6.351 | 41.37 | 0 | 4.4 | 27.2 |
| iat_mean | 0.135 | 0.002 | 5.389 | 6.351 | 41.37 | 0 | 0.001 | 0.007 |
| abs_stride_q90 | 15557719215 | 1822152 | 4.933 | 6.039 | 38.235 | 0 | 911090.2 | 4697574 |
| abs_stride_mean | 6066488377 | 674742.4 | 4.93 | 6.227 | 40.205 | 0 | 395213.7 | 2364311534 |
| abs_stride_q99 | 24462338736 | 8277304 | 3.913 | 5.337 | 31.172 | 0 | 4699519 | 60082168770 |
| size_bytes | 341483075 | 6440531 | 3.535 | 4.841 | 25.58 | 0 | 901735.8 | 573440809 |
| abs_stride_std | 10718826025 | 2202018 | 3.33 | 4.171 | 18.386 | 0 | 1097926 | 20869063973 |
| iat_lag1_autocorr | 0.022 | -0.001 | 3.216 | 3.125 | 8.809 | 0 | -0.002 | 0.072 |
| reuse_ratio | 0.027 | 0.027 | 1.053 | 1.623 | 4.16 | 0 | 0.001 | 0.062 |
| object_top10_share | 0.028 | 0.021 | 0.823 | 2.433 | 5.703 | 0 | 0.014 | 0.061 |
| object_top1_share | 0.006 | 0.004 | 0.755 | 1.716 | 2.833 | 0 | 0.002 | 0.011 |
| sample_record_rate | 536.649 | 455.111 | 0.628 | 0.787 | 0.667 | 0 | 150.787 | 942.08 |
| signed_stride_lag1_autocorr | -0.357 | -0.393 | 0.425 | 0.783 | 0.844 | 0 | -0.491 | -0.139 |
| burstiness_cv | 29.031 | 26.106 | 0.403 | 1.509 | 2.642 | 0 | 18.244 | 40.814 |
| backward_seek_ratio | 0.272 | 0.306 | 0.33 | -0.833 | 0.257 | 0 | 0.16 | 0.371 |
| object_unique | 3492.911 | 3614 | 0.203 | -3.567 | 13.797 | 0 | 3250.8 | 3955.4 |

## Outlier Files

| rel_path | outlier_score |
|---|---:|
| s3-cache-datasets/cache_dataset_oracleGeneral/2026-alibaba-thinkahead/c17.oracleGeneral.zst | 36.679 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2026-alibaba-thinkahead/c4.oracleGeneral.zst | 35.349 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2026-alibaba-thinkahead/c25.oracleGeneral.zst | 15.343 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2026-alibaba-thinkahead/c15.oracleGeneral.zst | 12.976 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2026-alibaba-thinkahead/c14.oracleGeneral.zst | 10.336 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2026-alibaba-thinkahead/c3.oracleGeneral.zst | 2.192 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2026-alibaba-thinkahead/c26.oracleGeneral.zst | 1.982 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2026-alibaba-thinkahead/c21.oracleGeneral.zst | 1.958 |

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
