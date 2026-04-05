# s3-cache-datasets / 2020_twitter

- Files: 54
- Bytes: 152248401229
- Formats: oracle_general
- Parsers: oracle_general
- ML Use Cases: request_sequence
- Heterogeneity Score: 0.634
- Suggested GAN Modes: 1
- Split By Format: no

## Observations

- Predominantly read-heavy.
- Highly bursty arrivals.

## GAN Guidance

- Opcode balance is extremely read-skewed; generation should not assume symmetric read/write behavior.
- Burstiness is high; inter-arrival and FFT/ACF losses should stay heavily weighted.
- Strongest feature coupling in this pass: ts_duration vs iat_zero_ratio (corr=-1).
- A small set of files are strong multivariate outliers; consider holding them out for ablation or separate mode inspection.

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| oracle_general | 54 | oracle_general |

## Clustering And Regimes

| Item | Value |
|---|---|
| DBSCAN clusters | 1 |
| DBSCAN noise fraction | 0.074 |
| Ordered PC1 changepoints | 0 |
| PCA variance explained by PC1 | 0.292 |

## Strongest Correlations

| Metric A | Metric B | Correlation |
|---|---|---:|
| ts_duration | iat_zero_ratio | -1 |
| ts_duration | iat_mean | 1 |
| iat_zero_ratio | iat_mean | -1 |
| ts_duration | iat_lag1_autocorr | -1 |
| iat_zero_ratio | iat_lag1_autocorr | 1 |
| iat_lag1_autocorr | iat_mean | -1 |
| reuse_ratio | forward_seek_ratio | -0.998 |
| reuse_ratio | backward_seek_ratio | -0.998 |

## Metrics

| Metric | Mean | Median | CV | Skew | Kurtosis | Missing | Q10 | Q90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| obj_size_q50 | 1213.102 | 124 | 3.511 | 4.77 | 23.58 | 0 | 44.6 | 952.4 |
| obj_size_q90 | 5572.667 | 506 | 3.2 | 4.61 | 23.236 | 0 | 67.4 | 4860.45 |
| obj_size_mean | 2169.323 | 337.486 | 2.984 | 4.342 | 19.867 | 0 | 56.318 | 2287.172 |
| iat_q99 | 0.131 | 0 | 2.591 | 2.266 | 3.26 | 0 | 0 | 1 |
| obj_size_std | 2902.36 | 204.103 | 2.477 | 3.755 | 15.608 | 0 | 16.849 | 7082.353 |
| obj_size_q99 | 11552.8 | 972.65 | 2.179 | 2.972 | 8.82 | 0 | 97 | 35029.43 |
| object_top1_share | 0.043 | 0.014 | 2.06 | 4.194 | 21.022 | 0 | 0.001 | 0.095 |
| reuse_ratio | 0.099 | 0.037 | 1.654 | 2.608 | 7.001 | 0 | 0.001 | 0.343 |
| iat_lag1_autocorr | -0.006 | -0.003 | 1.521 | -2.928 | 8.354 | 0 | -0.013 | -0.001 |
| ts_duration | 25.463 | 12 | 1.492 | 2.887 | 8.135 | 0 | 4 | 53.5 |
| iat_mean | 0.006 | 0.003 | 1.492 | 2.887 | 8.135 | 0 | 0.001 | 0.013 |
| object_top10_share | 0.124 | 0.064 | 1.277 | 2.827 | 11.796 | 0 | 0.005 | 0.295 |
| sample_record_rate | 495.862 | 341.333 | 0.933 | 1.654 | 3.102 | 0 | 78.011 | 1024 |
| size_bytes | 2819414838 | 1949975838 | 0.928 | 1.877 | 4.513 | 0 | 420565252 | 6285663942 |
| obj_size_min | 33.593 | 29 | 0.722 | 2.638 | 10.202 | 0 | 12.3 | 52.6 |
| iat_std | 0.066 | 0.054 | 0.634 | 1.816 | 3.291 | 0 | 0.031 | 0.113 |
| object_unique | 2020.074 | 1859 | 0.512 | 0.438 | -0.441 | 0 | 1061.4 | 3729.7 |
| burstiness_cv | 20.019 | 18.446 | 0.489 | 0.599 | 0.013 | 0 | 8.752 | 31.98 |

## Outlier Files

| rel_path | outlier_score |
|---|---:|
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_twitter/cluster50.oracleGeneral.sample10.zst | 29.948 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_twitter/cluster10.oracleGeneral.sample10.zst | 16.601 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_twitter/cluster35.oracleGeneral.sample10.zst | 16.008 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_twitter/cluster2.oracleGeneral.sample10.zst | 11.133 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_twitter/cluster25.oracleGeneral.sample10.zst | 10.257 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_twitter/cluster49.oracleGeneral.sample10.zst | 9.791 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_twitter/cluster8.oracleGeneral.sample10.zst | 9.633 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_twitter/cluster26.oracleGeneral.sample10.zst | 7.649 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv | ts_duration |
|---|---|---:|---:|---:|---:|
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_twitter/cluster18.oracleGeneral.sample10.zst | oracle_general | 0 | 0.07 | 45.238 | 2 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_twitter/cluster52.oracleGeneral.sample10.zst | oracle_general | 0 | 0.053 | 45.238 | 2 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_twitter/cluster1.oracleGeneral.sample10.zst | oracle_general | 0 | 0.043 | 36.932 | 3 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_twitter/cluster43.oracleGeneral.sample10.zst | oracle_general | 0 | 0 | 36.932 | 3 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_twitter/cluster25.oracleGeneral.sample10.zst | oracle_general | 0 | 0.425 | 31.98 | 4 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_twitter/cluster27.oracleGeneral.sample10.zst | oracle_general | 0 | 0.05 | 31.98 | 4 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_twitter/cluster46.oracleGeneral.sample10.zst | oracle_general | 0 | 0.006 | 31.98 | 4 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_twitter/cluster16.oracleGeneral.sample10.zst | oracle_general | 0 | 0.104 | 31.98 | 4 |
