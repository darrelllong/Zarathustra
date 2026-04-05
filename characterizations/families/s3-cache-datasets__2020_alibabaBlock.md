# s3-cache-datasets / 2020_alibabaBlock

- Files: 1001
- Bytes: 251207505629
- Formats: oracle_general, text_zst
- Parsers: generic_text, oracle_general
- ML Use Cases: request_sequence, structured_table
- Heterogeneity Score: 1.322
- Suggested GAN Modes: 8
- Split By Format: yes

## Observations

- Predominantly read-heavy.
- Very weak short-window reuse.
- Ordered feature trajectories show regime boundaries.

## GAN Guidance

- Family spans multiple encodings; keep format-aware preprocessing and avoid blindly pooling structured-table and request-sequence variants.
- Ordered PC1 changepoints suggest 22 regimes when files are ordered by trace start time.
- Opcode balance is extremely read-skewed; generation should not assume symmetric read/write behavior.
- Strongest feature coupling in this pass: iat_mean vs iat_q90 (corr=1).
- A small set of files are strong multivariate outliers; consider holding them out for ablation or separate mode inspection.

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| oracle_general | 1000 | oracle_general |
| text_zst | 1 | generic_text |

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
| forward_seek_ratio | backward_seek_ratio | -0.966 |
| obj_size_mean | obj_size_q50 | 0.966 |
| iat_mean | iat_q99 | 0.965 |
| obj_size_std | obj_size_q90 | 0.955 |
| obj_size_mean | obj_size_q90 | 0.95 |

## Metrics

| Metric | Mean | Median | CV | Skew | Kurtosis | Missing | Q10 | Q90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| size_bytes | 250956549 | 12159046 | 19.129 | 30.732 | 960.579 | 0 | 129190 | 118892594 |
| sample_records | 4051.168 | 4096 | 0.095 | -8.936 | 80.937 | 0 | 4096 | 4096 |
| ttl_present | 0 | 0 | N/A | N/A | N/A | 0 | 0 | 0 |
| iat_q90 | 226.127 | 1 | 24.788 | 29.696 | 908.719 | 0.001 | 0 | 6 |
| iat_mean | 74.817 | 0.463 | 21.934 | 29.062 | 880.817 | 0.001 | 0.004 | 4.549 |
| iat_q99 | 554.012 | 5 | 16.56 | 23.861 | 636.374 | 0.001 | 0 | 51.766 |
| iat_std | 189.843 | 1.372 | 14.377 | 24.964 | 697.598 | 0.001 | 0.1 | 16.471 |
| iat_q50 | 0.098 | 0 | 11.743 | 19.894 | 471.833 | 0.001 | 0 | 0 |
| ts_duration | 28093.1 | 1888 | 7.842 | 10.925 | 122.121 | 0.001 | 14 | 18630 |
| iat_lag1_autocorr | -0.022 | -0.009 | 7.7 | 0.44 | 6.978 | 0.001 | -0.171 | 0.1 |
| reuse_ratio | 0.004 | 0 | 6.5 | 18.349 | 389.668 | 0.001 | 0 | 0.007 |
| abs_stride_q50 | 4140612431 | 5566464 | 4.846 | 9.196 | 102.36 | 0.001 | 17305.6 | 4643355034 |
| sample_record_rate | 90.676 | 2.158 | 3.497 | 5.653 | 41.287 | 0.001 | 0.22 | 275.017 |
| obj_size_q50 | 41127.17 | 4096 | 2.753 | 3.095 | 8.121 | 0.001 | 4096 | 98124.8 |
| abs_stride_q99 | 68421707404 | 24548759921 | 2.343 | 11.239 | 190.592 | 0.001 | 13477125915 | 168754881462 |
| abs_stride_q90 | 30087000635 | 11194333594 | 2.339 | 6.558 | 67.002 | 0.001 | 14557594 | 65235863060 |
| abs_stride_mean | 10676758645 | 3774368419 | 2.291 | 5.711 | 43.55 | 0.001 | 1019158465 | 24258202243 |
| abs_stride_std | 17048914605 | 6147959222 | 2.156 | 7.651 | 87.506 | 0.001 | 3094033065 | 35565159045 |

## Outlier Files

| rel_path | outlier_score |
|---|---:|
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_alibabaBlock/1K/alibabaBlock_809.oracleGeneral.zst | 676.741 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_alibabaBlock/alibabaBlock_207.oracleGeneral.zst | 102.674 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_alibabaBlock/1K/alibabaBlock_811.oracleGeneral.zst | 99.494 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_alibabaBlock/1K/alibabaBlock_816.oracleGeneral.zst | 82.477 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_alibabaBlock/alibabaBlock_791.oracleGeneral.zst | 60.919 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_alibabaBlock/alibabaBlock_369.oracleGeneral.zst | 50.513 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_alibabaBlock/alibabaBlock_771.oracleGeneral.zst | 41.674 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_alibabaBlock/10K/alibabaBlock_805.oracleGeneral.zst | 29.994 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv | ts_duration |
|---|---|---:|---:|---:|---:|
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_alibabaBlock/alibabaBlock_4.oracleGeneral.zst | oracle_general | 0 | 0 | 63.984 | 1 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_alibabaBlock/alibabaBlock_810.oracleGeneral.zst | oracle_general | 0 | 0 | 63.899 | 55970 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_alibabaBlock/1M/alibabaBlock_808.oracleGeneral.zst | oracle_general | 0 | 0 | 54.095 | 211798 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_alibabaBlock/10K/alibabaBlock_821.oracleGeneral.zst | oracle_general | 0 | 0.012 | 51.321 | 606253 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_alibabaBlock/10K/alibabaBlock_822.oracleGeneral.zst | oracle_general | 0 | 0 | 45.625 | 956754 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_alibabaBlock/100K/alibabaBlock_831.oracleGeneral.zst | oracle_general | 0 | 0.001 | 45.238 | 2 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_alibabaBlock/100K/alibabaBlock_838.oracleGeneral.zst | oracle_general | 0 | 0.001 | 45.238 | 2 |
| s3-cache-datasets/cache_dataset_oracleGeneral/2020_alibabaBlock/100K/alibabaBlock_839.oracleGeneral.zst | oracle_general | 0 | 0.001 | 45.238 | 2 |
