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
- Sequential blocks are much more internally coherent than random file batches; block or curriculum sampling is likely safer than pure iid file sampling.
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
| DBSCAN clusters | 1 |
| DBSCAN noise fraction | 0.089 |
| Ordered PC1 changepoints | 7 |
| PCA variance explained by PC1 | 0.238 |
| Hurst exponent on ordered PC1 | 0.5 |
| Block/random distance ratio | 0.728 |
| Sampling recommendation | block_sampling_preserves_temporal_coherence |

### K Selection

| K | Within-SS | Silhouette |
|---:|---:|---:|
| 2 | 106776690978223506325504 | 0.938 |
| 3 | 35431607872077827145728 | 0.87 |
| 4 | 19829828452379996979200 | 0.882 |
| 5 | 19781653324182675521536 | 0.795 |
| 6 | 5254002674706204852224 | 0.822 |
| 7 | 5193513253169503666176 | 0.782 |
| 8 | 5202476924867592585216 | 0.81 |
| 9 | 5202440609969667571712 | 0.684 |
| 10 | 5202420337260752797696 | 0.628 |
| 11 | 63527868641057570816 | 0.619 |
| 12 | 5202399732524923224064 | 0.626 |

## Regime Transition Drivers

| Transition | Driver 1 | Effect | Driver 2 | Effect | Driver 3 | Effect |
|---|---|---:|---|---:|---|---:|
| 1 -> 2 | abs_stride_std | 1.414 | abs_stride_q99 | 1.414 | abs_stride_mean | 1.414 |
| 2 -> 3 | abs_stride_q99 | 6.153 | signed_stride_lag1_autocorr | 4.508 | abs_stride_q90 | 4.296 |
| 3 -> 4 | reuse_ratio | 7.06 | object_unique | 5.4 | abs_stride_q99 | 3.592 |
| 4 -> 5 | reuse_ratio | 2.852 | object_unique | 2.124 | abs_stride_q90 | 1.966 |
| 5 -> 6 | backward_seek_ratio | 14.842 | size_bytes | 4.113 | forward_seek_ratio | 3.71 |
| 6 -> 7 | forward_seek_ratio | 5.427 | backward_seek_ratio | 4.262 | reuse_ratio | 3.275 |
| 7 -> 8 | reuse_ratio | 5.413 | iat_std | 1.905 | sample_record_rate | 1.732 |

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




## Model-Aware Guidance

- Closest learned anchor: tencent_block (distance 1.894)
- Sampling: block
- Regime recipe: K≈8
- Char-file conditioning: yes
- PCF: validated
- Multi-scale critic: promising
- Mixed-type recovery: promising
- Retrieval memory: mixed
- Why: ordered files show temporal persistence; family looks multi-regime or high-heterogeneity; burstiness is materially above the calmer families
- Candidate conditioning additions: object_unique,signed_stride_lag1_autocorr
