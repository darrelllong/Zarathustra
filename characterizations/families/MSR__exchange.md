# MSR / exchange

- Files: 96
- Bytes: 1479416826
- Formats: exchange_etw
- Parsers: exchange_etw
- ML Use Cases: request_sequence
- Heterogeneity Score: 0.398
- Suggested GAN Modes: 8
- Split By Format: no

## Observations

- Substantial write pressure across sampled files.
- Very weak short-window reuse.
- Ordered feature trajectories show regime boundaries.

## GAN Guidance

- Ordered PC1 changepoints suggest 8 regimes when files are ordered by trace start time.
- Sequential blocks are much more internally coherent than random file batches; block or curriculum sampling is likely safer than pure iid file sampling.
- Write pressure is material; preserve write bursts and opcode transitions in conditioning.
- Tenant diversity is high; tenant/context conditioning is likely useful.
- Strongest feature coupling in this pass: ts_duration vs iat_mean (corr=1).
- A small set of files are strong multivariate outliers; consider holding them out for ablation or separate mode inspection.
- Current characterization suggests extra conditioning value from: object_unique, signed_stride_lag1_autocorr, obj_size_std.

## Conditioning Audit

| Item | Value |
|---|---|
| Near-constant current conditioning features | tenant_unique |
| Recommended candidate additions | object_unique, signed_stride_lag1_autocorr, obj_size_std |
| Highly redundant current pairs | forward_seek_ratio vs backward_seek_ratio (-0.997) |

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| exchange_etw | 96 | exchange_etw |

## Clustering And Regimes

| Item | Value |
|---|---|
| K-means selected K | 2 |
| Best silhouette K | 2 |
| DBSCAN clusters | 1 |
| DBSCAN noise fraction | 0.042 |
| Ordered PC1 changepoints | 7 |
| PCA variance explained by PC1 | 0.346 |
| Hurst exponent on ordered PC1 | 0.786 |
| Block/random distance ratio | 0.842 |
| Sampling recommendation | block_sampling_preserves_temporal_coherence |

### K Selection

| K | Within-SS | Silhouette |
|---:|---:|---:|
| 2 | 340593544697580116508672 | 0.497 |
| 3 | 203394640854876816408576 | 0.476 |
| 4 | 150101986367924351270912 | 0.481 |
| 5 | 109318100474162311069696 | 0.469 |
| 6 | 78884462175117018923008 | 0.49 |
| 7 | 63061686007813810159616 | 0.495 |
| 8 | 54982145782752600915968 | 0.458 |
| 9 | 49693255828232416002048 | 0.396 |
| 10 | 43096805986950675693568 | 0.402 |
| 11 | 37507104160922218790912 | 0.427 |
| 12 | 33436859146828469239808 | 0.42 |

## Regime Transition Drivers

| Transition | Driver 1 | Effect | Driver 2 | Effect | Driver 3 | Effect |
|---|---|---:|---|---:|---|---:|
| 1 -> 2 | obj_size_q90 | 11.314 | iat_q90 | 3.525 | response_time_q90 | 2.983 |
| 2 -> 3 | response_time_q90 | 7.336 | obj_size_q90 | 5.485 | iat_q90 | 3.578 |
| 3 -> 4 | signed_stride_lag1_autocorr | 2.822 | abs_stride_std | 1.834 | object_top1_share | 1.613 |
| 4 -> 5 | abs_stride_q99 | 8.118 | response_time_mean | 4.686 | signed_stride_lag1_autocorr | 4.106 |
| 5 -> 6 | obj_size_q90 | 2.145 | obj_size_std | 1.633 | abs_stride_q99 | 1.599 |
| 6 -> 7 | iat_zero_ratio | 1.414 | iat_min | 1.414 | forward_seek_ratio | 0.857 |
| 7 -> 8 | iat_q90 | 1.777 | tenant_top1_share | 1.597 | iat_min | 1.414 |

## Strongest Correlations

| Metric A | Metric B | Correlation |
|---|---|---:|
| ts_duration | iat_mean | 1 |
| forward_seek_ratio | backward_seek_ratio | -0.997 |
| iat_std | iat_q99 | 0.989 |
| ts_duration | iat_q99 | 0.982 |
| iat_mean | iat_q99 | 0.982 |
| ts_duration | iat_std | 0.978 |
| iat_mean | iat_std | 0.978 |
| object_top1_share | object_top10_share | 0.974 |

## Metrics

| Metric | Mean | Median | CV | Skew | Kurtosis | Missing | Q10 | Q90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| abs_stride_q50 | 3331750912 | 868892672 | 2.858 | 5.652 | 34.238 | 0 | 67383296 | 5380143104 |
| response_time_q50 | 398.776 | 119.5 | 2.49 | 4.584 | 21.641 | 0 | 66.5 | 581.25 |
| iat_zero_ratio | 0 | 0 | 2.358 | 6.456 | 46.544 | 0 | 0 | 0.001 |
| size_bytes | 15410592 | 11212104 | 1.734 | 6.295 | 40.825 | 0 | 4467314 | 20072056 |
| sample_record_rate | 0.001 | 0.001 | 1.494 | 6.017 | 40.391 | 0 | 0 | 0.001 |
| iat_min | 0.396 | 0 | 1.488 | 1.205 | 0.479 | 0 | 0 | 1 |
| response_time_q99 | 17164.81 | 10125.2 | 1.324 | 5.656 | 39.193 | 0 | 8724.475 | 29427.38 |
| reuse_ratio | 0.003 | 0.003 | 0.953 | 1.613 | 3.639 | 0 | 0 | 0.008 |
| iat_q99 | 55953.15 | 30229.1 | 0.9 | 0.854 | -0.434 | 0 | 7830.09 | 132060.4 |
| iat_std | 12344.99 | 7938.718 | 0.806 | 0.742 | -0.662 | 0 | 1990.603 | 27662.86 |
| iat_lag1_autocorr | 0.154 | 0.124 | 0.796 | 1.945 | 4.483 | 0 | 0.065 | 0.286 |
| response_time_std | 4368.947 | 3154.791 | 0.757 | 3.972 | 21.711 | 0 | 2267.463 | 7264.235 |
| response_time_mean | 1796.731 | 1569.455 | 0.71 | 3.622 | 18.952 | 0 | 806.39 | 2555.934 |
| obj_size_min | 549.333 | 512 | 0.666 | 9.798 | 96 | 0 | 512 | 512 |
| ts_duration | 10113275 | 7736664 | 0.637 | 0.715 | -0.646 | 0 | 3563557 | 19536576 |
| iat_mean | 2469.664 | 1889.295 | 0.637 | 0.715 | -0.646 | 0 | 870.221 | 4770.837 |
| obj_size_q90 | 38592 | 32768 | 0.474 | 0.47 | -1.197 | 0 | 16384 | 65536 |
| object_top1_share | 0.066 | 0.065 | 0.453 | 0.496 | 1.7 | 0 | 0.026 | 0.099 |

## Outlier Files

| rel_path | outlier_score | top drivers |
|---|---:|---|
| MSR/Exchange-Server-Traces/Exchange/Exchange.12-13-2007.03-46-AM.trace.csv.gz | 68.049 | response_time_q50 (z=109.039); abs_stride_q50 (z=91.264) |
| MSR/Exchange-Server-Traces/Exchange/Exchange.12-13-2007.03-31-AM.trace.csv.gz | 28.044 | response_time_q50 (z=117.657); abs_stride_q50 (z=77.024) |
| MSR/Exchange-Server-Traces/Exchange/Exchange.12-13-2007.02-37-PM.trace.csv.gz | 9.58 | response_time_q99 (z=137.213); response_time_std (z=27.946) |
| MSR/Exchange-Server-Traces/Exchange/Exchange.12-13-2007.06-17-AM.trace.csv.gz | 9.481 | obj_size_mean (z=5.53); response_time_q90 (z=-5.419) |
| MSR/Exchange-Server-Traces/Exchange/Exchange.12-13-2007.08-49-AM.trace.csv.gz | 8.615 | response_time_q99 (z=19.479); response_time_q50 (z=14.549) |
| MSR/Exchange-Server-Traces/Exchange/Exchange.12-13-2007.04-01-AM.trace.csv.gz | 6.671 | response_time_q50 (z=81.5); abs_stride_q50 (z=27.209) |
| MSR/Exchange-Server-Traces/Exchange/Exchange.12-13-2007.05-02-AM.trace.csv.gz | 6.505 | response_time_q50 (z=15.902); iat_lag1_autocorr (z=11.116) |
| MSR/Exchange-Server-Traces/Exchange/Exchange.12-13-2007.04-47-AM.trace.csv.gz | 5.9 | response_time_q50 (z=60.569); iat_lag1_autocorr (z=12.422) |

## Outlier Sensitivity

| N Removed | Metric | Baseline Median | Trimmed Median | Relative Shift |
|---:|---|---:|---:|---:|
| 1 | reuse_ratio | 0.003 | 0.003 | 0.048 |
| 3 | reuse_ratio | 0.003 | 0.003 | 0.048 |
| 5 | reuse_ratio | 0.003 | 0.003 | 0.048 |
| 10 | reuse_ratio | 0.003 | 0.003 | 0.048 |
| 10 | burstiness_cv | 4.495 | 4.648 | 0.034 |
| 5 | iat_q50 | 115.5 | 117 | 0.013 |
| 10 | iat_q50 | 115.5 | 116.5 | 0.009 |
| 1 | burstiness_cv | 4.495 | 4.532 | 0.008 |
| 3 | burstiness_cv | 4.495 | 4.532 | 0.008 |
| 5 | burstiness_cv | 4.495 | 4.532 | 0.008 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv | ts_duration |
|---|---|---:|---:|---:|---:|
| MSR/Exchange-Server-Traces/Exchange/Exchange.12-13-2007.12-44-AM.trace.csv.gz | exchange_etw | 0.93 | 0.011 | 7.508 | 15342122 |
| MSR/Exchange-Server-Traces/Exchange/Exchange.12-13-2007.06-17-AM.trace.csv.gz | exchange_etw | 0.911 | 0.002 | 6.974 | 7995869 |
| MSR/Exchange-Server-Traces/Exchange/Exchange.12-13-2007.02-30-AM.trace.csv.gz | exchange_etw | 0.915 | 0.007 | 6.964 | 16728960 |
| MSR/Exchange-Server-Traces/Exchange/Exchange.12-12-2007.09-12-PM.trace.csv.gz | exchange_etw | 0.844 | 0.008 | 6.282 | 11371180 |
| MSR/Exchange-Server-Traces/Exchange/Exchange.12-13-2007.02-45-AM.trace.csv.gz | exchange_etw | 0.923 | 0.01 | 6.277 | 24134214 |
| MSR/Exchange-Server-Traces/Exchange/Exchange.12-13-2007.03-31-AM.trace.csv.gz | exchange_etw | 0.13 | 0 | 6.22 | 639570 |
| MSR/Exchange-Server-Traces/Exchange/Exchange.12-12-2007.09-43-PM.trace.csv.gz | exchange_etw | 0.84 | 0.003 | 6.171 | 15091830 |
| MSR/Exchange-Server-Traces/Exchange/Exchange.12-13-2007.12-29-AM.trace.csv.gz | exchange_etw | 0.89 | 0 | 6.064 | 21284125 |
