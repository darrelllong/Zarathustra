# tencent / cloud_disk

- Files: 16296
- Bytes: 7270989623
- Formats: tencent_cloud_disk
- Parsers: tencent_cloud_disk
- ML Use Cases: aggregate_time_series
- Heterogeneity Score: 5.572
- Suggested GAN Modes: 8
- Split By Format: no

## Observations

- Ordered feature trajectories show regime boundaries.

## GAN Guidance

- High cross-file heterogeneity; favor regime conditioning or multiple family-specific GAN runs over a single unconditional model.
- Ordered PC1 changepoints suggest 666 regimes when files are ordered by trace start time.
- Strongest feature coupling in this pass: disk_usage_mean vs disk_usage_q50 (corr=0.99).
- A small set of files are strong multivariate outliers; consider holding them out for ablation or separate mode inspection.

## Conditioning Audit

| Item | Value |
|---|---|
| Near-constant current conditioning features | none flagged |
| Recommended candidate additions | none flagged |
| Highly redundant current pairs | none flagged |

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| tencent_cloud_disk | 16296 | tencent_cloud_disk |

## Clustering And Regimes

| Item | Value |
|---|---|
| K-means selected K | 2 |
| Best silhouette K | 2 |
| DBSCAN clusters | 4 |
| DBSCAN noise fraction | 0.069 |
| Ordered PC1 changepoints | 665 |
| PCA variance explained by PC1 | 0.343 |
| Hurst exponent on ordered PC1 | 0.5 |
| Block/random distance ratio | 0.899 |
| Sampling recommendation | random_sampling_is_less_problematic |

### K Selection

| K | Within-SS | Silhouette |
|---:|---:|---:|
| 2 | 147653035443648 | 0.989 |
| 3 | 137693618579900 | 0.953 |
| 4 | 133518188766394 | 0.547 |
| 5 | 129695423268484 | 0.523 |
| 6 | 18630991471181 | 0.535 |
| 7 | 13271660964015 | 0.519 |
| 8 | 11835303531926 | 0.55 |
| 9 | 10956997592229 | 0.523 |
| 10 | 9855285008859 | 0.517 |
| 11 | 9198232839002 | 0.439 |
| 12 | 8752816940199 | 0.427 |

## Regime Transition Drivers

| Transition | Driver 1 | Effect | Driver 2 | Effect | Driver 3 | Effect |
|---|---|---:|---|---:|---|---:|
| 1 -> 2 | disk_usage_q50 | 4.196 | disk_usage_q90 | 3.803 | disk_usage_mean | 3.572 |
| 2 -> 3 | total_iops_q90 | 4.101 | read_iops_q90 | 3.406 | write_iops_q90 | 3.276 |
| 3 -> 4 | total_iops_q90 | 0.695 | read_iops_q90 | 0.677 | size_bytes | 0.541 |
| 4 -> 5 | read_bw_q50 | 0.851 | idle_ratio | 0.779 | write_iops_mean | 0.749 |
| 5 -> 6 | total_iops_lag1_autocorr | 1.792 | disk_usage_q50 | 1.414 | disk_usage_mean | 1.408 |
| 6 -> 7 | total_iops_lag1_autocorr | 2.732 | total_bw_q90 | 2.292 | total_bw_q50 | 1.795 |
| 7 -> 8 | read_iops_q90 | 5.607 | write_iops_mean | 4.98 | total_iops_mean | 3.988 |
| 8 -> 9 | read_iops_q90 | 11.49 | write_iops_mean | 4.323 | total_iops_q90 | 4.139 |
| 9 -> 10 | total_bw_mean | 2.472 | write_bw_mean | 1.832 | write_iops_q90 | 1.499 |
| 10 -> 11 | ts_duration | 2.789 | total_bw_mean | 1.755 | write_bw_mean | 1.722 |
| 11 -> 12 | ts_duration | 2.007 | write_iops_mean | 1.168 | total_bw_q50 | 1.079 |
| 12 -> 13 | disk_usage_q90 | 0.868 | disk_usage_mean | 0.784 | disk_usage_q50 | 0.784 |
| 13 -> 14 | write_iops_q90 | 1.029 | total_iops_q90 | 0.979 | write_share_iops_mean | 0.968 |
| 14 -> 15 | read_iops_q90 | 1.414 | total_iops_q90 | 1.249 | write_bw_q50 | 1.109 |
| 15 -> 16 | total_bw_q50 | 2.454 | read_bw_q50 | 2.259 | read_bw_mean | 2.22 |
| 16 -> 17 | total_bw_q50 | 1.944 | read_bw_q50 | 1.808 | write_bw_mean | 1.8 |
| 17 -> 18 | idle_ratio | 3.274 | total_iops_lag1_autocorr | 1.844 | total_iops_q90 | 1.696 |
| 18 -> 19 | idle_ratio | 2.665 | write_share_iops_mean | 2.075 | read_iops_q90 | 1.585 |
| 19 -> 20 | write_bw_q50 | 1.222 | read_iops_q90 | 1.084 | read_bw_q90 | 0.996 |
| 20 -> 21 | read_iops_q90 | 1.001 | idle_ratio | 0.999 | total_bw_q90 | 0.91 |
| 21 -> 22 | read_bw_mean | 3.621 | read_bw_q90 | 1.896 | write_bw_q50 | 1.637 |
| 22 -> 23 | read_bw_mean | 2.755 | disk_usage_q50 | 1.657 | disk_usage_q90 | 1.657 |
| 23 -> 24 | disk_usage_q50 | 2.807 | disk_usage_q90 | 2.572 | disk_usage_mean | 2.412 |
| 24 -> 25 | write_bw_q90 | 2.416 | total_bw_q90 | 1.198 | disk_usage_mean | 0.953 |
| 25 -> 26 | read_bw_q90 | 1.647 | read_bw_mean | 1.532 | total_bw_q90 | 1.478 |
| 26 -> 27 | write_bw_mean | 1.447 | write_bw_q90 | 1.43 | total_bw_q90 | 1.413 |
| 27 -> 28 | read_iops_q99 | 1.107 | write_share_iops_mean | 1.011 | write_bw_mean | 0.789 |
| 28 -> 29 | write_bw_q90 | 1.022 | total_iops_lag1_autocorr | 0.862 | read_bw_mean | 0.674 |
| 29 -> 30 | total_bw_q50 | 1.198 | read_iops_q90 | 1.189 | write_bw_q90 | 1.159 |
| 30 -> 31 | size_bytes | 1.435 | read_iops_q90 | 1.189 | write_bw_q90 | 1.137 |
| 31 -> 32 | write_share_iops_mean | 2.52 | idle_ratio | 1.566 | total_iops_q90 | 1.414 |
| 32 -> 33 | write_share_iops_mean | 1.536 | disk_usage_mean | 1.42 | read_iops_q90 | 1.414 |
| 33 -> 34 | ts_duration | 3.093 | read_iops_q50 | 1.414 | write_iops_q50 | 1.414 |
| 34 -> 35 | ts_duration | 2.806 | read_iops_q50 | 1.414 | write_iops_q50 | 1.414 |
| 35 -> 36 | idle_ratio | 1.834 | read_bw_mean | 1.69 | read_bw_q90 | 1.686 |
| 36 -> 37 | read_bw_q90 | 2.063 | read_bw_mean | 1.883 | read_bw_q50 | 1.712 |
| 37 -> 38 | read_iops_q90 | 1.414 | write_iops_q90 | 1.414 | total_iops_q90 | 1.414 |
| 38 -> 39 | write_iops_mean | 1.283 | read_iops_q90 | 1.029 | total_iops_mean | 0.984 |
| 39 -> 40 | total_iops_lag1_autocorr | 1.847 | disk_usage_q90 | 1.509 | read_iops_q90 | 1.189 |
| 40 -> 41 | total_iops_lag1_autocorr | 1.429 | disk_usage_q90 | 1.264 | write_bw_mean | 0.937 |
| 41 -> 42 | write_bw_q50 | 2.098 | write_share_iops_mean | 1.694 | total_bw_q50 | 1.588 |
| 42 -> 43 | write_share_iops_mean | 2.48 | ts_duration | 1.414 | read_iops_mean | 1.342 |
| 43 -> 44 | size_bytes | 1.064 | write_iops_mean | 0.936 | total_iops_mean | 0.907 |
| 44 -> 45 | write_share_iops_mean | 1.572 | write_bw_q50 | 1.278 | total_iops_lag1_autocorr | 0.964 |
| 45 -> 46 | write_share_iops_mean | 4.101 | write_bw_q50 | 2.78 | ts_duration | 1.947 |
| 46 -> 47 | write_bw_q50 | 2.304 | total_iops_lag1_autocorr | 2.269 | ts_duration | 2.158 |
| 47 -> 48 | total_iops_lag1_autocorr | 1.491 | write_iops_q90 | 1.414 | total_iops_q90 | 1.414 |
| 48 -> 49 | read_iops_q99 | 1.606 | total_iops_q90 | 1.414 | read_iops_q90 | 1.414 |
| 49 -> 50 | write_iops_q90 | 1.228 | total_iops_q90 | 1.2 | read_iops_q90 | 1.029 |
| 50 -> 51 | idle_ratio | 1.473 | total_iops_q90 | 1.389 | write_iops_q90 | 1.363 |
| 51 -> 52 | total_iops_lag1_autocorr | 2.135 | write_iops_q99 | 1.349 | total_iops_q99 | 1.326 |
| 52 -> 53 | total_iops_lag1_autocorr | 1.427 | total_iops_q99 | 1.359 | read_iops_mean | 1.359 |
| 53 -> 54 | ts_duration | 2.834 | total_bw_mean | 1.333 | size_bytes | 1.09 |
| 54 -> 55 | ts_duration | 2.834 | read_iops_q90 | 1.029 | total_iops_q90 | 0.981 |
| 55 -> 56 | disk_usage_q90 | 2.696 | disk_usage_q50 | 1.836 | disk_usage_mean | 1.571 |
| 56 -> 57 | disk_usage_q90 | 2.562 | disk_usage_q50 | 1.678 | disk_usage_mean | 1.437 |
| 57 -> 58 | size_bytes | 1.632 | total_bw_q50 | 1.247 | read_bw_q90 | 1.157 |
| 58 -> 59 | size_bytes | 2.053 | total_bw_q50 | 1.4 | disk_usage_q90 | 1.39 |
| 59 -> 60 | read_iops_q90 | 4.346 | ts_duration | 3.128 | total_iops_q90 | 2.839 |
| 60 -> 61 | read_iops_q90 | 4.048 | ts_duration | 2.974 | total_iops_q90 | 2.701 |
| 61 -> 62 | read_iops_q90 | 1.414 | write_iops_q90 | 1.414 | total_iops_q90 | 1.414 |
| 62 -> 63 | read_iops_q90 | 1.414 | write_iops_q90 | 1.414 | total_iops_q90 | 1.414 |
| 63 -> 64 | write_bw_q90 | 1.942 | total_bw_q90 | 1.533 | read_bw_q90 | 1.501 |
| 64 -> 65 | idle_ratio | 1.567 | write_bw_q90 | 1.496 | total_bw_q90 | 1.485 |
| 65 -> 66 | size_bytes | 1.678 | write_share_iops_mean | 1.184 | write_bw_mean | 1.133 |
| 66 -> 67 | size_bytes | 1.733 | read_iops_q90 | 1.376 | total_iops_q90 | 1.376 |
| 67 -> 68 | disk_usage_q90 | 1.257 | total_bw_q90 | 1.092 | disk_usage_mean | 1.074 |
| 68 -> 69 | disk_usage_q90 | 0.951 | total_bw_q50 | 0.858 | size_bytes | 0.82 |
| 69 -> 70 | total_bw_q50 | 1.312 | write_bw_q50 | 1.078 | read_bw_q50 | 0.889 |
| 70 -> 71 | read_bw_q50 | 0.995 | read_iops_q90 | 0.923 | write_iops_q99 | 0.886 |
| 71 -> 72 | ts_duration | 1.414 | read_bw_q50 | 1.33 | write_share_iops_mean | 1.072 |
| 72 -> 73 | ts_duration | 1.414 | disk_usage_q90 | 0.844 | total_iops_q90 | 0.84 |
| 73 -> 74 | read_bw_q90 | 1.208 | total_bw_q90 | 0.856 | read_iops_q99 | 0.846 |
| 74 -> 75 | write_iops_q50 | 1.414 | total_iops_q50 | 1.414 | total_bw_mean | 1.307 |
| 75 -> 76 | write_iops_q50 | 1.414 | write_bw_mean | 1.231 | total_bw_mean | 0.947 |
| 76 -> 77 | write_bw_q50 | 1.521 | total_iops_q50 | 1.414 | write_bw_q90 | 1.134 |
| 77 -> 78 | read_bw_q50 | 4.204 | total_bw_q50 | 2.991 | read_bw_mean | 2.654 |
| 78 -> 79 | read_bw_q90 | 4.526 | total_bw_q50 | 4.083 | total_bw_q90 | 3.518 |
| 79 -> 80 | read_iops_q90 | 1.927 | write_iops_mean | 1.705 | write_iops_q99 | 1.702 |
| 80 -> 81 | write_bw_q90 | 1.536 | disk_usage_mean | 1.248 | write_iops_mean | 1.161 |
| 81 -> 82 | write_bw_q90 | 1.043 | read_bw_q50 | 1.007 | write_iops_q99 | 0.879 |
| 82 -> 83 | total_iops_lag1_autocorr | 1.329 | idle_ratio | 1.313 | read_iops_q90 | 1.129 |
| 83 -> 84 | idle_ratio | 1.016 | total_bw_q90 | 0.643 | disk_usage_q90 | 0.614 |
| 84 -> 85 | read_bw_q90 | 1.11 | write_bw_q50 | 0.846 | idle_ratio | 0.806 |
| 85 -> 86 | read_iops_q90 | 0.788 | idle_ratio | 0.744 | write_bw_q50 | 0.739 |
| 86 -> 87 | disk_usage_mean | 2.321 | disk_usage_q50 | 2.223 | disk_usage_q90 | 1.985 |
| 87 -> 88 | write_bw_q50 | 3.469 | disk_usage_q90 | 3.469 | disk_usage_mean | 3.259 |
| 88 -> 89 | read_bw_q50 | 1.24 | write_share_iops_mean | 1.233 | size_bytes | 1.209 |
| 89 -> 90 | ts_duration | 3.006 | write_bw_mean | 1.759 | total_bw_mean | 1.569 |
| 90 -> 91 | total_bw_mean | 1.4 | read_bw_q90 | 1.273 | total_bw_q90 | 1.101 |
| 91 -> 92 | total_iops_q99 | 0.939 | write_iops_q99 | 0.938 | write_share_iops_mean | 0.748 |
| 92 -> 93 | read_iops_q90 | 0.828 | total_iops_q90 | 0.788 | disk_usage_mean | 0.734 |
| 93 -> 94 | total_bw_q90 | 0.962 | total_bw_q50 | 0.768 | write_bw_q50 | 0.624 |
| 94 -> 95 | ts_duration | 2 | total_iops_lag1_autocorr | 0.935 | size_bytes | 0.882 |
| 95 -> 96 | ts_duration | 1.99 | disk_usage_mean | 1.372 | disk_usage_q50 | 1.37 |
| 96 -> 97 | write_iops_q90 | 1.414 | total_iops_q90 | 1.414 | read_iops_q90 | 1.414 |
| 97 -> 98 | read_iops_q90 | 1.414 | total_iops_q90 | 1.414 | write_iops_q90 | 1.414 |
| 98 -> 99 | read_bw_q50 | 1.61 | ts_duration | 1.414 | read_bw_mean | 1.393 |
| 99 -> 100 | ts_duration | 1.414 | read_iops_q90 | 1.109 | total_iops_lag1_autocorr | 1.031 |
| 100 -> 101 | total_iops_q99 | 2.902 | read_iops_mean | 2.228 | write_iops_q99 | 2.059 |
| 101 -> 102 | read_iops_mean | 1.926 | write_share_iops_mean | 1.418 | total_iops_q99 | 1.27 |
| 102 -> 103 | write_iops_q90 | 1.414 | read_iops_q90 | 1.414 | total_iops_q90 | 1.414 |
| 103 -> 104 | total_bw_mean | 1.465 | total_bw_q90 | 1.451 | ts_duration | 1.414 |
| 104 -> 105 | ts_duration | 1.414 | read_bw_q90 | 1.328 | total_bw_q90 | 1.16 |
| 105 -> 106 | write_bw_q50 | 1.084 | read_iops_q90 | 1.029 | total_bw_q50 | 0.719 |
| 106 -> 107 | size_bytes | 3.52 | write_share_iops_mean | 1.959 | disk_usage_mean | 1.911 |
| 107 -> 108 | read_iops_q90 | 2.982 | size_bytes | 2.725 | total_iops_q90 | 2.376 |
| 108 -> 109 | write_bw_mean | 1.789 | total_bw_mean | 1.542 | write_iops_q50 | 1.414 |
| 109 -> 110 | write_bw_q50 | 1.961 | write_bw_q90 | 1.93 | write_bw_mean | 1.919 |
| 110 -> 111 | read_bw_q90 | 0.875 | total_bw_q90 | 0.774 | read_bw_q50 | 0.7 |
| 111 -> 112 | read_bw_q90 | 0.855 | write_iops_q90 | 0.743 | read_bw_mean | 0.722 |
| 112 -> 113 | ts_duration | 1.414 | write_share_iops_mean | 0.571 | total_iops_lag1_autocorr | 0.56 |
| 113 -> 114 | total_iops_lag1_autocorr | 1.527 | ts_duration | 1.414 | read_iops_q90 | 0.856 |
| 114 -> 115 | total_iops_lag1_autocorr | 1.56 | ts_duration | 1.414 | disk_usage_q50 | 1.173 |
| 115 -> 116 | ts_duration | 1.414 | total_iops_lag1_autocorr | 1.056 | write_bw_q50 | 0.835 |
| 116 -> 117 | total_iops_q90 | 4.947 | read_iops_q90 | 4.469 | write_iops_q90 | 4 |
| 117 -> 118 | read_iops_q90 | 7.276 | total_iops_q90 | 4.947 | write_iops_q90 | 3.313 |
| 118 -> 119 | idle_ratio | 2.443 | write_share_iops_mean | 2.344 | total_iops_mean | 1.954 |
| 119 -> 120 | write_share_iops_mean | 2.052 | ts_duration | 1.899 | idle_ratio | 1.735 |
| 120 -> 121 | write_bw_q50 | 2.239 | read_bw_q50 | 1.9 | write_share_iops_mean | 1.887 |
| 121 -> 122 | total_bw_q50 | 1.481 | read_bw_q50 | 1.471 | read_iops_q99 | 1.451 |
| 122 -> 123 | read_bw_q50 | 1.628 | total_bw_q50 | 1.589 | disk_usage_q50 | 1.523 |
| 123 -> 124 | write_bw_q50 | 2.101 | total_bw_q50 | 2.042 | read_bw_mean | 1.863 |
| 124 -> 125 | write_iops_q90 | 7.071 | read_iops_q90 | 5.657 | read_iops_q99 | 4.181 |
| 125 -> 126 | read_bw_q90 | 3.436 | read_bw_q50 | 3.064 | size_bytes | 2.167 |
| 126 -> 127 | total_bw_q90 | 1.993 | write_bw_q50 | 1.499 | total_bw_q50 | 1.12 |
| 127 -> 128 | total_bw_q90 | 0.881 | total_iops_lag1_autocorr | 0.79 | read_iops_q90 | 0.788 |
| 128 -> 129 | ts_duration | 1.414 | write_share_iops_mean | 0.929 | total_bw_q50 | 0.882 |
| 129 -> 130 | ts_duration | 1.414 | read_iops_q90 | 1.414 | read_bw_q90 | 0.888 |
| 130 -> 131 | size_bytes | 2.732 | ts_duration | 1.877 | disk_usage_mean | 1.876 |
| 131 -> 132 | size_bytes | 2.754 | disk_usage_mean | 2.206 | disk_usage_q50 | 2.035 |
| 132 -> 133 | write_iops_q99 | 16.41 | total_iops_mean | 14.069 | read_iops_mean | 10.381 |
| 133 -> 134 | write_iops_q99 | 11.758 | total_iops_mean | 11.147 | read_iops_mean | 9.115 |
| 134 -> 135 | idle_ratio | 2.061 | write_bw_mean | 2.003 | total_iops_q99 | 1.478 |
| 135 -> 136 | write_bw_mean | 5.367 | total_bw_mean | 2.939 | idle_ratio | 2.647 |
| 136 -> 137 | ts_duration | 1.414 | write_iops_q90 | 0.948 | write_bw_q50 | 0.905 |
| 137 -> 138 | ts_duration | 1.414 | write_iops_q99 | 1.331 | disk_usage_q50 | 1.227 |
| 138 -> 139 | read_bw_q90 | 2.176 | total_iops_lag1_autocorr | 1.819 | read_iops_mean | 1.659 |
| 139 -> 140 | write_iops_q90 | 1.414 | total_iops_q90 | 1.414 | write_bw_q90 | 0.944 |
| 140 -> 141 | disk_usage_q90 | 1.649 | read_iops_mean | 1.632 | read_iops_q90 | 1.414 |
| 141 -> 142 | total_iops_lag1_autocorr | 1.272 | read_bw_q50 | 1.213 | write_bw_q90 | 1.191 |
| 142 -> 143 | write_bw_q50 | 0.973 | write_bw_q90 | 0.703 | total_bw_q50 | 0.693 |
| 143 -> 144 | write_share_iops_mean | 2.361 | size_bytes | 2.208 | total_iops_q90 | 1.438 |
| 144 -> 145 | write_share_iops_mean | 3.621 | ts_duration | 1.414 | write_bw_q50 | 1.289 |
| 145 -> 146 | total_iops_lag1_autocorr | 2.262 | idle_ratio | 1.706 | read_iops_q50 | 1.414 |
| 146 -> 147 | total_iops_lag1_autocorr | 2.389 | idle_ratio | 1.985 | size_bytes | 1.894 |
| 147 -> 148 | write_bw_q50 | 2.596 | write_share_iops_mean | 1.998 | total_iops_q99 | 1.342 |
| 148 -> 149 | read_iops_q90 | 1.109 | write_share_iops_mean | 0.803 | read_bw_q50 | 0.69 |
| 149 -> 150 | write_iops_q90 | 1.245 | read_iops_q90 | 1.106 | total_iops_q90 | 1.066 |
| 150 -> 151 | read_bw_q50 | 1.299 | write_bw_q50 | 1.122 | read_iops_q90 | 1.026 |
| 151 -> 152 | total_bw_mean | 2.943 | write_bw_q90 | 2.538 | write_bw_mean | 2.204 |
| 152 -> 153 | total_bw_mean | 2.83 | write_bw_q90 | 2.67 | write_bw_mean | 2.32 |
| 153 -> 154 | write_bw_q90 | 0.839 | write_bw_q50 | 0.605 | total_iops_q90 | 0.589 |
| 154 -> 155 | write_iops_q90 | 0.894 | write_iops_q99 | 0.82 | write_bw_mean | 0.807 |
| 155 -> 156 | total_iops_lag1_autocorr | 1.728 | read_iops_q99 | 1.661 | total_iops_q99 | 1.507 |
| 156 -> 157 | idle_ratio | 3.312 | read_iops_q99 | 2.012 | total_iops_q99 | 1.684 |
| 157 -> 158 | total_iops_q90 | 1.414 | read_iops_q90 | 1.414 | write_iops_q90 | 1.414 |
| 158 -> 159 | total_iops_q90 | 2.379 | total_iops_lag1_autocorr | 2.352 | write_iops_q90 | 1.949 |
| 159 -> 160 | total_bw_q90 | 1.918 | write_bw_q90 | 1.721 | read_bw_q90 | 1.408 |
| 160 -> 161 | total_iops_lag1_autocorr | 3.234 | disk_usage_mean | 2.192 | disk_usage_q50 | 2.043 |
| 161 -> 162 | total_iops_lag1_autocorr | 3.982 | write_iops_q90 | 2.257 | disk_usage_mean | 2.215 |
| 162 -> 163 | disk_usage_mean | 2.868 | disk_usage_q50 | 2.656 | disk_usage_q90 | 2.371 |
| 163 -> 164 | total_bw_mean | 3.754 | disk_usage_q50 | 2.797 | disk_usage_mean | 2.586 |
| 164 -> 165 | read_bw_q50 | 2.072 | total_bw_q90 | 2 | read_bw_q90 | 1.897 |
| 165 -> 166 | idle_ratio | 1.634 | total_iops_q90 | 1.513 | read_iops_q99 | 1.513 |
| 166 -> 167 | ts_duration | 2.028 | write_bw_mean | 1.497 | write_share_iops_mean | 1.48 |
| 167 -> 168 | ts_duration | 2.006 | write_bw_mean | 1.299 | total_iops_lag1_autocorr | 1.141 |
| 168 -> 169 | write_bw_mean | 4.13 | total_bw_mean | 2.053 | write_share_iops_mean | 1.469 |
| 169 -> 170 | write_bw_mean | 4.862 | write_bw_q50 | 2.119 | total_bw_mean | 2.065 |
| 170 -> 171 | ts_duration | 2.949 | disk_usage_mean | 1.78 | write_bw_q90 | 1.671 |
| 171 -> 172 | ts_duration | 3.338 | write_bw_mean | 1.387 | write_bw_q90 | 1.101 |
| 172 -> 173 | write_share_iops_mean | 2.42 | total_iops_lag1_autocorr | 1.368 | read_bw_q50 | 1.342 |
| 173 -> 174 | write_bw_q50 | 1.911 | write_share_iops_mean | 1.821 | total_iops_lag1_autocorr | 1.411 |
| 174 -> 175 | write_bw_q50 | 1.045 | read_iops_q90 | 0.828 | total_iops_q90 | 0.729 |
| 175 -> 176 | write_bw_mean | 1.347 | total_bw_mean | 1.257 | write_bw_q50 | 0.972 |
| 176 -> 177 | write_bw_q90 | 2.333 | total_bw_q90 | 2.01 | read_bw_mean | 1.839 |
| 177 -> 178 | write_bw_mean | 1.031 | total_bw_q90 | 0.992 | write_bw_q90 | 0.948 |
| 178 -> 179 | ts_duration | 1.414 | disk_usage_mean | 1.054 | disk_usage_q50 | 0.94 |
| 179 -> 180 | ts_duration | 1.414 | read_bw_q90 | 0.959 | total_iops_lag1_autocorr | 0.805 |
| 180 -> 181 | ts_duration | 2.062 | write_bw_q90 | 1.362 | read_bw_q90 | 1.341 |
| 181 -> 182 | idle_ratio | 2.619 | ts_duration | 2.258 | write_iops_q90 | 1.818 |
| 182 -> 183 | total_iops_lag1_autocorr | 7.106 | total_iops_q90 | 3.928 | write_iops_q90 | 2.546 |
| 183 -> 184 | total_iops_lag1_autocorr | 9.37 | read_bw_q90 | 1.52 | idle_ratio | 1.477 |
| 184 -> 185 | total_bw_q90 | 3.375 | write_iops_q99 | 2.746 | total_bw_mean | 2.655 |
| 185 -> 186 | total_bw_mean | 2.859 | total_iops_q99 | 2.562 | write_iops_q99 | 2.385 |
| 186 -> 187 | write_share_iops_mean | 2.205 | disk_usage_q90 | 1.973 | read_iops_mean | 1.549 |
| 187 -> 188 | write_share_iops_mean | 2.594 | write_bw_q50 | 1.731 | ts_duration | 1.414 |
| 188 -> 189 | write_bw_q50 | 0.738 | total_bw_q50 | 0.703 | write_bw_mean | 0.51 |
| 189 -> 190 | read_bw_q50 | 2.794 | idle_ratio | 2.741 | total_bw_q50 | 2.38 |
| 190 -> 191 | read_bw_q50 | 2.826 | total_bw_q50 | 2.649 | read_bw_mean | 2.089 |
| 191 -> 192 | total_bw_q90 | 1.736 | write_iops_mean | 1.73 | total_iops_mean | 1.633 |
| 192 -> 193 | write_iops_mean | 1.598 | write_iops_q50 | 1.414 | read_iops_q50 | 1.414 |
| 193 -> 194 | total_bw_q90 | 3.325 | write_bw_q90 | 3.25 | write_bw_q50 | 2.931 |
| 194 -> 195 | idle_ratio | 1.538 | read_iops_q99 | 1.483 | ts_duration | 1.414 |
| 195 -> 196 | size_bytes | 2.307 | read_bw_mean | 2.099 | read_bw_q50 | 2.062 |
| 196 -> 197 | total_bw_mean | 2.299 | read_bw_mean | 2.158 | read_iops_q99 | 2.061 |
| 197 -> 198 | disk_usage_q90 | 2.761 | disk_usage_mean | 2.019 | disk_usage_q50 | 1.808 |
| 198 -> 199 | idle_ratio | 1.739 | read_iops_mean | 1.595 | read_iops_q99 | 1.487 |
| 199 -> 200 | total_iops_lag1_autocorr | 2.378 | write_bw_mean | 1.882 | total_bw_mean | 1.517 |
| 200 -> 201 | write_bw_mean | 1.722 | idle_ratio | 1.462 | ts_duration | 1.414 |
| 201 -> 202 | ts_duration | 2.872 | total_iops_mean | 1.815 | write_iops_q99 | 1.516 |
| 202 -> 203 | read_iops_mean | 1.723 | idle_ratio | 1.664 | total_iops_q99 | 1.596 |
| 203 -> 204 | disk_usage_q90 | 3.316 | disk_usage_mean | 3.254 | disk_usage_q50 | 3.131 |
| 204 -> 205 | disk_usage_q50 | 4.727 | disk_usage_mean | 3.864 | disk_usage_q90 | 3.773 |
| 205 -> 206 | write_share_iops_mean | 2.023 | write_bw_mean | 1.813 | total_iops_lag1_autocorr | 1.719 |
| 206 -> 207 | write_share_iops_mean | 2.403 | read_bw_mean | 2.316 | read_bw_q90 | 1.996 |
| 207 -> 208 | read_bw_q90 | 0.827 | write_bw_q90 | 0.703 | read_bw_mean | 0.64 |
| 208 -> 209 | disk_usage_mean | 0.883 | disk_usage_q50 | 0.867 | total_iops_lag1_autocorr | 0.836 |
| 209 -> 210 | disk_usage_q90 | 0.72 | disk_usage_mean | 0.714 | disk_usage_q50 | 0.687 |
| 210 -> 211 | write_iops_q99 | 0.434 | write_bw_q90 | 0.347 | total_iops_q99 | 0.336 |
| 211 -> 212 | read_iops_q99 | 1.861 | read_iops_mean | 1.726 | total_iops_mean | 1.549 |
| 212 -> 213 | read_iops_q99 | 1.834 | read_iops_mean | 1.738 | idle_ratio | 1.612 |
| 213 -> 214 | total_iops_lag1_autocorr | 1.452 | write_bw_q90 | 1.43 | read_iops_q50 | 1.414 |
| 214 -> 215 | write_bw_q50 | 1.556 | total_iops_lag1_autocorr | 1.45 | read_iops_q50 | 1.414 |
| 215 -> 216 | write_iops_q90 | 1.654 | read_iops_q90 | 1.616 | total_iops_q90 | 1.485 |
| 216 -> 217 | read_iops_q90 | 1.414 | total_iops_q90 | 1.342 | write_iops_q90 | 1.266 |
| 217 -> 218 | ts_duration | 2.834 | read_iops_q50 | 1.414 | total_iops_q50 | 1.414 |
| 218 -> 219 | ts_duration | 3.207 | read_iops_q50 | 1.414 | total_iops_q50 | 1.414 |
| 219 -> 220 | total_iops_lag1_autocorr | 2.896 | idle_ratio | 2.328 | write_iops_q99 | 2.208 |
| 220 -> 221 | write_iops_q99 | 2.33 | total_iops_q99 | 2.164 | read_iops_q90 | 2.04 |
| 221 -> 222 | write_bw_q50 | 1.618 | total_bw_q50 | 1.585 | read_bw_q50 | 1.567 |
| 222 -> 223 | read_bw_q50 | 1.677 | write_bw_q50 | 1.614 | total_bw_q50 | 1.573 |
| 223 -> 224 | total_iops_lag1_autocorr | 0.635 | read_bw_q90 | 0.633 | disk_usage_q90 | 0.6 |
| 224 -> 225 | write_bw_q50 | 1.806 | read_bw_q90 | 1.628 | total_bw_q90 | 1.409 |
| 225 -> 226 | write_bw_mean | 1.35 | disk_usage_mean | 1.019 | disk_usage_q90 | 1.006 |
| 226 -> 227 | total_iops_lag1_autocorr | 1.533 | total_iops_q90 | 1.416 | write_iops_q90 | 1.368 |
| 227 -> 228 | disk_usage_q90 | 4.019 | disk_usage_q50 | 3.026 | idle_ratio | 2.931 |
| 228 -> 229 | disk_usage_q50 | 4.4 | disk_usage_mean | 4.212 | idle_ratio | 4.116 |
| 229 -> 230 | read_iops_q90 | 1.762 | read_bw_q50 | 1.302 | write_iops_q90 | 1.243 |
| 230 -> 231 | read_iops_q90 | 1.342 | total_iops_q99 | 0.994 | total_iops_q90 | 0.931 |
| 231 -> 232 | disk_usage_mean | 1.827 | disk_usage_q50 | 1.748 | disk_usage_q90 | 1.619 |
| 232 -> 233 | disk_usage_mean | 3.506 | disk_usage_q50 | 3.506 | disk_usage_q90 | 3.506 |
| 233 -> 234 | total_iops_lag1_autocorr | 15.477 | total_iops_q50 | 9.9 | read_iops_q50 | 7.071 |
| 234 -> 235 | total_iops_q50 | 9.9 | read_iops_q50 | 7.071 | idle_ratio | 6.007 |
| 235 -> 236 | disk_usage_q90 | 3.453 | disk_usage_q50 | 3.313 | disk_usage_mean | 3.241 |
| 236 -> 237 | disk_usage_q50 | 5.814 | disk_usage_mean | 5.197 | disk_usage_q90 | 4.459 |
| 237 -> 238 | read_iops_q50 | 8.202 | idle_ratio | 4.462 | disk_usage_q50 | 3.487 |
| 238 -> 239 | read_iops_q50 | 8.202 | idle_ratio | 4.57 | disk_usage_mean | 3.66 |
| 239 -> 240 | read_iops_q90 | 3.062 | total_iops_q90 | 1.937 | write_iops_q90 | 1.825 |
| 240 -> 241 | read_bw_q90 | 2.364 | size_bytes | 2.276 | read_iops_q90 | 1.961 |
| 241 -> 242 | total_iops_q90 | 2.157 | write_iops_q90 | 1.961 | write_bw_mean | 1.946 |
| 242 -> 243 | write_iops_q90 | 2.5 | total_iops_q90 | 2.157 | write_bw_mean | 2.146 |
| 243 -> 244 | idle_ratio | 2.643 | write_iops_q90 | 2.391 | total_iops_q90 | 1.813 |
| 244 -> 245 | idle_ratio | 2.753 | write_iops_q90 | 2.391 | total_iops_q90 | 1.813 |
| 245 -> 246 | disk_usage_mean | 2.98 | read_iops_q99 | 2.828 | total_bw_q50 | 2.828 |
| 246 -> 247 | write_bw_mean | 1.604 | read_bw_mean | 1.558 | total_bw_mean | 1.525 |
| 247 -> 248 | read_iops_q90 | 24.033 | read_iops_q99 | 9.217 | total_iops_q90 | 7.951 |
| 248 -> 249 | size_bytes | 1.955 | write_bw_q90 | 1.635 | total_bw_q90 | 1.625 |
| 249 -> 250 | total_bw_q90 | 1.38 | write_bw_q90 | 1.373 | read_bw_mean | 1.321 |
| 250 -> 251 | write_share_iops_mean | 3.088 | disk_usage_q50 | 1.949 | disk_usage_q90 | 1.741 |
| 251 -> 252 | write_share_iops_mean | 2.288 | disk_usage_mean | 2.185 | disk_usage_q90 | 1.968 |
| 252 -> 253 | read_iops_q90 | 1.414 | write_iops_q90 | 1.414 | total_iops_q90 | 1.414 |
| 253 -> 254 | disk_usage_q90 | 1.129 | write_bw_q90 | 0.709 | read_iops_q99 | 0.626 |
| 254 -> 255 | total_iops_lag1_autocorr | 1.424 | read_iops_q99 | 1.289 | write_share_iops_mean | 1.288 |
| 255 -> 256 | total_iops_lag1_autocorr | 11.905 | disk_usage_mean | 2.979 | disk_usage_q90 | 2.821 |
| 256 -> 257 | read_iops_q99 | 23.633 | total_iops_q99 | 5.697 | write_iops_q99 | 4.749 |
| 257 -> 258 | total_iops_lag1_autocorr | 0.989 | write_bw_mean | 0.961 | disk_usage_q90 | 0.879 |
| 258 -> 259 | write_bw_q50 | 409.606 | total_bw_q50 | 191.622 | write_bw_q90 | 132.382 |
| 259 -> 260 | write_bw_q50 | 181.926 | write_bw_q90 | 88.465 | total_bw_q50 | 84.8 |
| 260 -> 261 | total_iops_q90 | 3.162 | write_iops_q90 | 3.131 | read_iops_q90 | 3.111 |
| 261 -> 262 | write_iops_q90 | 3.771 | total_iops_q90 | 3.471 | read_iops_q90 | 3.111 |
| 262 -> 263 | total_bw_mean | 1.785 | total_iops_lag1_autocorr | 1.724 | write_bw_mean | 1.639 |
| 263 -> 264 | total_bw_mean | 2.156 | write_bw_mean | 1.909 | total_iops_lag1_autocorr | 1.517 |
| 264 -> 265 | disk_usage_mean | 2.284 | size_bytes | 1.982 | disk_usage_q50 | 1.73 |
| 265 -> 266 | disk_usage_mean | 2.34 | disk_usage_q50 | 2.286 | size_bytes | 1.794 |
| 266 -> 267 | size_bytes | 2.686 | read_bw_q90 | 2.106 | idle_ratio | 1.909 |
| 267 -> 268 | size_bytes | 3.435 | idle_ratio | 3.355 | read_iops_q90 | 3.326 |
| 268 -> 269 | read_bw_q50 | 0.893 | write_bw_q50 | 0.85 | read_bw_q90 | 0.808 |
| 269 -> 270 | read_iops_q90 | 7.603 | total_iops_q90 | 3.932 | write_iops_q90 | 2.041 |
| 270 -> 271 | disk_usage_q90 | 1.39 | read_iops_q99 | 1.33 | total_iops_lag1_autocorr | 1.187 |
| 271 -> 272 | read_iops_q90 | 3.721 | total_iops_q90 | 1.768 | write_iops_q90 | 1.167 |
| 272 -> 273 | write_share_iops_mean | 1.694 | ts_duration | 1.414 | total_iops_q99 | 1.386 |
| 273 -> 274 | ts_duration | 1.414 | total_iops_q99 | 1.358 | write_iops_mean | 1.251 |
| 274 -> 275 | disk_usage_q90 | 0.981 | disk_usage_mean | 0.74 | total_bw_mean | 0.716 |
| 275 -> 276 | disk_usage_q90 | 1.177 | disk_usage_mean | 1.104 | total_bw_mean | 0.874 |
| 276 -> 277 | read_bw_q50 | 3.614 | total_bw_q50 | 2.861 | read_bw_mean | 2.764 |
| 277 -> 278 | read_bw_q50 | 3.81 | total_bw_q50 | 3.583 | total_bw_q90 | 3.115 |
| 278 -> 279 | write_bw_q50 | 1.426 | write_bw_q90 | 0.85 | write_share_iops_mean | 0.774 |
| 279 -> 280 | total_iops_q90 | 1.142 | read_iops_q90 | 1.092 | write_iops_q90 | 1.074 |
| 280 -> 281 | idle_ratio | 1.995 | read_bw_q50 | 1.886 | total_iops_q99 | 1.451 |
| 281 -> 282 | write_bw_q90 | 4.579 | total_bw_q90 | 4.227 | write_share_iops_mean | 3.933 |
| 282 -> 283 | size_bytes | 3.46 | write_iops_q90 | 2.919 | total_iops_q90 | 2.647 |
| 283 -> 284 | idle_ratio | 125.803 | write_bw_q90 | 17.113 | total_bw_q90 | 14.92 |
| 284 -> 285 | total_bw_q50 | 2.575 | write_bw_q90 | 2.43 | read_bw_q50 | 2.164 |
| 285 -> 286 | read_bw_q50 | 2.487 | total_bw_q50 | 2.333 | total_iops_q99 | 1.616 |
| 286 -> 287 | write_share_iops_mean | 4.65 | disk_usage_mean | 3.42 | disk_usage_q50 | 3.327 |
| 287 -> 288 | disk_usage_q50 | 2.185 | disk_usage_mean | 1.939 | idle_ratio | 1.444 |
| 288 -> 289 | idle_ratio | 4.604 | total_iops_lag1_autocorr | 2.282 | read_iops_q99 | 1.892 |
| 289 -> 290 | idle_ratio | 9.483 | total_iops_lag1_autocorr | 2.012 | read_iops_q99 | 1.892 |
| 290 -> 291 | write_share_iops_mean | 1.782 | idle_ratio | 1.483 | read_iops_mean | 1.042 |
| 291 -> 292 | write_share_iops_mean | 1.647 | idle_ratio | 1.424 | total_iops_q99 | 1.414 |
| 292 -> 293 | total_bw_q50 | 3.814 | total_bw_mean | 3.028 | total_bw_q90 | 1.954 |
| 293 -> 294 | total_bw_q50 | 3.055 | total_bw_mean | 2.439 | total_bw_q90 | 2.009 |
| 294 -> 295 | size_bytes | 1.43 | read_iops_q90 | 0.99 | total_iops_lag1_autocorr | 0.936 |
| 295 -> 296 | read_iops_q90 | 2.157 | total_iops_mean | 1.802 | total_iops_q90 | 1.724 |
| 296 -> 297 | idle_ratio | 2.3 | write_iops_mean | 1.481 | read_iops_mean | 1.466 |
| 297 -> 298 | idle_ratio | 1.573 | total_iops_lag1_autocorr | 1.493 | write_bw_q90 | 1.459 |
| 298 -> 299 | read_bw_q90 | 1.733 | total_bw_q90 | 1.63 | disk_usage_mean | 1.59 |
| 299 -> 300 | read_bw_q90 | 9.296 | total_bw_q50 | 7.453 | read_bw_q50 | 5.509 |
| 300 -> 301 | write_bw_q90 | 1.464 | total_iops_q90 | 1.414 | read_iops_q90 | 1.414 |
| 301 -> 302 | disk_usage_q50 | 2.252 | disk_usage_mean | 1.946 | total_bw_q50 | 1.857 |
| 302 -> 303 | write_bw_mean | 1.693 | total_bw_mean | 1.609 | read_bw_mean | 1.549 |
| 303 -> 304 | write_bw_mean | 1.717 | total_bw_mean | 1.652 | read_bw_mean | 1.638 |
| 304 -> 305 | size_bytes | 6.18 | write_iops_q99 | 1.414 | total_iops_q99 | 1.414 |
| 305 -> 306 | write_bw_q50 | 4.698 | total_bw_q50 | 3.391 | read_iops_q99 | 3.149 |
| 306 -> 307 | total_bw_mean | 3.503 | read_iops_q90 | 2.889 | write_bw_q90 | 2.498 |
| 307 -> 308 | total_bw_mean | 2.394 | size_bytes | 1.633 | read_bw_mean | 1.538 |
| 308 -> 309 | disk_usage_mean | 1.855 | disk_usage_q50 | 1.789 | read_bw_mean | 1.58 |
| 309 -> 310 | disk_usage_mean | 1.789 | read_bw_mean | 1.574 | read_bw_q90 | 1.433 |
| 310 -> 311 | read_iops_q99 | 1.414 | total_iops_q99 | 1.414 | write_iops_q99 | 1.414 |
| 311 -> 312 | total_iops_lag1_autocorr | 1.134 | read_bw_q90 | 0.97 | write_bw_mean | 0.831 |
| 312 -> 313 | ts_duration | 1.414 | total_iops_q90 | 1.414 | read_iops_q90 | 1.414 |
| 313 -> 314 | disk_usage_q90 | 1.8 | disk_usage_q50 | 1.768 | write_bw_q50 | 1.7 |
| 314 -> 315 | size_bytes | 6.348 | total_bw_mean | 1.968 | total_bw_q50 | 1.578 |
| 315 -> 316 | read_iops_q99 | 1.166 | total_iops_mean | 1.127 | total_iops_q99 | 1.042 |
| 316 -> 317 | size_bytes | 6.051 | ts_duration | 1.414 | idle_ratio | 0.977 |
| 317 -> 318 | size_bytes | 6.25 | ts_duration | 1.414 | total_iops_q99 | 1.08 |
| 318 -> 319 | total_iops_lag1_autocorr | 16.316 | size_bytes | 10.274 | write_bw_q90 | 3.022 |
| 319 -> 320 | total_iops_lag1_autocorr | 5.542 | read_iops_q99 | 2.214 | read_bw_mean | 2.065 |
| 320 -> 321 | size_bytes | 5.959 | total_iops_mean | 3.518 | idle_ratio | 3.183 |
| 321 -> 322 | total_iops_mean | 4.951 | idle_ratio | 4.26 | read_iops_mean | 3.439 |
| 322 -> 323 | write_share_iops_mean | 3.17 | read_iops_mean | 2.276 | total_iops_mean | 1.813 |
| 323 -> 324 | write_share_iops_mean | 3.437 | size_bytes | 2.883 | read_iops_mean | 2.461 |
| 324 -> 325 | total_iops_lag1_autocorr | 1.117 | read_iops_mean | 0.87 | write_share_iops_mean | 0.776 |
| 325 -> 326 | size_bytes | 6.853 | write_iops_mean | 1.866 | read_iops_q99 | 1.45 |
| 326 -> 327 | write_share_iops_mean | 1.972 | read_iops_q99 | 1.414 | read_bw_q50 | 1.389 |
| 327 -> 328 | total_iops_mean | 2.236 | read_iops_mean | 1.976 | read_iops_q99 | 1.414 |
| 328 -> 329 | size_bytes | 5.949 | disk_usage_q50 | 2.828 | disk_usage_q90 | 2.828 |
| 329 -> 330 | disk_usage_mean | 6.959 | idle_ratio | 2.455 | write_iops_q99 | 2.272 |
| 330 -> 331 | size_bytes | 4.102 | write_iops_q99 | 1.414 | total_iops_q99 | 1.414 |
| 331 -> 332 | size_bytes | 3.933 | idle_ratio | 1.333 | write_iops_mean | 1.021 |
| 332 -> 333 | total_iops_q99 | 1.414 | read_bw_q50 | 1.414 | read_bw_q90 | 1.414 |
| 333 -> 334 | read_iops_q99 | 1.414 | total_iops_q99 | 1.163 | write_iops_q99 | 1.109 |
| 334 -> 335 | total_bw_q90 | 49.498 | total_bw_q50 | 41.012 | read_bw_q50 | 28.284 |
| 335 -> 336 | total_iops_lag1_autocorr | 4.327 | total_iops_mean | 1.905 | write_iops_mean | 1.8 |
| 336 -> 337 | total_bw_q50 | 17.253 | read_bw_q50 | 16.971 | write_bw_q50 | 12.728 |
| 337 -> 338 | write_iops_q99 | 1.414 | total_iops_q99 | 1.414 | total_iops_lag1_autocorr | 1.412 |
| 338 -> 339 | total_bw_q90 | 49.498 | total_bw_q50 | 42.426 | read_bw_q90 | 33.941 |
| 339 -> 340 | size_bytes | 5.167 | disk_usage_mean | 1.466 | read_iops_q99 | 1.414 |
| 340 -> 341 | write_iops_mean | 1.159 | total_iops_lag1_autocorr | 1.136 | total_iops_mean | 1.126 |
| 341 -> 342 | write_iops_q90 | 1.414 | total_iops_q90 | 1.414 | disk_usage_q90 | 1.414 |
| 342 -> 343 | read_iops_q99 | 0.788 | total_iops_q99 | 0.754 | disk_usage_q90 | 0.588 |
| 343 -> 344 | disk_usage_mean | 2.576 | disk_usage_q50 | 2.53 | write_iops_q99 | 1.414 |
| 344 -> 345 | idle_ratio | 1.589 | write_iops_q99 | 1.523 | total_iops_q99 | 1.503 |
| 345 -> 346 | write_iops_q90 | 1.414 | total_iops_q90 | 1.414 | write_iops_mean | 0.998 |
| 346 -> 347 | write_bw_mean | 1.317 | size_bytes | 1.294 | write_iops_q99 | 1.182 |
| 347 -> 348 | write_bw_q50 | 3.589 | write_bw_mean | 2.048 | write_bw_q90 | 1.937 |
| 348 -> 349 | disk_usage_q90 | 1.534 | write_share_iops_mean | 1.507 | read_iops_q90 | 1.414 |
| 349 -> 350 | write_bw_q50 | 1.321 | size_bytes | 1.307 | write_bw_q90 | 1.257 |
| 350 -> 351 | total_bw_q90 | 1.513 | read_bw_q50 | 1.402 | write_bw_q90 | 1.345 |
| 351 -> 352 | idle_ratio | 4.888 | write_iops_q50 | 2.502 | size_bytes | 2.409 |
| 352 -> 353 | read_bw_q90 | 4.145 | idle_ratio | 3.598 | total_bw_q90 | 2.698 |
| 353 -> 354 | write_share_iops_mean | 1.78 | read_iops_q99 | 1.196 | total_iops_q99 | 1.044 |
| 354 -> 355 | total_iops_q90 | 1.414 | write_iops_q90 | 1.414 | write_share_iops_mean | 1.13 |
| 355 -> 356 | write_iops_q90 | 1.414 | read_iops_q90 | 1.414 | total_iops_q90 | 1.414 |
| 356 -> 357 | idle_ratio | 2.534 | read_iops_q99 | 1.858 | total_iops_q99 | 1.644 |
| 357 -> 358 | write_iops_mean | 4.611 | total_iops_mean | 4.145 | read_iops_mean | 2.865 |
| 358 -> 359 | write_iops_mean | 4.926 | total_iops_mean | 4.478 | idle_ratio | 4.419 |
| 359 -> 360 | disk_usage_q50 | 14.619 | disk_usage_mean | 12.982 | disk_usage_q90 | 12.962 |
| 360 -> 361 | disk_usage_q50 | 2.335 | disk_usage_q90 | 2.252 | disk_usage_mean | 2.223 |
| 361 -> 362 | disk_usage_q50 | 1.985 | disk_usage_mean | 1.95 | disk_usage_q90 | 1.93 |
| 362 -> 363 | disk_usage_q50 | 2.926 | disk_usage_q90 | 2.625 | disk_usage_mean | 2.399 |
| 363 -> 364 | read_iops_q99 | 2.68 | read_iops_mean | 2.651 | total_iops_q99 | 2.54 |
| 364 -> 365 | write_iops_q90 | 2.828 | total_iops_q90 | 2.828 | idle_ratio | 2.795 |
| 365 -> 366 | read_iops_q99 | 1.414 | write_iops_q90 | 1.414 | total_iops_q90 | 1.414 |
| 366 -> 367 | write_share_iops_mean | 2.696 | read_iops_q99 | 1.414 | read_iops_mean | 1.271 |
| 367 -> 368 | write_iops_mean | 1.414 | read_iops_q50 | 1.414 | read_iops_q90 | 1.414 |
| 368 -> 369 | write_iops_mean | 1.414 | total_iops_mean | 1.414 | read_iops_q50 | 1.414 |
| 369 -> 370 | disk_usage_q50 | 15.346 | disk_usage_q90 | 9.184 | disk_usage_mean | 7.604 |
| 370 -> 371 | disk_usage_q50 | 43 | disk_usage_q90 | 35.301 | disk_usage_mean | 10.954 |
| 371 -> 372 | total_iops_lag1_autocorr | 5.403 | write_bw_q90 | 1.817 | total_iops_q99 | 1.492 |
| 372 -> 373 | total_iops_lag1_autocorr | 5.585 | total_bw_q90 | 1.901 | read_bw_q90 | 1.731 |
| 373 -> 374 | write_share_iops_mean | 0.711 | write_iops_q99 | 0.632 | size_bytes | 0.519 |
| 374 -> 375 | total_iops_lag1_autocorr | 1.212 | write_share_iops_mean | 0.72 | write_iops_q99 | 0.485 |
| 375 -> 376 | write_iops_q99 | 1.109 | total_iops_q99 | 0.894 | idle_ratio | 0.811 |
| 376 -> 377 | write_iops_q99 | 1.414 | total_iops_q99 | 1.414 | idle_ratio | 1.222 |
| 377 -> 378 | read_bw_q50 | 5.814 | read_bw_mean | 4.7 | total_bw_mean | 4.023 |
| 378 -> 379 | write_share_iops_mean | 10.754 | read_bw_q50 | 4.714 | write_bw_q50 | 4.243 |
| 379 -> 380 | read_iops_q99 | 7.041 | total_iops_q99 | 2.761 | write_iops_q99 | 2.354 |
| 380 -> 381 | read_iops_q99 | 6.589 | total_iops_q99 | 2.704 | write_iops_q99 | 2.29 |
| 381 -> 382 | disk_usage_mean | 1.421 | write_iops_q50 | 1.414 | total_iops_q50 | 1.414 |
| 382 -> 383 | total_bw_q50 | 1.713 | write_bw_q50 | 1.667 | write_iops_q50 | 1.414 |
| 383 -> 384 | write_share_iops_mean | 2.521 | write_iops_q90 | 2.121 | total_iops_q90 | 2.121 |
| 384 -> 385 | write_iops_q50 | 1.414 | total_iops_q50 | 1.414 | read_iops_q99 | 1.276 |
| 385 -> 386 | size_bytes | 2.67 | disk_usage_q90 | 2.219 | disk_usage_mean | 2.068 |
| 386 -> 387 | write_bw_mean | 1.869 | total_iops_lag1_autocorr | 1.647 | total_bw_mean | 1.615 |
| 387 -> 388 | size_bytes | 2.199 | total_iops_q99 | 1.177 | write_iops_q99 | 1.088 |
| 388 -> 389 | size_bytes | 1.853 | disk_usage_q90 | 1 | disk_usage_mean | 0.911 |
| 389 -> 390 | read_iops_mean | 4.297 | total_iops_mean | 2.312 | write_iops_q99 | 2.192 |
| 390 -> 391 | read_iops_mean | 7.697 | total_iops_mean | 2.896 | write_iops_q99 | 2.273 |
| 391 -> 392 | total_iops_mean | 1.113 | size_bytes | 1.016 | write_iops_q99 | 0.788 |
| 392 -> 393 | read_iops_q99 | 1.414 | total_iops_q99 | 1.32 | write_iops_q99 | 1.285 |
| 393 -> 394 | idle_ratio | 1.865 | write_iops_mean | 1.448 | write_iops_q90 | 1.414 |
| 394 -> 395 | write_iops_mean | 2.216 | idle_ratio | 2.035 | disk_usage_mean | 1.422 |
| 395 -> 396 | total_iops_mean | 1.678 | write_iops_q99 | 1.414 | total_iops_q99 | 1.414 |
| 396 -> 397 | read_iops_mean | 1.744 | total_iops_mean | 1.628 | total_iops_q99 | 1.414 |
| 397 -> 398 | read_iops_mean | 1.31 | total_iops_mean | 0.862 | write_iops_q99 | 0.774 |
| 398 -> 399 | write_share_iops_mean | 1.531 | disk_usage_q50 | 1.414 | disk_usage_q90 | 1.414 |
| 399 -> 400 | write_share_iops_mean | 1.438 | write_iops_mean | 1.043 | write_bw_mean | 0.859 |
| 400 -> 401 | disk_usage_mean | 1.645 | disk_usage_q90 | 1.372 | read_iops_q99 | 1.342 |
| 401 -> 402 | read_iops_mean | 1.544 | read_bw_q50 | 1.177 | idle_ratio | 1.028 |
| 402 -> 403 | total_bw_q50 | 45.255 | total_bw_mean | 31.834 | read_bw_q50 | 31.113 |
| 403 -> 404 | write_bw_q50 | 7.071 | write_bw_mean | 6.677 | write_share_iops_mean | 6.196 |
| 404 -> 405 | read_iops_q99 | 1.414 | write_iops_q50 | 1.414 | write_iops_q90 | 1.414 |
| 405 -> 406 | idle_ratio | 1.977 | read_iops_q99 | 1.94 | total_iops_q99 | 1.57 |
| 406 -> 407 | disk_usage_q90 | 2.121 | idle_ratio | 1.656 | disk_usage_mean | 1.638 |
| 407 -> 408 | write_share_iops_mean | 2.126 | write_iops_q90 | 1.414 | total_iops_q90 | 1.414 |
| 408 -> 409 | write_share_iops_mean | 1.573 | read_iops_q99 | 1.414 | write_iops_q90 | 1.414 |
| 409 -> 410 | disk_usage_q90 | 3.138 | ts_duration | 3.027 | write_bw_q90 | 2.707 |
| 410 -> 411 | ts_duration | 2.845 | disk_usage_q90 | 1.697 | read_iops_q99 | 1.414 |
| 411 -> 412 | idle_ratio | 1.163 | total_iops_lag1_autocorr | 1.04 | total_iops_mean | 0.966 |
| 412 -> 413 | disk_usage_q50 | 2.2 | disk_usage_mean | 2.028 | disk_usage_q90 | 1.97 |
| 413 -> 414 | write_bw_q90 | 1.59 | disk_usage_mean | 1.47 | read_iops_mean | 1.431 |
| 414 -> 415 | write_iops_q99 | 4.576 | total_iops_q99 | 4.099 | read_iops_q99 | 1.886 |
| 415 -> 416 | write_bw_q90 | 1.15 | disk_usage_q90 | 1.029 | total_bw_q90 | 0.826 |
| 416 -> 417 | read_iops_q99 | 1.414 | write_iops_q90 | 1.414 | total_iops_q90 | 1.414 |
| 417 -> 418 | total_iops_lag1_autocorr | 1.188 | disk_usage_q90 | 0.798 | write_share_iops_mean | 0.465 |
| 418 -> 419 | write_bw_q90 | 2.994 | total_bw_q90 | 2.215 | read_bw_q90 | 2.143 |
| 419 -> 420 | write_bw_q90 | 4.57 | total_bw_q90 | 3.273 | read_bw_q90 | 2.691 |
| 420 -> 421 | write_iops_q99 | 1.414 | total_iops_q99 | 1.414 | idle_ratio | 1.334 |
| 421 -> 422 | write_iops_q99 | 1.414 | total_iops_q99 | 1.414 | total_iops_mean | 1.304 |
| 422 -> 423 | disk_usage_q50 | 1.265 | total_iops_lag1_autocorr | 1.218 | write_bw_mean | 1.066 |
| 423 -> 424 | total_iops_lag1_autocorr | 1.553 | disk_usage_q90 | 1.199 | idle_ratio | 1.06 |
| 424 -> 425 | ts_duration | 3.336 | total_iops_mean | 1.768 | disk_usage_q90 | 1.698 |
| 425 -> 426 | ts_duration | 2.81 | disk_usage_q50 | 1.98 | disk_usage_q90 | 1.98 |
| 426 -> 427 | write_bw_q90 | 6.829 | total_bw_q90 | 6.587 | total_bw_mean | 4.733 |
| 427 -> 428 | write_bw_q90 | 6.82 | total_bw_q90 | 6.56 | total_bw_mean | 4.985 |
| 428 -> 429 | ts_duration | 2.958 | read_iops_q90 | 1.414 | write_iops_q90 | 1.414 |
| 429 -> 430 | ts_duration | 3.055 | read_iops_q90 | 1.414 | write_iops_q90 | 1.414 |
| 430 -> 431 | write_iops_q99 | 0.992 | idle_ratio | 0.934 | write_iops_mean | 0.77 |
| 431 -> 432 | read_iops_q99 | 1.414 | total_iops_lag1_autocorr | 1.225 | write_iops_mean | 1.224 |
| 432 -> 433 | write_iops_mean | 5.296 | total_iops_mean | 4.229 | read_iops_mean | 2.309 |
| 433 -> 434 | write_iops_mean | 4.833 | total_iops_mean | 3.793 | read_iops_mean | 2.16 |
| 434 -> 435 | write_share_iops_mean | 1.742 | write_iops_q90 | 1.414 | read_iops_q90 | 1.414 |
| 435 -> 436 | disk_usage_q90 | 1.949 | disk_usage_mean | 1.662 | disk_usage_q50 | 1.649 |
| 436 -> 437 | read_iops_q90 | 1.414 | read_iops_q99 | 1.317 | total_iops_q99 | 1.303 |
| 437 -> 438 | read_iops_q90 | 1.414 | total_iops_q90 | 1.28 | total_iops_q99 | 1.262 |
| 438 -> 439 | read_iops_q90 | 107.48 | write_bw_q50 | 93.928 | write_bw_q90 | 91.515 |
| 439 -> 440 | read_iops_q90 | 107.48 | write_bw_q90 | 79.415 | write_bw_q50 | 73.956 |
| 440 -> 441 | disk_usage_q90 | 1.715 | disk_usage_mean | 1.595 | total_iops_lag1_autocorr | 1.44 |
| 441 -> 442 | disk_usage_mean | 1.589 | total_iops_lag1_autocorr | 1.439 | ts_duration | 1.414 |
| 442 -> 443 | disk_usage_q90 | 1.213 | write_iops_mean | 1.148 | total_iops_mean | 1.093 |
| 443 -> 444 | read_iops_q99 | 1.296 | total_iops_q99 | 1.102 | write_iops_q99 | 0.981 |
| 444 -> 445 | write_share_iops_mean | 1.608 | write_iops_q90 | 1.414 | total_iops_q90 | 1.414 |
| 445 -> 446 | write_iops_q90 | 1.414 | total_iops_q90 | 1.414 | total_iops_mean | 1.109 |
| 446 -> 447 | write_iops_mean | 7.304 | idle_ratio | 4.286 | total_iops_mean | 2.883 |
| 447 -> 448 | total_iops_lag1_autocorr | 2.674 | total_iops_q50 | 2.357 | idle_ratio | 2.305 |
| 448 -> 449 | write_iops_q90 | 1.414 | total_iops_q90 | 1.414 | disk_usage_q90 | 1.265 |
| 449 -> 450 | total_bw_q90 | 1.581 | write_bw_q50 | 1.397 | write_iops_q99 | 1.328 |
| 450 -> 451 | total_bw_q50 | 2.639 | write_bw_mean | 2.47 | total_bw_mean | 2.246 |
| 451 -> 452 | disk_usage_q90 | 3.333 | total_iops_lag1_autocorr | 3.012 | total_bw_q50 | 2.933 |
| 452 -> 453 | read_iops_q99 | 1.414 | write_iops_q90 | 1.414 | total_iops_q90 | 1.414 |
| 453 -> 454 | write_bw_mean | 1.827 | write_bw_q90 | 1.652 | write_bw_q50 | 1.644 |
| 454 -> 455 | total_iops_lag1_autocorr | 3.112 | write_iops_mean | 1.996 | total_iops_mean | 1.872 |
| 455 -> 456 | total_iops_lag1_autocorr | 2.04 | write_bw_mean | 1.514 | idle_ratio | 1.481 |
| 456 -> 457 | disk_usage_q50 | 3.25 | disk_usage_q90 | 3.004 | disk_usage_mean | 2.777 |
| 457 -> 458 | write_share_iops_mean | 6.448 | idle_ratio | 4.021 | total_iops_lag1_autocorr | 2.109 |
| 458 -> 459 | read_bw_q50 | 9.142 | read_bw_mean | 8.693 | write_bw_q50 | 8.573 |
| 459 -> 460 | read_iops_q99 | 1.414 | total_iops_lag1_autocorr | 0.833 | write_iops_q90 | 0.632 |
| 460 -> 461 | read_iops_q99 | 1.414 | total_iops_lag1_autocorr | 1.133 | total_iops_q99 | 0.993 |
| 461 -> 462 | total_bw_q90 | 1.997 | size_bytes | 1.956 | total_bw_mean | 1.89 |
| 462 -> 463 | size_bytes | 2.89 | read_bw_q90 | 2.06 | total_bw_q90 | 2.017 |
| 463 -> 464 | disk_usage_q50 | 1.762 | disk_usage_mean | 1.726 | disk_usage_q90 | 1.695 |
| 464 -> 465 | write_bw_q50 | 7.593 | total_bw_q50 | 4.178 | write_bw_mean | 3.95 |
| 465 -> 466 | read_bw_q50 | 5.187 | read_bw_mean | 4.494 | total_bw_q90 | 4.362 |
| 466 -> 467 | read_bw_q50 | 4.941 | read_bw_mean | 4.579 | disk_usage_q90 | 4.515 |
| 467 -> 468 | write_iops_q90 | 1.414 | total_iops_q90 | 1.414 | write_bw_q50 | 1.408 |
| 468 -> 469 | write_bw_q50 | 1.569 | total_bw_q50 | 1.542 | disk_usage_q50 | 1.423 |
| 469 -> 470 | write_share_iops_mean | 1.057 | total_iops_lag1_autocorr | 0.92 | write_bw_mean | 0.681 |
| 470 -> 471 | write_iops_mean | 1.255 | total_iops_mean | 1.246 | read_iops_mean | 1.177 |
| 471 -> 472 | write_iops_q99 | 1.414 | total_iops_q99 | 1.414 | idle_ratio | 1.146 |
| 472 -> 473 | size_bytes | 0.842 | write_iops_mean | 0.753 | write_share_iops_mean | 0.545 |
| 473 -> 474 | size_bytes | 2.38 | disk_usage_mean | 2.033 | disk_usage_q50 | 1.94 |
| 474 -> 475 | write_bw_q50 | 1.51 | read_bw_q90 | 1.494 | write_bw_mean | 1.49 |
| 475 -> 476 | total_iops_lag1_autocorr | 2.827 | write_bw_q50 | 1.692 | write_bw_mean | 1.593 |
| 476 -> 477 | write_share_iops_mean | 3.421 | disk_usage_q90 | 1.99 | write_bw_q50 | 1.715 |
| 477 -> 478 | write_iops_q50 | 2.121 | total_iops_q50 | 2.121 | total_bw_q90 | 1.781 |
| 478 -> 479 | total_iops_lag1_autocorr | 1.764 | idle_ratio | 1.374 | read_bw_q90 | 0.945 |
| 479 -> 480 | write_bw_q90 | 0.754 | write_iops_q99 | 0.74 | read_iops_mean | 0.706 |
| 480 -> 481 | idle_ratio | 1.284 | write_share_iops_mean | 1.164 | total_iops_lag1_autocorr | 1.149 |
| 481 -> 482 | total_iops_lag1_autocorr | 2.946 | total_bw_mean | 1.879 | write_bw_q90 | 1.744 |
| 482 -> 483 | total_iops_lag1_autocorr | 1.276 | total_bw_q50 | 1.088 | total_iops_q99 | 1.078 |
| 483 -> 484 | write_iops_q99 | 1.418 | read_iops_q90 | 1.118 | total_iops_q99 | 1.093 |
| 484 -> 485 | write_iops_q99 | 1.514 | total_iops_q99 | 1.328 | idle_ratio | 1.174 |
| 485 -> 486 | write_share_iops_mean | 2.135 | write_iops_q50 | 2.058 | read_bw_q50 | 1.999 |
| 486 -> 487 | write_share_iops_mean | 2.096 | size_bytes | 1.583 | total_iops_q90 | 1.472 |
| 487 -> 488 | write_bw_q50 | 136.822 | write_bw_q90 | 98.168 | read_iops_q90 | 87.406 |
| 488 -> 489 | read_iops_q90 | 115.947 | total_iops_q50 | 74.388 | write_bw_q50 | 68.353 |
| 489 -> 490 | disk_usage_q90 | 1.265 | disk_usage_q50 | 1.131 | disk_usage_mean | 0.993 |
| 490 -> 491 | total_iops_mean | 2.135 | read_iops_mean | 1.814 | write_iops_mean | 1.796 |
| 491 -> 492 | idle_ratio | 16.393 | read_iops_mean | 1.808 | total_iops_lag1_autocorr | 1.709 |
| 492 -> 493 | idle_ratio | 9.882 | read_iops_q90 | 1.414 | write_iops_q50 | 1.414 |
| 493 -> 494 | write_iops_q90 | 1.414 | total_iops_q90 | 1.414 | total_iops_q99 | 1.231 |
| 494 -> 495 | total_iops_lag1_autocorr | 1.156 | write_bw_mean | 1.142 | disk_usage_mean | 1.038 |
| 495 -> 496 | idle_ratio | 15.308 | total_iops_q90 | 1.749 | read_iops_q90 | 1.616 |
| 496 -> 497 | idle_ratio | 2.066 | write_iops_q50 | 1.571 | total_iops_q50 | 1.571 |
| 497 -> 498 | write_iops_q50 | 1.414 | read_iops_q50 | 1.414 | total_iops_q50 | 1.414 |
| 498 -> 499 | read_iops_q90 | 1.414 | write_iops_q50 | 1.414 | write_iops_q90 | 1.414 |
| 499 -> 500 | read_iops_q90 | 2.867 | write_bw_q90 | 2.72 | total_iops_q90 | 2.445 |
| 500 -> 501 | read_iops_q90 | 2.743 | total_iops_q90 | 2.177 | write_iops_q90 | 1.815 |
| 501 -> 502 | total_iops_lag1_autocorr | 4.548 | write_bw_q90 | 0.772 | write_iops_q90 | 0.582 |
| 502 -> 503 | write_iops_q50 | 26.87 | total_iops_lag1_autocorr | 6.248 | idle_ratio | 2.551 |
| 503 -> 504 | write_iops_q50 | 26.87 | idle_ratio | 6.221 | write_share_iops_mean | 4.383 |
| 504 -> 505 | write_share_iops_mean | 4.326 | read_iops_mean | 3.171 | total_bw_q50 | 3.117 |
| 505 -> 506 | idle_ratio | 3.168 | write_iops_q50 | 2.121 | total_iops_q50 | 2.121 |
| 506 -> 507 | write_share_iops_mean | 1.712 | read_bw_mean | 1.343 | total_bw_mean | 1.267 |
| 507 -> 508 | read_bw_q50 | 1.416 | read_bw_mean | 1.416 | read_iops_q90 | 1.415 |
| 508 -> 509 | idle_ratio | 1.429 | read_iops_q90 | 1.416 | read_bw_mean | 1.415 |
| 509 -> 510 | read_iops_q90 | 1.83 | total_iops_q90 | 1.64 | write_iops_q90 | 1.572 |
| 510 -> 511 | idle_ratio | 6.852 | read_iops_q90 | 5.367 | total_iops_q90 | 5.08 |
| 511 -> 512 | write_bw_q90 | 2.454 | disk_usage_q50 | 1.628 | write_share_iops_mean | 1.574 |
| 512 -> 513 | write_bw_q90 | 2.042 | total_bw_q90 | 1.73 | write_share_iops_mean | 1.613 |
| 513 -> 514 | write_share_iops_mean | 3.279 | read_iops_q90 | 0.971 | write_bw_q50 | 0.828 |
| 514 -> 515 | idle_ratio | 3.907 | write_share_iops_mean | 1.759 | write_iops_q50 | 1.414 |
| 515 -> 516 | total_iops_lag1_autocorr | 1.518 | total_iops_q99 | 1.453 | read_iops_q99 | 1.439 |
| 516 -> 517 | total_iops_q99 | 1.724 | write_iops_q99 | 1.578 | write_iops_mean | 1.495 |
| 517 -> 518 | write_share_iops_mean | 1.756 | read_iops_q90 | 1.414 | write_iops_q90 | 1.414 |
| 518 -> 519 | write_bw_q90 | 1.441 | write_bw_mean | 1.379 | write_share_iops_mean | 1.252 |
| 519 -> 520 | read_iops_q99 | 0.876 | total_iops_q99 | 0.778 | write_iops_q99 | 0.661 |
| 520 -> 521 | write_iops_q90 | 1.414 | total_iops_q90 | 1.414 | disk_usage_q90 | 1.333 |
| 521 -> 522 | write_iops_q90 | 1.414 | total_iops_q90 | 1.414 | write_bw_mean | 1.414 |
| 522 -> 523 | write_iops_q90 | 1.414 | total_iops_q90 | 1.414 | write_bw_mean | 1.395 |
| 523 -> 524 | total_iops_lag1_autocorr | 3.181 | size_bytes | 1.331 | write_share_iops_mean | 1.282 |
| 524 -> 525 | total_iops_lag1_autocorr | 5.081 | read_iops_mean | 1.173 | read_iops_q99 | 1.029 |
| 525 -> 526 | size_bytes | 1.599 | total_iops_lag1_autocorr | 1.467 | write_bw_q50 | 1.327 |
| 526 -> 527 | total_iops_q99 | 2.873 | write_iops_q99 | 2.088 | total_iops_lag1_autocorr | 1.882 |
| 527 -> 528 | read_bw_mean | 1.598 | read_bw_q50 | 1.436 | write_iops_q90 | 1.414 |
| 528 -> 529 | write_bw_q90 | 1.613 | total_bw_mean | 1.574 | write_bw_mean | 1.256 |
| 529 -> 530 | read_iops_q99 | 7.071 | total_iops_lag1_autocorr | 5.511 | idle_ratio | 3.594 |
| 530 -> 531 | read_iops_q99 | 7.071 | read_bw_mean | 3.795 | total_iops_lag1_autocorr | 3.103 |
| 531 -> 532 | disk_usage_q90 | 2.159 | disk_usage_mean | 2.094 | disk_usage_q50 | 1.715 |
| 532 -> 533 | disk_usage_q90 | 2.557 | disk_usage_mean | 2.469 | disk_usage_q50 | 2.319 |
| 533 -> 534 | total_iops_lag1_autocorr | 4.422 | read_iops_q90 | 1.414 | write_iops_q50 | 1.414 |
| 534 -> 535 | total_iops_lag1_autocorr | 2.273 | read_iops_q90 | 1.414 | write_iops_q50 | 1.414 |
| 535 -> 536 | total_bw_q50 | 3.113 | read_bw_q50 | 2.621 | read_bw_q90 | 2.585 |
| 536 -> 537 | total_bw_q50 | 2.89 | read_bw_q50 | 2.513 | size_bytes | 2.242 |
| 537 -> 538 | ts_duration | 2.035 | write_iops_q90 | 1.414 | total_iops_q90 | 1.414 |
| 538 -> 539 | disk_usage_q50 | 1.997 | disk_usage_mean | 1.975 | disk_usage_q90 | 1.56 |
| 539 -> 540 | total_iops_lag1_autocorr | 1.676 | write_share_iops_mean | 1.585 | total_bw_q50 | 1.498 |
| 540 -> 541 | ts_duration | 1.414 | read_iops_q90 | 1.414 | write_iops_q50 | 1.414 |
| 541 -> 542 | write_bw_q90 | 1.415 | ts_duration | 1.414 | read_iops_q90 | 1.414 |
| 542 -> 543 | ts_duration | 2.885 | idle_ratio | 2.669 | write_iops_q50 | 2.121 |
| 543 -> 544 | read_bw_mean | 2.421 | total_bw_q50 | 1.789 | read_bw_q50 | 1.664 |
| 544 -> 545 | write_iops_q50 | 4 | total_iops_q50 | 4 | write_iops_q99 | 1.476 |
| 545 -> 546 | write_iops_q99 | 1.731 | total_iops_q99 | 1.691 | read_iops_q90 | 1.414 |
| 546 -> 547 | read_iops_q90 | 1.414 | write_share_iops_mean | 1.222 | read_iops_q99 | 1.205 |
| 547 -> 548 | ts_duration | 1.414 | read_iops_q90 | 1.414 | write_bw_mean | 1.369 |
| 548 -> 549 | ts_duration | 1.414 | write_iops_q90 | 1.414 | total_iops_q90 | 1.414 |
| 549 -> 550 | ts_duration | 2.243 | total_iops_lag1_autocorr | 1.199 | write_bw_q50 | 1.085 |
| 550 -> 551 | total_iops_lag1_autocorr | 1.779 | size_bytes | 1.496 | read_bw_q50 | 1.481 |
| 551 -> 552 | ts_duration | 2.928 | total_iops_lag1_autocorr | 1.859 | read_bw_q50 | 1.472 |
| 552 -> 553 | idle_ratio | 49.663 | disk_usage_mean | 5.152 | disk_usage_q50 | 4.438 |
| 553 -> 554 | idle_ratio | 43.877 | write_iops_mean | 6.062 | write_iops_q90 | 4.243 |
| 554 -> 555 | write_bw_q50 | 2.333 | read_bw_q50 | 1.741 | total_bw_q50 | 1.551 |
| 555 -> 556 | read_bw_q90 | 2.426 | size_bytes | 2.417 | total_bw_q90 | 2.303 |
| 556 -> 557 | read_bw_q90 | 1.76 | total_bw_q90 | 1.736 | write_bw_q90 | 1.713 |
| 557 -> 558 | write_iops_q90 | 1.414 | total_iops_q90 | 1.414 | write_iops_q50 | 1.414 |
| 558 -> 559 | read_bw_q90 | 1.999 | read_bw_mean | 1.891 | total_bw_q90 | 1.836 |
| 559 -> 560 | write_iops_mean | 2.18 | write_iops_q90 | 2.121 | total_iops_q90 | 1.697 |
| 560 -> 561 | write_iops_q90 | 2.121 | write_iops_mean | 2.023 | total_iops_q90 | 1.697 |
| 561 -> 562 | idle_ratio | 1.89 | write_iops_q90 | 1.414 | total_iops_q90 | 1.414 |
| 562 -> 563 | idle_ratio | 2.149 | write_iops_q90 | 1.414 | total_iops_q90 | 1.414 |
| 563 -> 564 | write_iops_q99 | 1.414 | read_iops_q99 | 1.414 | total_iops_q99 | 1.414 |
| 564 -> 565 | read_bw_q50 | 2.183 | write_iops_mean | 1.687 | read_iops_mean | 1.644 |
| 565 -> 566 | disk_usage_mean | 2.349 | disk_usage_q90 | 1.98 | disk_usage_q50 | 1.768 |
| 566 -> 567 | disk_usage_q50 | 4.243 | disk_usage_q90 | 3.839 | disk_usage_mean | 3.818 |
| 567 -> 568 | size_bytes | 2.768 | write_iops_q99 | 1.414 | total_iops_q99 | 1.414 |
| 568 -> 569 | size_bytes | 8.363 | write_bw_q50 | 2.644 | disk_usage_q90 | 2.058 |
| 569 -> 570 | total_bw_mean | 2.141 | write_bw_q90 | 1.917 | write_iops_mean | 1.511 |
| 570 -> 571 | write_bw_q90 | 1.804 | total_iops_mean | 1.43 | read_iops_mean | 1.42 |
| 571 -> 572 | read_iops_q99 | 1.266 | total_iops_q99 | 1.128 | write_iops_q99 | 1.074 |
| 572 -> 573 | total_bw_q50 | 1.02 | write_bw_q50 | 1.016 | read_iops_q99 | 0.948 |
| 573 -> 574 | idle_ratio | 5.84 | write_iops_q50 | 4.243 | total_iops_q50 | 3.182 |
| 574 -> 575 | idle_ratio | 6.492 | write_iops_q50 | 4.243 | total_iops_q50 | 3.182 |
| 575 -> 576 | ts_duration | 3.245 | total_iops_lag1_autocorr | 2.221 | read_iops_mean | 1.591 |
| 576 -> 577 | total_iops_lag1_autocorr | 2.194 | read_iops_mean | 1.573 | read_iops_q90 | 1.414 |
| 577 -> 578 | write_share_iops_mean | 1.942 | read_iops_q99 | 1.414 | ts_duration | 1.331 |
| 578 -> 579 | write_share_iops_mean | 1.755 | ts_duration | 1.414 | read_iops_q99 | 1.414 |
| 579 -> 580 | write_bw_q90 | 1.564 | total_bw_q50 | 1.488 | write_bw_q50 | 1.465 |
| 580 -> 581 | ts_duration | 4.243 | write_bw_mean | 1.593 | write_bw_q50 | 1.519 |
| 581 -> 582 | idle_ratio | 6.014 | write_iops_q90 | 2.443 | read_iops_q99 | 2.343 |
| 582 -> 583 | idle_ratio | 13.437 | size_bytes | 3.628 | write_iops_q90 | 2.443 |
| 583 -> 584 | idle_ratio | 18.443 | total_iops_lag1_autocorr | 1.914 | read_iops_q99 | 1.414 |
| 584 -> 585 | write_bw_q50 | 19.038 | read_bw_q50 | 16.606 | total_bw_q50 | 16.169 |
| 585 -> 586 | disk_usage_mean | 2.302 | write_share_iops_mean | 1.921 | disk_usage_q50 | 1.828 |
| 586 -> 587 | write_share_iops_mean | 2.041 | disk_usage_mean | 1.986 | disk_usage_q50 | 1.897 |
| 587 -> 588 | disk_usage_q90 | 3.799 | disk_usage_mean | 2.581 | size_bytes | 2.502 |
| 588 -> 589 | disk_usage_q90 | 3.518 | disk_usage_mean | 2.498 | size_bytes | 2.31 |
| 589 -> 590 | idle_ratio | 1.62 | write_iops_q90 | 1.414 | total_iops_q90 | 1.414 |
| 590 -> 591 | read_iops_q99 | 1.31 | total_iops_q99 | 1.102 | write_iops_q99 | 1.042 |
| 591 -> 592 | total_iops_lag1_autocorr | 6.481 | write_iops_mean | 2.152 | total_iops_mean | 1.954 |
| 592 -> 593 | total_iops_lag1_autocorr | 6.901 | write_iops_mean | 1.824 | total_iops_mean | 1.63 |
| 593 -> 594 | total_bw_mean | 1.103 | write_bw_q50 | 1.053 | write_bw_mean | 0.93 |
| 594 -> 595 | size_bytes | 0.688 | total_iops_lag1_autocorr | 0.618 | disk_usage_q50 | 0.525 |
| 595 -> 596 | write_iops_mean | 1.181 | disk_usage_q50 | 1.109 | disk_usage_mean | 1.049 |
| 596 -> 597 | write_iops_mean | 1.294 | total_iops_mean | 0.993 | write_share_iops_mean | 0.914 |
| 597 -> 598 | write_bw_q50 | 2.693 | total_bw_q50 | 2.501 | write_iops_mean | 2.119 |
| 598 -> 599 | write_bw_q50 | 4.459 | total_bw_q50 | 2.782 | write_iops_mean | 1.881 |
| 599 -> 600 | write_iops_q90 | 1.414 | total_iops_q90 | 1.414 | idle_ratio | 0.923 |
| 600 -> 601 | size_bytes | 1.212 | write_bw_q50 | 1.049 | write_bw_mean | 1.035 |
| 601 -> 602 | total_iops_q90 | 1.414 | write_iops_q90 | 1.414 | total_iops_lag1_autocorr | 1.25 |
| 602 -> 603 | total_iops_lag1_autocorr | 1.515 | total_iops_q90 | 1.414 | write_iops_q90 | 1.414 |
| 603 -> 604 | disk_usage_q50 | 4.202 | disk_usage_mean | 3.499 | disk_usage_q90 | 3.138 |
| 604 -> 605 | disk_usage_q50 | 3.131 | disk_usage_mean | 2.82 | disk_usage_q90 | 2.353 |
| 605 -> 606 | total_bw_q50 | 11.314 | read_bw_q50 | 9.9 | size_bytes | 4.086 |
| 606 -> 607 | read_bw_q50 | 9.428 | total_bw_q50 | 5.354 | size_bytes | 4.872 |
| 607 -> 608 | total_iops_lag1_autocorr | 0.855 | read_iops_mean | 0.74 | disk_usage_q50 | 0.728 |
| 608 -> 609 | write_bw_q50 | 1.576 | write_bw_mean | 1.385 | read_iops_q99 | 1.317 |
| 609 -> 610 | read_iops_q99 | 1.274 | total_bw_mean | 1.113 | write_bw_q50 | 1.05 |
| 610 -> 611 | read_iops_q99 | 0.972 | write_bw_q50 | 0.724 | total_iops_q99 | 0.638 |
| 611 -> 612 | write_bw_q90 | 0.446 | total_bw_q50 | 0.414 | read_bw_q50 | 0.392 |
| 612 -> 613 | write_bw_mean | 2.096 | write_bw_q90 | 1.902 | write_share_iops_mean | 1.681 |
| 613 -> 614 | write_share_iops_mean | 1.486 | write_bw_mean | 1.431 | write_iops_q90 | 1.414 |
| 614 -> 615 | write_bw_mean | 2.541 | total_iops_lag1_autocorr | 1.519 | read_iops_mean | 1.24 |
| 615 -> 616 | write_bw_mean | 3.966 | total_bw_mean | 2.505 | write_bw_q90 | 1.827 |
| 616 -> 617 | read_iops_q99 | 1.229 | write_iops_q99 | 1.164 | total_iops_q99 | 1.103 |
| 617 -> 618 | read_iops_q99 | 1.229 | write_iops_q99 | 1.141 | total_iops_q99 | 1.103 |
| 618 -> 619 | idle_ratio | 1.47 | total_iops_lag1_autocorr | 1.425 | total_iops_mean | 1.414 |
| 619 -> 620 | total_iops_q99 | 1.414 | total_iops_lag1_autocorr | 1.365 | read_iops_mean | 1.238 |
| 620 -> 621 | total_iops_q99 | 1.414 | total_bw_q50 | 1.231 | total_iops_lag1_autocorr | 1.222 |
| 621 -> 622 | read_bw_q90 | 1.601 | total_bw_q90 | 1.524 | read_bw_q50 | 1.495 |
| 622 -> 623 | read_iops_mean | 1.417 | read_iops_q99 | 1.414 | write_iops_q90 | 1.414 |
| 623 -> 624 | write_iops_q90 | 1.414 | read_iops_q90 | 1.414 | total_iops_q90 | 1.414 |
| 624 -> 625 | write_share_iops_mean | 1.341 | read_bw_mean | 1.054 | size_bytes | 0.86 |
| 625 -> 626 | read_iops_q99 | 2.65 | write_bw_q90 | 2.229 | idle_ratio | 1.791 |
| 626 -> 627 | total_iops_q90 | 1.414 | write_iops_q90 | 1.414 | size_bytes | 1.395 |
| 627 -> 628 | total_iops_q90 | 1.414 | write_iops_q90 | 1.414 | size_bytes | 1.37 |
| 628 -> 629 | read_bw_q50 | 3.941 | read_iops_q99 | 2.179 | write_share_iops_mean | 2.109 |
| 629 -> 630 | read_iops_q99 | 2.121 | write_share_iops_mean | 1.946 | idle_ratio | 1.558 |
| 630 -> 631 | size_bytes | 37.95 | total_iops_lag1_autocorr | 1.742 | total_iops_q90 | 1.414 |
| 631 -> 632 | size_bytes | 7.299 | total_iops_lag1_autocorr | 1.621 | total_iops_q90 | 1.414 |
| 632 -> 633 | size_bytes | 7.079 | total_iops_lag1_autocorr | 1.636 | read_bw_q50 | 1.446 |
| 633 -> 634 | write_bw_q50 | 1.696 | read_bw_q50 | 1.635 | total_bw_q50 | 1.612 |
| 634 -> 635 | size_bytes | 7.944 | disk_usage_q90 | 2.03 | disk_usage_mean | 1.736 |
| 635 -> 636 | size_bytes | 9.517 | disk_usage_q90 | 3.328 | disk_usage_mean | 3.299 |
| 636 -> 637 | total_iops_lag1_autocorr | 1.927 | read_bw_q90 | 1.479 | read_bw_mean | 1.455 |
| 637 -> 638 | disk_usage_q90 | 1.768 | size_bytes | 1.632 | write_iops_mean | 1.446 |
| 638 -> 639 | total_iops_lag1_autocorr | 2.506 | write_bw_q90 | 2.404 | write_bw_q50 | 2.236 |
| 639 -> 640 | total_iops_lag1_autocorr | 4.378 | read_iops_q99 | 1.774 | total_iops_q99 | 1.744 |
| 640 -> 641 | total_iops_q90 | 30.641 | total_iops_q99 | 20.005 | write_iops_q99 | 18.242 |
| 641 -> 642 | total_iops_q90 | 30.641 | total_iops_lag1_autocorr | 18.991 | write_iops_mean | 17.014 |
| 642 -> 643 | read_iops_mean | 1.523 | total_iops_lag1_autocorr | 1.391 | total_iops_mean | 1.074 |
| 643 -> 644 | read_iops_mean | 3.412 | total_iops_mean | 1.83 | write_iops_mean | 1.468 |
| 644 -> 645 | read_iops_q99 | 1.414 | write_iops_q90 | 1.414 | read_iops_q90 | 1.414 |
| 645 -> 646 | read_iops_mean | 1.415 | total_iops_mean | 1.414 | read_iops_q99 | 1.414 |
| 646 -> 647 | idle_ratio | 35.35 | write_iops_mean | 2.375 | disk_usage_q50 | 1.697 |
| 647 -> 648 | disk_usage_mean | 1.414 | disk_usage_q50 | 1.414 | disk_usage_q90 | 1.414 |
| 648 -> 649 | idle_ratio | 1.428 | write_iops_q99 | 1.414 | total_iops_q99 | 1.414 |
| 649 -> 650 | disk_usage_mean | 1.414 | disk_usage_q50 | 1.414 | disk_usage_q90 | 1.414 |
| 650 -> 651 | write_bw_q90 | 1.414 | disk_usage_mean | 1.414 | disk_usage_q50 | 1.414 |
| 651 -> 652 | write_iops_q99 | 1.414 | total_iops_q99 | 1.414 | write_bw_q50 | 1.414 |
| 652 -> 653 | size_bytes | 1.799 | write_bw_q50 | 1.414 | write_bw_q90 | 1.342 |
| 653 -> 654 | disk_usage_q50 | 2.232 | disk_usage_mean | 1.861 | disk_usage_q90 | 1.821 |
| 654 -> 655 | total_iops_lag1_autocorr | 1.463 | read_iops_q99 | 1.414 | total_iops_q99 | 1.414 |
| 655 -> 656 | write_iops_q90 | 1.414 | total_iops_q90 | 1.414 | read_iops_q99 | 1.374 |
| 656 -> 657 | size_bytes | 1.011 | read_iops_q99 | 0.788 | total_iops_lag1_autocorr | 0.706 |
| 657 -> 658 | write_iops_q99 | 4.331 | total_iops_q99 | 3.268 | total_iops_lag1_autocorr | 3.035 |
| 658 -> 659 | idle_ratio | 1.229 | read_iops_q50 | 1.05 | size_bytes | 1.039 |
| 659 -> 660 | idle_ratio | 14.568 | total_iops_q99 | 11.793 | write_iops_q99 | 10.37 |
| 660 -> 661 | write_iops_q99 | 5.646 | total_iops_q99 | 3.305 | read_iops_q99 | 1.996 |
| 661 -> 662 | write_iops_q99 | 6.367 | total_iops_q99 | 3.305 | read_iops_q99 | 2.047 |
| 662 -> 663 | total_iops_lag1_autocorr | 4.928 | size_bytes | 2.113 | write_iops_mean | 1.466 |
| 663 -> 664 | total_iops_lag1_autocorr | 2.801 | read_bw_q50 | 2.333 | total_bw_q50 | 1.888 |
| 664 -> 665 | read_bw_mean | 16.21 | total_bw_mean | 1.618 | write_share_iops_mean | 1.491 |
| 665 -> 666 | read_bw_mean | 11.624 | total_bw_mean | 1.611 | write_bw_mean | 1.475 |

## Strongest Correlations

| Metric A | Metric B | Correlation |
|---|---|---:|
| disk_usage_mean | disk_usage_q50 | 0.992 |
| write_iops_q99 | total_iops_q99 | 0.992 |
| write_iops_q90 | total_iops_q90 | 0.986 |
| write_iops_mean | total_iops_mean | 0.981 |
| write_iops_q50 | total_iops_q50 | 0.979 |
| read_iops_mean | read_iops_q50 | 0.975 |
| disk_usage_mean | disk_usage_q90 | 0.972 |
| disk_usage_q50 | disk_usage_q90 | 0.962 |

## Metrics

| Metric | Mean | Median | CV | Skew | Kurtosis | Missing | Q10 | Q90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| write_iops_q50 | 91.756 | 0 | 18.994 | 38.016 | 1942.171 | 0 | 0 | 5 |
| read_iops_q50 | 34.682 | 0 | 17.811 | 25.687 | 750.971 | 0 | 0 | 1 |
| total_iops_q50 | 126.763 | 0 | 17.477 | 30.964 | 1285.027 | 0 | 0 | 5 |
| read_iops_mean | 50.577 | 1.226 | 12.323 | 23.96 | 680.125 | 0 | 0.009 | 22.226 |
| write_iops_mean | 161.3 | 3.477 | 11.439 | 31.919 | 1496.346 | 0 | 0.023 | 58.399 |
| total_iops_mean | 211.877 | 4.964 | 10.947 | 26.854 | 1039.733 | 0 | 0.033 | 82.164 |
| read_iops_q90 | 92.984 | 0 | 9.721 | 16.129 | 304.732 | 0 | 0 | 30 |
| write_iops_q90 | 355.159 | 0 | 9.278 | 15.089 | 305.127 | 0 | 0 | 54 |
| total_iops_q90 | 447.516 | 0 | 8.817 | 14.119 | 263.386 | 0 | 0 | 87.5 |
| write_bw_q50 | 164.848 | 18 | 7.482 | 20.399 | 688.161 | 0 | 0 | 108 |
| read_bw_q90 | 212.087 | 51 | 7.204 | 15.678 | 264.985 | 0 | 1 | 231 |
| read_bw_q50 | 93.676 | 42 | 6.969 | 31.528 | 1108.153 | 0 | 0 | 155 |
| write_iops_q99 | 922.225 | 22.05 | 6.48 | 12.485 | 199.069 | 0 | 0 | 799.95 |
| total_bw_q50 | 259.608 | 60 | 6.166 | 17.811 | 439.668 | 0 | 0 | 282 |
| total_iops_q99 | 1161.729 | 36.175 | 6.082 | 11.677 | 173.179 | 0 | 0 | 1132.375 |
| read_iops_q99 | 248.075 | 11 | 5.9 | 11.302 | 158.98 | 0 | 0 | 294.225 |
| write_bw_mean | 217.492 | 31.136 | 5.572 | 17.983 | 554.08 | 0 | 0.112 | 234.676 |
| read_bw_mean | 115.822 | 44.857 | 5.518 | 21.308 | 561.773 | 0 | 0.526 | 169.865 |

## Outlier Files

| rel_path | outlier_score | top drivers |
|---|---:|---|
| tencent/Cloud_Disk_dataset/disk_load_data/e2f36e2d-bbb1-40b4-a244-217dc1e858b2 | 3628.767 | disk_usage_mean (z=1015.145); disk_usage_q50 (z=943.667) |
| tencent/Cloud_Disk_dataset/disk_load_data/0542e5a2-5d01-4384-9855-377bc518bfee | 2544.877 | write_bw_q90 (z=1227.048); write_iops_q99 (z=1201.247) |
| tencent/Cloud_Disk_dataset/disk_load_data/90f2be84-2425-4bd6-b811-a18e54606acd | 1789.964 | write_iops_mean (z=31386.37); total_iops_mean (z=24957.62) |
| tencent/Cloud_Disk_dataset/disk_load_data/5c52932a-00a3-408b-84e5-a68fb2c6ec9f | 1595.924 | write_iops_mean (z=31180.8); total_iops_mean (z=24123.63) |
| tencent/Cloud_Disk_dataset/disk_load_data/cff8ae8d-e084-4c31-8c93-5262ddc5af30 | 1360.473 | write_iops_q99 (z=6964.578); total_iops_q99 (z=4331.773) |
| tencent/Cloud_Disk_dataset/disk_load_data/c8ad9779-f1bb-4ed2-9437-fa8503a19a9d | 1258.802 | write_iops_mean (z=8154.778); total_iops_mean (z=6529.007) |
| tencent/Cloud_Disk_dataset/disk_load_data/59d89b6e-c4b0-4df5-bcb8-0b650f770d33 | 791.352 | write_iops_mean (z=16109.07); total_iops_mean (z=14190.67) |
| tencent/Cloud_Disk_dataset/disk_load_data/e8cc7217-02a7-41ce-afc1-09f9f802ebcb | 755.589 | write_bw_q90 (z=818.619); write_iops_q99 (z=630.708) |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv | ts_duration |
|---|---|---:|---:|---:|---:|
| tencent/Cloud_Disk_dataset/disk_load_data/00076732-908a-4dd1-9917-b4a9d8e5b572 | tencent_cloud_disk | N/A | N/A | N/A | 1228500 |
| tencent/Cloud_Disk_dataset/disk_load_data/0013006a-f3ff-48ba-8378-da10670eff26 | tencent_cloud_disk | N/A | N/A | N/A | 1228500 |
| tencent/Cloud_Disk_dataset/disk_load_data/00141396-c9ca-4c12-ad94-69749e435b26 | tencent_cloud_disk | N/A | N/A | N/A | 1228500 |
| tencent/Cloud_Disk_dataset/disk_load_data/001418d5-0d5d-43f7-9234-5ea9c4036571 | tencent_cloud_disk | N/A | N/A | N/A | 1228500 |
| tencent/Cloud_Disk_dataset/disk_load_data/001bfa72-3dd6-49e5-aad2-9ee27c48530d | tencent_cloud_disk | N/A | N/A | N/A | 1228500 |
| tencent/Cloud_Disk_dataset/disk_load_data/001f1661-72fe-4800-a06a-4b17723ffc03 | tencent_cloud_disk | N/A | N/A | N/A | 1228500 |
| tencent/Cloud_Disk_dataset/disk_load_data/001f31fe-e8fd-4fba-86b4-5a0f78be94a0 | tencent_cloud_disk | N/A | N/A | N/A | 1228500 |
| tencent/Cloud_Disk_dataset/disk_load_data/00223656-80e7-48b6-a12e-a0ec751464b8 | tencent_cloud_disk | N/A | N/A | N/A | 1228500 |




## Model-Aware Guidance

- Closest learned anchor: alibaba (distance 4.56)
- Sampling: random-ok
- Regime recipe: K≈8
- Char-file conditioning: yes
- PCF: promising
- Multi-scale critic: promising
- Mixed-type recovery: not-primary
- Retrieval memory: not-primary
- Why: family looks multi-regime or high-heterogeneity
