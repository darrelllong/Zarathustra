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

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| tencent_cloud_disk | 16296 | tencent_cloud_disk |

## Clustering And Regimes

| Item | Value |
|---|---|
| DBSCAN clusters | 3 |
| DBSCAN noise fraction | 0.072 |
| Ordered PC1 changepoints | 665 |
| PCA variance explained by PC1 | 0.343 |

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

| rel_path | outlier_score |
|---|---:|
| tencent/Cloud_Disk_dataset/disk_load_data/e2f36e2d-bbb1-40b4-a244-217dc1e858b2 | 3628.767 |
| tencent/Cloud_Disk_dataset/disk_load_data/0542e5a2-5d01-4384-9855-377bc518bfee | 2544.877 |
| tencent/Cloud_Disk_dataset/disk_load_data/90f2be84-2425-4bd6-b811-a18e54606acd | 1789.964 |
| tencent/Cloud_Disk_dataset/disk_load_data/5c52932a-00a3-408b-84e5-a68fb2c6ec9f | 1595.924 |
| tencent/Cloud_Disk_dataset/disk_load_data/cff8ae8d-e084-4c31-8c93-5262ddc5af30 | 1360.473 |
| tencent/Cloud_Disk_dataset/disk_load_data/c8ad9779-f1bb-4ed2-9437-fa8503a19a9d | 1258.802 |
| tencent/Cloud_Disk_dataset/disk_load_data/59d89b6e-c4b0-4df5-bcb8-0b650f770d33 | 791.352 |
| tencent/Cloud_Disk_dataset/disk_load_data/e8cc7217-02a7-41ce-afc1-09f9f802ebcb | 755.589 |

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
