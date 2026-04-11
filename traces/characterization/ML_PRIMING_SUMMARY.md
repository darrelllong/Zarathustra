# Trace Characterization Summary

- Inventory files: 31701
- Trace candidates: 30628
- Characterized files: 30628
- ML use cases: {"aggregate_time_series": 16296, "generic_sequence_or_log": 575, "inventory_only": 25, "request_sequence": 13732}
- Parsers: {"baleen24": 374, "exchange_etw": 96, "generic_text": 575, "lcs": 6137, "lcs_stub": 3, "oracle_general": 7122, "parquet_stub": 25, "tencent_cloud_disk": 16296}

## Group Highlights

### Baleen24/Baleen24/baleen24
- files: 374
- bytes: 31376065904
- write_ratio median: 0.6455078125
- reuse_ratio median: 0.3108669108669109
- burstiness_cv median: 1.4151548238544374

### MSR/exchange/exchange_etw
- files: 96
- bytes: 1479416826
- write_ratio median: 0.7861328125
- reuse_ratio median: 0.002686202686202686
- burstiness_cv median: 4.532285443249396

### alibaba/alibaba/oracle_general
- files: 1000
- bytes: 100765044913
- write_ratio median: 0.0
- reuse_ratio median: 0.0002442002442002442
- burstiness_cv median: 3.8326010867456564

### s3-cache-datasets/2007_msr/oracle_general
- files: 14
- bytes: 1637389703
- write_ratio median: 0.0
- reuse_ratio median: 0.005372405372405372
- burstiness_cv median: 7.337830060719586

### s3-cache-datasets/2007_wiki/text_zst
- files: 4
- bytes: 39128779019

### s3-cache-datasets/2015_cloudphysics/text_zst
- files: 106
- bytes: 9026863149

### s3-cache-datasets/2016_wiki/text
- files: 1
- bytes: 95213049

### s3-cache-datasets/2016_wiki/text_zst
- files: 1
- bytes: 26491912192

### s3-cache-datasets/2017_docker/parquet
- files: 7
- bytes: 600585203

### s3-cache-datasets/2017_systor/text_zst
- files: 6
- bytes: 34993572898

### s3-cache-datasets/2018_tencentPhoto/oracle_general
- files: 2
- bytes: 72132172910
- write_ratio median: 0.0
- reuse_ratio median: 0.009035409035409036
- burstiness_cv median: 28.600699292150185

### s3-cache-datasets/2018_tencentPhoto/text_zst
- files: 2
- bytes: 90918912199

### s3-cache-datasets/2019_wiki/oracle_general
- files: 3
- bytes: 53649628938
- write_ratio median: 0.0
- reuse_ratio median: 0.0014652014652014652
- burstiness_cv median: 45.23825814507009

### s3-cache-datasets/2019_wiki/text
- files: 1
- bytes: 65605391

### s3-cache-datasets/2019_wiki/text_zst
- files: 2
- bytes: 25760060833

### s3-cache-datasets/2020_alibabaBlock/oracle_general
- files: 1000
- bytes: 100786283892
- write_ratio median: 0.0
- reuse_ratio median: 0.0002442002442002442
- burstiness_cv median: 3.8326010867456564

### s3-cache-datasets/2020_alibabaBlock/text_zst
- files: 1
- bytes: 150421221737

### s3-cache-datasets/2020_tencentBlock/oracle_general
- files: 4995
- bytes: 163244704467
- write_ratio median: 0.0
- reuse_ratio median: 0.010256410256410256
- burstiness_cv median: 2.9582028184317846

### s3-cache-datasets/2020_tencentBlock/text_zst
- files: 1
- bytes: 137224724902

### s3-cache-datasets/2020_twitter/text_zst
- files: 54
- bytes: 152248401229

### s3-cache-datasets/2020_twr_cdn/parquet
- files: 2
- bytes: 293223291415

### s3-cache-datasets/2020_twr_cdn.zst/text_zst
- files: 1
- bytes: 231608521285

### s3-cache-datasets/2022_metaCDN/oracle_general
- files: 3
- bytes: 2234960001
- write_ratio median: 0.0
- reuse_ratio median: 0.0844932844932845
- burstiness_cv median: 26.06386558397561

### s3-cache-datasets/2022_metaKV/oracle_general
- files: 4
- bytes: 65307778508
- write_ratio median: 0.0
- reuse_ratio median: 0.914041514041514
- burstiness_cv median: 21.307275752662516

### s3-cache-datasets/2022_metaKV/text_zst
- files: 24
- bytes: 85515981910

### s3-cache-datasets/2022_metaStorage/text_zst
- files: 5
- bytes: 616636804

### s3-cache-datasets/2023_metaCDN/text_zst
- files: 3
- bytes: 3731742149

### s3-cache-datasets/2023_metaStorage/text
- files: 1
- bytes: 1554951754

### s3-cache-datasets/2023_metaStorage/text_zst
- files: 5
- bytes: 1329098047

### s3-cache-datasets/2024_google/parquet
- files: 6
- bytes: 66153501928

### s3-cache-datasets/2024_google/text_zst
- files: 186
- bytes: 118301931428

### s3-cache-datasets/2026-alibaba-thinkahead/oracle_general
- files: 45
- bytes: 15366738380
- write_ratio median: 0.0
- reuse_ratio median: 0.026617826617826617
- burstiness_cv median: 26.105554964413226

### s3-cache-datasets/alibaba/lcs
- files: 1000
- bytes: 477220451258
- write_ratio median: 0.975830078125
- reuse_ratio median: 0.0007326007326007326
- burstiness_cv median: 6.3993878267121955

### s3-cache-datasets/cache_trace_twitter_memcache/oracle_general
- files: 54
- bytes: 1735611432504
- write_ratio median: 0.0
- reuse_ratio median: 0.006837606837606838
- burstiness_cv median: 45.23825814507009

### s3-cache-datasets/cache_trace_twitter_memcache/parquet
- files: 10
- bytes: 179182912738

### s3-cache-datasets/cache_trace_twitter_memcache/text_zst
- files: 162
- bytes: 2264718133466

### s3-cache-datasets/cloudphysics/lcs
- files: 106
- bytes: 64319531250
- write_ratio median: 0.86279296875
- reuse_ratio median: 0.003418803418803419
- burstiness_cv median: 8.924473218191773

### s3-cache-datasets/cloudphysics/text
- files: 1
- bytes: 986

### s3-cache-datasets/metaKV/lcs
- files: 3
- bytes: 26152938221

### s3-cache-datasets/metaKV/text_zst
- files: 8
- bytes: 7126300497

### s3-cache-datasets/msr/lcs
- files: 36
- bytes: 8130400717
- write_ratio median: 0.919921875
- reuse_ratio median: 0.0004884004884004884
- burstiness_cv median: 8.696346043443816

### s3-cache-datasets/other/oracle_general
- files: 2
- bytes: 386344048303
- write_ratio median: 0.017333984375
- reuse_ratio median: 0.050793650793650794
- burstiness_cv median: 63.984373092185564

### s3-cache-datasets/tencentBlock/lcs
- files: 4995
- bytes: 817117209599
- write_ratio median: 0.98046875
- reuse_ratio median: 0.004395604395604396
- burstiness_cv median: 4.51412153008547

### tencent/cloud_disk/tencent_cloud_disk
- files: 16296
- bytes: 7270989623
- idle_ratio median: 0.90673828125
- total_iops q50 median: 0.0
