# s3-cache-datasets / 2007_wiki

- Files: 4
- Bytes: 39128779019
- Formats: text_zst
- Parsers: wiki_hash_text

## Observations

- Very weak short-window reuse.
- Highly bursty arrivals.

## Format Breakdown

| Format | Files | Parsers |
|---|---:|---|
| text_zst | 4 | wiki_hash_text |

## Metrics

| Metric | Mean | Median | Min | Max | SD |
|---|---:|---:|---:|---:|---:|
| reuse_ratio | 0.019 | 0.027 | 0.002 | 0.029 | 0.015 |
| burstiness_cv | 37.5 | 40.529 | 4.959 | 63.984 | 30.978 |
| iat_q50 | 0 | 0 | 0 | 0 | 0 |
| iat_q90 | 0 | 0 | 0 | 0 | 0 |
| iat_lag1_autocorr | -0.011 | -0.002 | -0.041 | 0 | 0.02 |
| forward_seek_ratio | 0.491 | 0.488 | 0.484 | 0.501 | 0.009 |
| backward_seek_ratio | 0.49 | 0.489 | 0.483 | 0.496 | 0.007 |

## Notable Files

| rel_path | format | write_ratio | reuse_ratio | burstiness_cv |
|---|---|---:|---:|---:|
| s3-cache-datasets/cache_dataset_txt/2007_wiki/wiki/2007/wiki.2007.sort.hash.csv.zst | text_zst | N/A | 0.002 | 63.984 |
| s3-cache-datasets/cache_dataset_txt/2007_wiki/wiki/2007/wiki.2007.sort.raw.csv.zst | text_zst | N/A | N/A | 63.984 |
| s3-cache-datasets/cache_dataset_txt/2007_wiki/wiki/2007/wiki.2007.sort.hash.sample10.csv.zst | text_zst | N/A | 0.027 | 17.073 |
| s3-cache-datasets/cache_dataset_txt/2007_wiki/wiki/2007/wiki.2007.sort.hash.sample100.csv.zst | text_zst | N/A | 0.029 | 4.959 |
