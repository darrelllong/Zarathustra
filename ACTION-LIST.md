# Action List

## Immediate Priorities

1. Treat `tencent__cloud_disk`, `alibaba__alibaba`, `s3-cache-datasets__2020_tencentBlock`, `s3-cache-datasets__2020_alibabaBlock`, `s3-cache-datasets__alibaba`, `s3-cache-datasets__tencentBlock`, and `s3-cache-datasets__cache_trace_twitter_memcache` as first-tier GAN targets for segmented or regime-aware training.
2. Split mixed-format families before GAN training instead of pooling them blindly:
   `s3-cache-datasets__cache_trace_twitter_memcache`, `s3-cache-datasets__2020_tencentBlock`, `s3-cache-datasets__2020_alibabaBlock`, `s3-cache-datasets__2018_tencentPhoto`, `s3-cache-datasets__2019_wiki`, `s3-cache-datasets__2020_twr_cdn`, `s3-cache-datasets__2022_metaKV`, `s3-cache-datasets__cloudphysics`, `s3-cache-datasets__2016_wiki`, `s3-cache-datasets__2023_metaStorage`, `s3-cache-datasets__2024_google`.
3. Use the `suggested_modes` values from [characterizations/families.csv](/Users/darrell/Zarathustra/characterizations/families.csv) as the starting point for mode count or regime count, not as a hard truth.
4. Prioritize locality-aware conditioning and losses for `s3-cache-datasets__metaKV` and `s3-cache-datasets__2022_metaKV`.
5. Prioritize burst-sensitive inter-arrival, FFT, and ACF objectives for `s3-cache-datasets__cache_trace_twitter_memcache`, `s3-cache-datasets__2020_twitter`, `s3-cache-datasets__2007_wiki`, `s3-cache-datasets__2018_tencentPhoto`, and `s3-cache-datasets__2022_metaCDN`.

## Family Tiers

- Tier 1, multi-mode or regime-heavy:
  `tencent__cloud_disk`, `alibaba__alibaba`, `s3-cache-datasets__2020_tencentBlock`, `s3-cache-datasets__2020_alibabaBlock`, `s3-cache-datasets__alibaba`, `s3-cache-datasets__tencentBlock`, `s3-cache-datasets__cache_trace_twitter_memcache`, `MSR__exchange`, `Baleen24__Baleen24`, `s3-cache-datasets__2015_cloudphysics`, `s3-cache-datasets__msr`.
- Tier 2, strong but simpler:
  `s3-cache-datasets__2020_twitter`, `s3-cache-datasets__metaKV`, `s3-cache-datasets__2022_metaKV`, `s3-cache-datasets__2026-alibaba-thinkahead`, `s3-cache-datasets__2017_systor`, `s3-cache-datasets__2007_msr`.
- Tier 3, structured-table or lower-signal families:
  `s3-cache-datasets__2017_docker`, `s3-cache-datasets__2024_google`, `s3-cache-datasets__2023_metaCDN`, `s3-cache-datasets__2023_metaStorage`, `s3-cache-datasets__2020_twr_cdn`, `s3-cache-datasets__2016_wiki`.

## Concrete Training Moves

1. Build separate experiment queues for:
   `cache_trace_twitter_memcache`
   `2020_tencentBlock`
   `2020_alibabaBlock`
   `alibaba`
   `tencentBlock`
   `Baleen24`
   `exchange`
2. For mixed-format families, train one model per encoding first, then decide whether cross-format distillation is worth doing.
3. For `tencent__cloud_disk`, do not treat it like a plain request-sequence family. Use regime segmentation or clustered subsets.
4. For read-dominated families, do not assume balanced opcode priors.
5. For write-heavy families like `Baleen24`, `MSR__exchange`, `s3-cache-datasets__alibaba`, and `s3-cache-datasets__tencentBlock`, keep opcode-transition fidelity as a first-class metric.
6. Inspect outlier traces called out in the family reports before folding them into benchmark sets.

## Rebuttal-Driven Follow-Ups

Completed in the current R pass:

- silhouette-based K diagnostics
- per-regime feature attribution tables
- feature-level outlier decomposition
- top-N outlier removal sensitivity summaries
- block-vs-random temporal sampling diagnostics
- conditioning audits for redundancy, near-constant dimensions, and the 3 candidate additions

Still pending:

1. Expand the candidate conditioning set for the GAN analysis notebook or next training plan to include:
   `object_unique`, `signed_stride_lag1_autocorr`, and `obj_size_std`.
2. Audit conditioning redundancy specifically for `tencentBlock` and `alibaba`:
   `backward_seek_ratio` is largely determined by `1 - forward_seek_ratio - reuse_ratio`, and `opcode_switch_ratio` appears low-signal in the main corpora.
3. Add per-regime feature attribution on top of changepoints so we can say what shifts at each regime boundary instead of only counting boundaries.
4. Add outlier decomposition so every flagged outlier is explained by feature contribution, not only by a Mahalanobis score.
5. Test block or sequential file sampling for `alibaba`-like high-persistence families instead of assuming file-level i.i.d. sampling is harmless.
6. Add K-ablation experiments around the suggested mode count instead of treating the current `suggested_modes` heuristic as final truth.
7. Add within-file window analysis for the top GAN corpora, because file-level summaries do not fully describe window-level generation difficulty.

## Files To Use

- Rollup: [characterizations/README.md](/Users/darrell/Zarathustra/characterizations/README.md)
- Family table: [characterizations/families.csv](/Users/darrell/Zarathustra/characterizations/families.csv)
- Compact summaries: [characterizations/FAMILY-SUMMARIES.md](/Users/darrell/Zarathustra/characterizations/FAMILY-SUMMARIES.md)
- Per-family reports: [characterizations/families](/Users/darrell/Zarathustra/characterizations/families)
