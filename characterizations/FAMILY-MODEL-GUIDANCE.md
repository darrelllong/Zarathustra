# Family Model Guidance

This report maps the validated and frontier model learnings from the Tencent and Alibaba training corpora back onto every logical trace family.

## alibaba / alibaba

- Closest learned anchor: alibaba (distance 0)
- Family kind: request_sequence
- Sampling: block
- Regime recipe: K≈8
- Char-file conditioning: yes
- PCF status: promising
- Multi-scale critic status: promising
- Mixed-type recovery status: mixed
- Retrieval memory status: unknown
- Candidate conditioning additions: object_unique,signed_stride_lag1_autocorr,obj_size_std

### Why

- ordered files show temporal persistence; family looks multi-regime or high-heterogeneity

### Defaults

- use block or sequential file sampling
- use char-file conditioning
- use a regime sampler around K≈8

### Candidates

- PCF loss is promising on the alibaba corpus
- multi-scale critic looks promising for higher-mode families
- test extra conditioning features: object_unique,signed_stride_lag1_autocorr,obj_size_std

## Baleen24 / Baleen24

- Closest learned anchor: tencent_block (distance 111.661)
- Family kind: request_sequence
- Sampling: block
- Regime recipe: K≈8
- Char-file conditioning: yes
- PCF status: validated
- Multi-scale critic status: promising
- Mixed-type recovery status: promising
- Retrieval memory status: mixed
- Candidate conditioning additions: object_unique,signed_stride_lag1_autocorr,obj_size_std

### Why

- ordered files show temporal persistence; family looks multi-regime or high-heterogeneity; reuse/locality is not negligible

### Defaults

- use block or sequential file sampling
- use char-file conditioning
- use a regime sampler around K≈8
- treat locality as first-class in the loss/conditioning stack

### Candidates

- PCF loss is validated on the tencent_block corpus
- multi-scale critic looks promising for higher-mode families
- mixed-type recovery is promising for request-sequence windows
- test extra conditioning features: object_unique,signed_stride_lag1_autocorr,obj_size_std

## MSR / exchange

- Closest learned anchor: tencent_block (distance 1.81)
- Family kind: request_sequence
- Sampling: block
- Regime recipe: K≈8
- Char-file conditioning: yes
- PCF status: validated
- Multi-scale critic status: promising
- Mixed-type recovery status: promising
- Retrieval memory status: mixed
- Candidate conditioning additions: object_unique,signed_stride_lag1_autocorr,obj_size_std

### Why

- ordered files show temporal persistence; family looks multi-regime or high-heterogeneity

### Defaults

- use block or sequential file sampling
- use char-file conditioning
- use a regime sampler around K≈8

### Candidates

- PCF loss is validated on the tencent_block corpus
- multi-scale critic looks promising for higher-mode families
- mixed-type recovery is promising for request-sequence windows
- test extra conditioning features: object_unique,signed_stride_lag1_autocorr,obj_size_std

## s3-cache-datasets / 2007_msr

- Closest learned anchor: tencent_block (distance 1.102)
- Family kind: request_sequence
- Sampling: random-ok
- Regime recipe: single
- Char-file conditioning: yes
- PCF status: validated
- Multi-scale critic status: promising
- Mixed-type recovery status: promising
- Retrieval memory status: mixed
- Candidate conditioning additions: object_unique,signed_stride_lag1_autocorr,obj_size_std

### Why

- burstiness is materially above the calmer families

### Defaults

- use char-file conditioning
- keep burst-sensitive temporal objectives on

### Candidates

- PCF loss is validated on the tencent_block corpus
- mixed-type recovery is promising for request-sequence windows
- test extra conditioning features: object_unique,signed_stride_lag1_autocorr,obj_size_std

## s3-cache-datasets / 2007_wiki

- Closest learned anchor: tencent_block (distance 3.396)
- Family kind: request_sequence
- Sampling: random-ok
- Regime recipe: single
- Char-file conditioning: yes
- PCF status: validated
- Multi-scale critic status: promising
- Mixed-type recovery status: promising
- Retrieval memory status: mixed
- Candidate conditioning additions: object_unique,signed_stride_lag1_autocorr

### Why

- burstiness is materially above the calmer families

### Defaults

- use char-file conditioning
- keep burst-sensitive temporal objectives on

### Candidates

- PCF loss is validated on the tencent_block corpus
- mixed-type recovery is promising for request-sequence windows
- test extra conditioning features: object_unique,signed_stride_lag1_autocorr

## s3-cache-datasets / 2015_cloudphysics

- Closest learned anchor: tencent_block (distance 1.2)
- Family kind: request_sequence
- Sampling: random-ok
- Regime recipe: K≈8
- Char-file conditioning: yes
- PCF status: validated
- Multi-scale critic status: promising
- Mixed-type recovery status: promising
- Retrieval memory status: mixed
- Candidate conditioning additions: object_unique,signed_stride_lag1_autocorr,obj_size_std

### Why

- family looks multi-regime or high-heterogeneity; burstiness is materially above the calmer families

### Defaults

- use char-file conditioning
- use a regime sampler around K≈8
- keep burst-sensitive temporal objectives on

### Candidates

- PCF loss is validated on the tencent_block corpus
- multi-scale critic looks promising for higher-mode families
- mixed-type recovery is promising for request-sequence windows
- test extra conditioning features: object_unique,signed_stride_lag1_autocorr,obj_size_std

## s3-cache-datasets / 2016_wiki

- Closest learned anchor: tencent_block (distance 2.68)
- Family kind: structured_table
- Sampling: split-by-format-first
- Regime recipe: single
- Char-file conditioning: no
- PCF status: not-primary
- Multi-scale critic status: not-primary
- Mixed-type recovery status: not-primary
- Retrieval memory status: not-primary

### Why

- formats/parsers are mixed

### Defaults

- split formats before training
- current window GAN is a weaker fit than it is for request-sequence families

## s3-cache-datasets / 2017_docker

- Closest learned anchor: tencent_block (distance 2.68)
- Family kind: structured_table
- Sampling: random-ok
- Regime recipe: single
- Char-file conditioning: no
- PCF status: not-primary
- Multi-scale critic status: not-primary
- Mixed-type recovery status: not-primary
- Retrieval memory status: not-primary

### Why

- no single pathological axis dominates this family

### Defaults

- current window GAN is a weaker fit than it is for request-sequence families

## s3-cache-datasets / 2017_systor

- Closest learned anchor: tencent_block (distance 1.208)
- Family kind: request_sequence
- Sampling: random-ok
- Regime recipe: single
- Char-file conditioning: yes
- PCF status: validated
- Multi-scale critic status: promising
- Mixed-type recovery status: promising
- Retrieval memory status: mixed
- Candidate conditioning additions: object_unique,signed_stride_lag1_autocorr,obj_size_std

### Why

- no single pathological axis dominates this family

### Defaults

- use char-file conditioning

### Candidates

- PCF loss is validated on the tencent_block corpus
- mixed-type recovery is promising for request-sequence windows
- test extra conditioning features: object_unique,signed_stride_lag1_autocorr,obj_size_std

## s3-cache-datasets / 2018_tencentPhoto

- Closest learned anchor: tencent_block (distance 2.524)
- Family kind: request_sequence
- Sampling: split-by-format-first
- Regime recipe: single
- Char-file conditioning: yes
- PCF status: validated
- Multi-scale critic status: promising
- Mixed-type recovery status: promising
- Retrieval memory status: mixed
- Candidate conditioning additions: object_unique,signed_stride_lag1_autocorr,obj_size_std

### Why

- burstiness is materially above the calmer families; formats/parsers are mixed

### Defaults

- split formats before training
- use char-file conditioning
- keep burst-sensitive temporal objectives on

### Candidates

- PCF loss is validated on the tencent_block corpus
- mixed-type recovery is promising for request-sequence windows
- test extra conditioning features: object_unique,signed_stride_lag1_autocorr,obj_size_std

## s3-cache-datasets / 2019_wiki

- Closest learned anchor: tencent_block (distance 2.197)
- Family kind: request_sequence
- Sampling: split-by-format-first
- Regime recipe: single
- Char-file conditioning: yes
- PCF status: validated
- Multi-scale critic status: promising
- Mixed-type recovery status: promising
- Retrieval memory status: mixed
- Candidate conditioning additions: object_unique,signed_stride_lag1_autocorr,obj_size_std

### Why

- burstiness is materially above the calmer families; formats/parsers are mixed

### Defaults

- split formats before training
- use char-file conditioning
- keep burst-sensitive temporal objectives on

### Candidates

- PCF loss is validated on the tencent_block corpus
- mixed-type recovery is promising for request-sequence windows
- test extra conditioning features: object_unique,signed_stride_lag1_autocorr,obj_size_std

## s3-cache-datasets / 2020_alibabaBlock

- Closest learned anchor: alibaba (distance 0.539)
- Family kind: request_sequence
- Sampling: split-by-format-first
- Regime recipe: K≈8
- Char-file conditioning: yes
- PCF status: promising
- Multi-scale critic status: promising
- Mixed-type recovery status: mixed
- Retrieval memory status: unknown
- Candidate conditioning additions: object_unique,signed_stride_lag1_autocorr,obj_size_std

### Why

- ordered files show temporal persistence; family looks multi-regime or high-heterogeneity; formats/parsers are mixed

### Defaults

- split formats before training
- use char-file conditioning
- use a regime sampler around K≈8

### Candidates

- PCF loss is promising on the alibaba corpus
- multi-scale critic looks promising for higher-mode families
- test extra conditioning features: object_unique,signed_stride_lag1_autocorr,obj_size_std

## s3-cache-datasets / 2020_tencentBlock

- Closest learned anchor: alibaba (distance 0.553)
- Family kind: request_sequence
- Sampling: split-by-format-first
- Regime recipe: K≈8
- Char-file conditioning: yes
- PCF status: promising
- Multi-scale critic status: promising
- Mixed-type recovery status: mixed
- Retrieval memory status: unknown
- Candidate conditioning additions: object_unique,signed_stride_lag1_autocorr,obj_size_std

### Why

- ordered files show temporal persistence; family looks multi-regime or high-heterogeneity; formats/parsers are mixed

### Defaults

- split formats before training
- use char-file conditioning
- use a regime sampler around K≈8

### Candidates

- PCF loss is promising on the alibaba corpus
- multi-scale critic looks promising for higher-mode families
- test extra conditioning features: object_unique,signed_stride_lag1_autocorr,obj_size_std

## s3-cache-datasets / 2020_twitter

- Closest learned anchor: tencent_block (distance 2.059)
- Family kind: request_sequence
- Sampling: random-ok
- Regime recipe: single
- Char-file conditioning: yes
- PCF status: validated
- Multi-scale critic status: promising
- Mixed-type recovery status: promising
- Retrieval memory status: mixed
- Candidate conditioning additions: object_unique,signed_stride_lag1_autocorr,obj_size_std

### Why

- burstiness is materially above the calmer families

### Defaults

- use char-file conditioning
- keep burst-sensitive temporal objectives on

### Candidates

- PCF loss is validated on the tencent_block corpus
- mixed-type recovery is promising for request-sequence windows
- test extra conditioning features: object_unique,signed_stride_lag1_autocorr,obj_size_std

## s3-cache-datasets / 2020_twr_cdn

- Closest learned anchor: tencent_block (distance 2.573)
- Family kind: structured_table
- Sampling: split-by-format-first
- Regime recipe: single
- Char-file conditioning: no
- PCF status: not-primary
- Multi-scale critic status: not-primary
- Mixed-type recovery status: not-primary
- Retrieval memory status: not-primary

### Why

- formats/parsers are mixed

### Defaults

- split formats before training
- current window GAN is a weaker fit than it is for request-sequence families

## s3-cache-datasets / 2022_metaCDN

- Closest learned anchor: tencent_block (distance 7.353)
- Family kind: request_sequence
- Sampling: random-ok
- Regime recipe: single
- Char-file conditioning: yes
- PCF status: validated
- Multi-scale critic status: promising
- Mixed-type recovery status: promising
- Retrieval memory status: mixed
- Candidate conditioning additions: object_unique,signed_stride_lag1_autocorr,obj_size_std

### Why

- burstiness is materially above the calmer families

### Defaults

- use char-file conditioning
- keep burst-sensitive temporal objectives on

### Candidates

- PCF loss is validated on the tencent_block corpus
- mixed-type recovery is promising for request-sequence windows
- test extra conditioning features: object_unique,signed_stride_lag1_autocorr,obj_size_std

## s3-cache-datasets / 2022_metaKV

- Closest learned anchor: tencent_block (distance 37.034)
- Family kind: request_sequence
- Sampling: split-by-format-first
- Regime recipe: single
- Char-file conditioning: yes
- PCF status: validated
- Multi-scale critic status: promising
- Mixed-type recovery status: promising
- Retrieval memory status: mixed
- Candidate conditioning additions: object_unique,signed_stride_lag1_autocorr,obj_size_std

### Why

- reuse/locality is not negligible; burstiness is materially above the calmer families; formats/parsers are mixed

### Defaults

- split formats before training
- use char-file conditioning
- treat locality as first-class in the loss/conditioning stack
- keep burst-sensitive temporal objectives on

### Candidates

- PCF loss is validated on the tencent_block corpus
- mixed-type recovery is promising for request-sequence windows
- test extra conditioning features: object_unique,signed_stride_lag1_autocorr,obj_size_std

## s3-cache-datasets / 2022_metaStorage

- Closest learned anchor: tencent_block (distance 250.486)
- Family kind: request_sequence
- Sampling: random-ok
- Regime recipe: single
- Char-file conditioning: yes
- PCF status: validated
- Multi-scale critic status: promising
- Mixed-type recovery status: promising
- Retrieval memory status: mixed
- Candidate conditioning additions: object_unique,signed_stride_lag1_autocorr,obj_size_std

### Why

- reuse/locality is not negligible; burstiness is materially above the calmer families

### Defaults

- use char-file conditioning
- treat locality as first-class in the loss/conditioning stack
- keep burst-sensitive temporal objectives on

### Candidates

- PCF loss is validated on the tencent_block corpus
- mixed-type recovery is promising for request-sequence windows
- test extra conditioning features: object_unique,signed_stride_lag1_autocorr,obj_size_std

## s3-cache-datasets / 2023_metaCDN

- Closest learned anchor: tencent_block (distance 2.483)
- Family kind: structured_table
- Sampling: random-ok
- Regime recipe: single
- Char-file conditioning: no
- PCF status: not-primary
- Multi-scale critic status: not-primary
- Mixed-type recovery status: not-primary
- Retrieval memory status: not-primary

### Why

- no single pathological axis dominates this family

### Defaults

- current window GAN is a weaker fit than it is for request-sequence families

## s3-cache-datasets / 2023_metaStorage

- Closest learned anchor: tencent_block (distance 2.68)
- Family kind: structured_table
- Sampling: split-by-format-first
- Regime recipe: single
- Char-file conditioning: no
- PCF status: not-primary
- Multi-scale critic status: not-primary
- Mixed-type recovery status: not-primary
- Retrieval memory status: not-primary

### Why

- formats/parsers are mixed

### Defaults

- split formats before training
- current window GAN is a weaker fit than it is for request-sequence families

## s3-cache-datasets / 2024_google

- Closest learned anchor: tencent_block (distance 2.68)
- Family kind: structured_table
- Sampling: split-by-format-first
- Regime recipe: single
- Char-file conditioning: no
- PCF status: not-primary
- Multi-scale critic status: not-primary
- Mixed-type recovery status: not-primary
- Retrieval memory status: not-primary

### Why

- formats/parsers are mixed

### Defaults

- split formats before training
- current window GAN is a weaker fit than it is for request-sequence families

## s3-cache-datasets / 2026-alibaba-thinkahead

- Closest learned anchor: tencent_block (distance 1.894)
- Family kind: request_sequence
- Sampling: block
- Regime recipe: K≈8
- Char-file conditioning: yes
- PCF status: validated
- Multi-scale critic status: promising
- Mixed-type recovery status: promising
- Retrieval memory status: mixed
- Candidate conditioning additions: object_unique,signed_stride_lag1_autocorr

### Why

- ordered files show temporal persistence; family looks multi-regime or high-heterogeneity; burstiness is materially above the calmer families

### Defaults

- use block or sequential file sampling
- use char-file conditioning
- use a regime sampler around K≈8
- keep burst-sensitive temporal objectives on

### Candidates

- PCF loss is validated on the tencent_block corpus
- multi-scale critic looks promising for higher-mode families
- mixed-type recovery is promising for request-sequence windows
- test extra conditioning features: object_unique,signed_stride_lag1_autocorr

## s3-cache-datasets / alibaba

- Closest learned anchor: tencent_block (distance 0.344)
- Family kind: request_sequence
- Sampling: block
- Regime recipe: K≈8
- Char-file conditioning: yes
- PCF status: validated
- Multi-scale critic status: promising
- Mixed-type recovery status: promising
- Retrieval memory status: mixed
- Candidate conditioning additions: object_unique,signed_stride_lag1_autocorr

### Why

- ordered files show temporal persistence; family looks multi-regime or high-heterogeneity; burstiness is materially above the calmer families

### Defaults

- use block or sequential file sampling
- use char-file conditioning
- use a regime sampler around K≈8
- keep burst-sensitive temporal objectives on

### Candidates

- PCF loss is validated on the tencent_block corpus
- multi-scale critic looks promising for higher-mode families
- mixed-type recovery is promising for request-sequence windows
- test extra conditioning features: object_unique,signed_stride_lag1_autocorr

## s3-cache-datasets / cache_trace_twitter_memcache

- Closest learned anchor: tencent_block (distance 1.582)
- Family kind: request_sequence
- Sampling: split-by-format-first
- Regime recipe: K≈8
- Char-file conditioning: yes
- PCF status: validated
- Multi-scale critic status: promising
- Mixed-type recovery status: promising
- Retrieval memory status: mixed
- Candidate conditioning additions: object_unique,signed_stride_lag1_autocorr,obj_size_std

### Why

- ordered files show temporal persistence; family looks multi-regime or high-heterogeneity; burstiness is materially above the calmer families; formats/parsers are mixed

### Defaults

- split formats before training
- use char-file conditioning
- use a regime sampler around K≈8
- keep burst-sensitive temporal objectives on

### Candidates

- PCF loss is validated on the tencent_block corpus
- multi-scale critic looks promising for higher-mode families
- mixed-type recovery is promising for request-sequence windows
- test extra conditioning features: object_unique,signed_stride_lag1_autocorr,obj_size_std

## s3-cache-datasets / cloudphysics

- Closest learned anchor: tencent_block (distance 0.656)
- Family kind: request_sequence
- Sampling: split-by-format-first
- Regime recipe: single
- Char-file conditioning: yes
- PCF status: validated
- Multi-scale critic status: promising
- Mixed-type recovery status: promising
- Retrieval memory status: mixed
- Candidate conditioning additions: object_unique,signed_stride_lag1_autocorr

### Why

- ordered files show temporal persistence; burstiness is materially above the calmer families; formats/parsers are mixed

### Defaults

- split formats before training
- use char-file conditioning
- keep burst-sensitive temporal objectives on

### Candidates

- PCF loss is validated on the tencent_block corpus
- mixed-type recovery is promising for request-sequence windows
- test extra conditioning features: object_unique,signed_stride_lag1_autocorr

## s3-cache-datasets / metaKV

- Closest learned anchor: tencent_block (distance 31.234)
- Family kind: request_sequence
- Sampling: random-ok
- Regime recipe: single
- Char-file conditioning: yes
- PCF status: validated
- Multi-scale critic status: promising
- Mixed-type recovery status: promising
- Retrieval memory status: mixed
- Candidate conditioning additions: signed_stride_lag1_autocorr,obj_size_std

### Why

- reuse/locality is not negligible; burstiness is materially above the calmer families

### Defaults

- use char-file conditioning
- treat locality as first-class in the loss/conditioning stack
- keep burst-sensitive temporal objectives on

### Candidates

- PCF loss is validated on the tencent_block corpus
- mixed-type recovery is promising for request-sequence windows
- test extra conditioning features: signed_stride_lag1_autocorr,obj_size_std

## s3-cache-datasets / msr

- Closest learned anchor: tencent_block (distance 0.827)
- Family kind: request_sequence
- Sampling: random-ok
- Regime recipe: K≈8
- Char-file conditioning: yes
- PCF status: validated
- Multi-scale critic status: promising
- Mixed-type recovery status: promising
- Retrieval memory status: mixed
- Candidate conditioning additions: object_unique,signed_stride_lag1_autocorr

### Why

- family looks multi-regime or high-heterogeneity; burstiness is materially above the calmer families

### Defaults

- use char-file conditioning
- use a regime sampler around K≈8
- keep burst-sensitive temporal objectives on

### Candidates

- PCF loss is validated on the tencent_block corpus
- multi-scale critic looks promising for higher-mode families
- mixed-type recovery is promising for request-sequence windows
- test extra conditioning features: object_unique,signed_stride_lag1_autocorr

## s3-cache-datasets / other

- Closest learned anchor: tencent_block (distance 4.115)
- Family kind: request_sequence
- Sampling: random-ok
- Regime recipe: single
- Char-file conditioning: yes
- PCF status: validated
- Multi-scale critic status: promising
- Mixed-type recovery status: promising
- Retrieval memory status: mixed
- Candidate conditioning additions: object_unique,signed_stride_lag1_autocorr,obj_size_std

### Why

- burstiness is materially above the calmer families

### Defaults

- use char-file conditioning
- keep burst-sensitive temporal objectives on

### Candidates

- PCF loss is validated on the tencent_block corpus
- mixed-type recovery is promising for request-sequence windows
- test extra conditioning features: object_unique,signed_stride_lag1_autocorr,obj_size_std

## s3-cache-datasets / tencentBlock

- Closest learned anchor: tencent_block (distance 0)
- Family kind: request_sequence
- Sampling: block
- Regime recipe: K≈8
- Char-file conditioning: yes
- PCF status: validated
- Multi-scale critic status: promising
- Mixed-type recovery status: promising
- Retrieval memory status: mixed
- Candidate conditioning additions: object_unique,signed_stride_lag1_autocorr

### Why

- ordered files show temporal persistence; family looks multi-regime or high-heterogeneity

### Defaults

- use block or sequential file sampling
- use char-file conditioning
- use a regime sampler around K≈8

### Candidates

- PCF loss is validated on the tencent_block corpus
- multi-scale critic looks promising for higher-mode families
- mixed-type recovery is promising for request-sequence windows
- test extra conditioning features: object_unique,signed_stride_lag1_autocorr

## tencent / cloud_disk

- Closest learned anchor: alibaba (distance 4.56)
- Family kind: aggregate_time_series
- Sampling: random-ok
- Regime recipe: K≈8
- Char-file conditioning: yes
- PCF status: promising
- Multi-scale critic status: promising
- Mixed-type recovery status: not-primary
- Retrieval memory status: not-primary

### Why

- family looks multi-regime or high-heterogeneity

### Defaults

- use char-file conditioning
- use a regime sampler around K≈8

### Candidates

- PCF loss is promising on the alibaba corpus
- multi-scale critic looks promising for higher-mode families

