# Peer-Review Ledger

Findings that don't fit cleanly under one of the per-vendor peer-review files
(`PEER-REVIEW-LANL.md`, `PEER-REVIEW-LLNL.md`, `PEER-REVIEW-Sandia.md`,
`PEER-REVIEW-GEMINI.md`, `PEER-REVIEW-Copilot.md`). New cross-cutting tooling,
infrastructure controls, and methodology checks live here.

---

## Cache Simulator

### Round 1 (2026-04-30) — `tools/cachesim` Stood Up As A Synthetic-Vs-Real Yardstick

#### Finding

`tools/cachesim/` (Rust crate) was lifted from a pre-staged skeleton to a
working six-policy simulator that loads a trace once and runs every
(policy × cache_size) pair in parallel via rayon. On vinge (20 cores) a 13.9M-
access oracleGeneral `.zst` clears 30 simulations in ~3 s wall, ~33 s CPU.

Policies, all on a shared O(1) doubly-linked arena (`policy::util::DList`),
each line-by-line transcribed against its primary reference:

| Policy | Reference                                  | Notes                                |
| ------ | ------------------------------------------ | ------------------------------------ |
| FIFO   | folklore                                   | Insert-time order, no hit reordering |
| LRU    | Belady-era textbook                        | Recency only                         |
| SLRU   | Karedla–Love–Wherry, IEEE Computer 1994    | 80/20 default, `with_split` exposed  |
| ARC    | Megiddo–Modha, FAST 2003                   | Cases I–IV + REPLACE with `key_in_b2 ∧ |T1|=p` tie-break |
| CAR    | Bansal–Modha, FAST 2004                    | ARC over CLOCK lists; lazy promotion |
| SIEVE  | Yang–Zhang–Yue–Vigfusson, NSDI 2024        | One-bit FIFO with reverse hand       |

Validation: 40 tests (30 unit + 10 integration). Cross-policy property tests
in `tests/policies_invariants.rs` lock down: cold-only trace → 0 hits;
working-set ≤ cap → eventual all-hits after warmup; cap ≥ universe →
only-first-touch misses; capacity invariant under mixed workload; Mattson
1970 trace (`1,2,3,4,1,2,5,1,2,3,4,5`, cap=3) pinned to LRU=2 hits, FIFO=3
hits.

Trace plumbing: oracleGeneral `.zst` reader and CSV reader. CSV auto-detects
`stream_id` / `stream` / `tenant`. Loader time-sorts the in-memory access
vector by `ts` (so block-concatenated multi-stream synthetic CSVs interleave
naturally) and assigns each `(stream_id, obj_id)` a fresh sequential cache
key (so streams with overlapping `obj_id` namespaces stay disjoint).

#### Required Control

Any new policy added to the simulator must:

1. Use `policy::util::DList<P>` for any list-of-cache-keys structure, so all
   `push`/`pop`/`remove`/`move_to_front` operations stay O(1).
2. Land with #[cfg(test)] property tests asserting the algorithm's invariants
   (capacity bounds, paper-cited adaptivity rules) — *not* hand-traced
   sequence tests that break on benign refactors.
3. Pass the cross-policy properties in `tests/policies_invariants.rs`.
4. Be added to `PolicyKind`, `make()`, and the CLI `--policy` value-enum so
   it shows up in the parallel sweep alongside the others.

The simulator is a *yardstick*, not a goal — its only purpose is to measure
how closely a synthetic trace reproduces a real trace's cacheing behaviour.

---

### Round 2 (2026-04-30) — Multi-Stream Synthetic CSVs Aliased Cache Keys

#### Finding

The synthetic `.csv` files emitted by `llgan/generate.py` and
`altgan/generate.py` carry a `stream_id` column and (per stream) re-use the
same `obj_id` namespace 0..k. The first cachesim CSV reader read `obj_id`
directly into the cache key, so every stream's `obj_id 7` collided onto the
same cache slot. Plus, the synthetic CSVs are written as 8 (or 4) contiguous
6 250-row blocks rather than time-interleaved — the cache saw stream 0's
entire access pattern before stream 1 began, which made reuse distance
artificially huge.

Symptom on `alibaba_v195_ep110_bernoulli265_8x50k.csv`: 4 680 unique
`obj_id`s reported, miss-ratios 0.93–0.42 across cap=64..4096, ARC vs CAR
diverging by 26 percentage points at cap=4 096 (which does not happen for
correct ARC/CAR implementations).

#### Required Control

Loader (`main.rs`) must:

1. Time-sort all in-memory accesses by `ts` so multi-stream blocks interleave.
2. Build cache keys via `(stream_id, obj_id) → next_id` in a `HashMap`, so
   each stream's namespace is disjoint.

After fix the same alibaba CSV reports 36 629 unique cache keys (8×) and
the ARC-vs-CAR divergence at cap=4 096 collapses to 0.003 — within the
expected paper bound. Any new trace reader must populate `Access.stream_id`
honestly; oracleGeneral leaves it 0 (single-stream).

---

### Round 3 (2026-04-30) — Alibaba `bernoulli265_8x50k` Synth Is Not Cache-Faithful

#### Finding

Real alibaba `alibabaBlock_0` (13.9M accesses, 4.74M unique) vs synth
`alibaba_v195_ep110_bernoulli265_8x50k.csv` (50k accesses, 36.6k unique
after disambiguation), at matched cache-as-fraction-of-universe:

```
frac %     ARC real / synth   CAR              FIFO             LRU              SIEVE            SLRU
0.02 / 0.02   0.4573 / 0.9876   0.4573 / 0.9881   0.4962 / 0.9869   0.4607 / 0.9868   0.5153 / 0.9978   0.4597 / 0.9995
0.09 / 0.09   0.4548 / 0.9653   0.4547 / 0.9651   0.4653 / 0.9647   0.4555 / 0.9646   0.4701 / 0.9928   0.4562 / 0.9894
0.35 / 0.35   0.4576 / 0.9143   0.4521 / 0.9143   0.4569 / 0.9137   0.4544 / 0.9136   0.4569 / 0.9777   0.4543 / 0.9610
1.40 / 1.38   0.4039 / 0.8421   0.3978 / 0.8396   0.4317 / 0.8347   0.4267 / 0.8332   0.4100 / 0.9327   0.4537 / 0.9101
5.59 / 5.53   0.3655 / 0.7962   0.3651 / 0.7992   0.3958 / 0.7838   0.3783 / 0.7828   0.3650 / 0.8537   0.3733 / 0.8224
```

Mean |Δ| (HRC-MAE) for ARC ≈ **0.473**; same order of magnitude for every
policy. Two diagnostics:

- **Shape error.** Real alibaba's HRC is nearly flat at ~0.45 across the
  small-cap range, then drops sharply. The synth declines smoothly with cap,
  closer to a uniform-random access pattern than to the heavy reuse-tail the
  real trace exhibits.
- **Policy-ranking flip.** On real, SIEVE is competitive with ARC. On synth,
  SIEVE is the *worst* policy. SIEVE's lazy-promotion bit is the most
  sensitive read of "second-touch is reliable"; the synth doesn't deliver
  that signal.

#### Required Control

`bernoulli265_8x50k` is not the alibaba synth that should be cited as
cache-faithful. Any future alibaba object-process promotion must clear an
HRC-MAE gate computed by cachesim against `alibabaBlock_0` (or another
agreed real reference) at five matched cap fractions; LANL's published gate
target is **HRC-MAE(real, fake) ≤ 0.009** for ARC.

---

### Round 4 (2026-04-30) — Tencent `bernoulli615_4x100k` ~5× Closer Than Alibaba, Still Off LANL Gate

#### Finding

Comparison repeated for tencent: real `tencentBlock_10004` (1.04M accesses,
127k unique) vs synth `tencent_v165_ep45_bernoulli615_4x100k.csv` (100k
accesses, 38.6k unique). At five matched cap fractions:

```
frac %       ARC                CAR                FIFO               LRU                SIEVE              SLRU
 0.50/ 0.50  0.5697/0.7884 +.22  0.5688/0.7888 +.22  0.6172/0.7885 +.17  0.5967/0.7874 +.19  0.5850/0.8559 +.27  0.6084/0.8247 +.22
 2.00/ 2.00  0.5310/0.5773 +.05  0.5297/0.5901 +.06  0.5709/0.5846 +.01  0.5520/0.5767 +.02  0.5339/0.7069 +.17  0.5483/0.6970 +.15
 5.00/ 5.00  0.5248/0.4874 -.04  0.5239/0.4907 -.03  0.5487/0.4907 -.06  0.5325/0.4836 -.05  0.5249/0.5761 +.05  0.5284/0.5708 +.04
15.00/15.00  0.4958/0.4215 -.07  0.4959/0.4281 -.07  0.5316/0.4222 -.11  0.5229/0.4200 -.10  0.5226/0.4571 -.07  0.5230/0.4289 -.09
50.00/50.00  0.2500/0.3975 +.15  0.2917/0.3989 +.11  0.3063/0.3861 +.08  0.2484/0.3859 +.14  0.2957/0.4024 +.11  0.4998/0.3934 -.11
HRC-MAE             0.1048              0.0977              0.0864              0.1009              0.1335              0.1215
```

ARC HRC-MAE = **0.105** — about 5× closer to real than the alibaba synth
above, but still ~12× the README's LANL Tencent gate of 0.009. The Δ-sign
flip at ~5% cap fraction reveals the same shape error: synth is too
pessimistic at tiny caches and too optimistic mid-range.

Note: the real side here was a 1.04M-record block, the synth 100k records;
the length asymmetry was suspicious so a length-matched comparison was run
(Round 5).

#### Required Control

`bernoulli615_4x100k` is the closest tencent synth tested but does not yet
clear the published LANL gate. Before any "synth ≈ real" claim, two more
controls are needed:

1. Length-match the real to the synth (Round 5 below).
2. Compare against the *exact* real slice the synth was tuned against
   (LANL setup file `checkpoints/tencent_v165/long_rollout_lanl_setup_real.json`
   pins n_records=100000, n_streams=4, seed=42 against `tencent_block_1M`).

---

### Round 5 (2026-04-30) — Length-Matched Tencent Comparison Closes The Per-Trace Asymmetry, Confirms The Shape Error

#### Finding

Clipped real to match synth length: `~/traces/tencent_block_1M_clipped/tencentBlock_10004_100k.oracleGeneral.zst`
(525 KB compressed, first 100k records of tencentBlock_10004; original
file untouched). Both traces 100k accesses; real 48 272 unique, synth
38 594 unique.

```
  cap         ARC                 CAR                 FIFO                LRU                 SIEVE               SLRU
       REAL   SYNTH   Δ     REAL   SYNTH   Δ     REAL   SYNTH   Δ     REAL   SYNTH   Δ     REAL   SYNTH   Δ     REAL   SYNTH   Δ
   32  0.6586 0.8993 +.241  0.6657 0.8985 +.233  0.8092 0.8972 +.088  0.8070 0.8967 +.090  0.8857 0.9593 +.074  0.8474 0.9544 +.107
  128  0.6157 0.8251 +.209  0.6153 0.8249 +.210  0.6774 0.8246 +.147  0.6229 0.8240 +.201  0.6375 0.8902 +.253  0.7419 0.8562 +.114
  512  0.5798 0.6562 +.076  0.5778 0.6628 +.085  0.6182 0.6612 +.043  0.5969 0.6561 +.059  0.5988 0.7618 +.163  0.6125 0.7416 +.129
 2048  0.5477 0.4855 -.062  0.5476 0.4880 -.060  0.5803 0.4874 -.093  0.5648 0.4813 -.083  0.5567 0.5724 +.016  0.5831 0.5620 -.021
 8192  0.5398 0.4089 -.131  0.5398 0.4155 -.124  0.5534 0.3963 -.157  0.5427 0.3881 -.155  0.5404 0.4329 -.108  0.5449 0.4218 -.123
HRC-MAE       0.1439              0.1423              0.1057              0.1176              0.1225              0.0989
```

HRC-MAE moves *up* on the recency-aware policies (ARC 0.105 → 0.144,
CAR 0.098 → 0.142). The earlier 1.04M-record real was getting credit from
extra long-tail reuse the synth doesn't generate; clipping kills the unfair
length advantage and the gap widens. The cross-over near cap ≈ 1024 is
preserved — synth structurally over-estimates miss ratio at small caches
and under-estimates at large caches.

FIFO and SLRU HRC-MAE are now the lowest of the six (~0.10), because both
real and synth flatten out at clipped length and the policies that don't
exploit reuse depth see the smallest gap.

#### Required Control

For all future synthetic-vs-real claims:

1. **Length-match real to synth** before computing HRC-MAE. The current
   `cachesim` CLI does not do this automatically — clip the `.zst` by
   piping through `zstd -dc | head -c N×24 | zstd -19 -o …`. A
   `--clip-to-length` flag is the obvious next CLI improvement; until it
   lands, document the clip in the run notes.
2. **Match the LANL evaluation setup** (n_records, n_streams, seed) before
   comparing against the published 0.009 gate. The setup is in
   `checkpoints/tencent_v165/long_rollout_lanl_setup_real.json`.
3. **Report HRC-MAE per policy.** Single-policy ARC HRC-MAE is the headline,
   but the per-policy spread (here 0.099–0.144) reveals which kind of
   structure the synth gets wrong: a low-FIFO / high-SIEVE gap means the
   synth's second-touch distribution is mismatched, not just the cold-miss
   rate.

The pending Mattson single-pass HRC + paired-trace HRC-MAE feature
(`tools/cachesim` README §"Validation gates") will collapse this whole
write-up into a single JSON sidecar field per (policy, real, synth) triple
once it lands.
