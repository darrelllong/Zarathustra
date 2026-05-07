# Zarathustra

> *"I am Zarathustra the godless — I cook every chance in my pot. And only when it has been thoroughly cooked do I welcome it as my food."*
> — Gathas, Yasna 43 (attrib. Zarathustra / Zoroaster, c. 1500–1000 BCE)

> *"Iron sharpeneth iron; so a man sharpeneth the countenance of his friend."*
> — Proverbs 27:17

The project takes its name from the ancient Persian prophet Zarathustra — Zoroaster in Greek — who articulated the first great dualist cosmology: Ahura Mazda, the Wise Lord of light and truth, in eternal struggle against Angra Mainyu, the destructive spirit of darkness and deception. From that primal opposition all creation emerges and all meaning is made.

Here the duality has moved up the stack. Two adversarial teams — **LLNL** and **LANL** — each propose synthetic-trace generators; an unblinking **cache simulator** refuses to be fooled by either. What survives is whatever can actually reproduce a real workload's behaviour on real cache eviction policies. Iron sharpens iron.

Zarathustra generates synthetic I/O workload traces that are statistically indistinguishable from production traces under cache-fidelity tests. The headline metric is **mean HRC-MAE on a six-policy cache-simulation surface** — closeness, in eviction-policy hit-ratio space, to a real reference trace. Not embedding distance. Not held-out token likelihood. Cache behaviour.

## Motivation

Storage and CDN researchers need realistic workloads for benchmarking, capacity planning, and cache-policy comparison. Real production traces are proprietary, enormous, and tied to a single point in time. A generator trained on a broad corpus can produce arbitrary volumes of synthetic workload that reflects the statistical character of real systems — burstiness, reuse structure, size distributions, working-set evolution — without exposing sensitive data.

The original lineage:

> H. Zhang, Z. Yang, Y. Xie, Y. Wu, J. Li, D. Feng, A. Wildani, and D. Long,
> *"Accurate Generation of I/O Workloads Using Generative Adversarial Networks,"*
> in *Proceedings of the 17th IEEE International Conference on Networking, Architecture, and Storage (NAS 2024)*, Guangzhou, China, Nov. 2024. ([PDF](pubs/Accurate_Generation_of_I_O_Workloads_Using_Generative_Adversarial Networks.pdf))

The NAS '24 paper showed an LSTM-WGAN (LLGAN) could match a single trace's marginal distributions better than classical models. Zarathustra is what came after: a corpus-level generator under an adversarial peer-review regime, evaluated on cache simulation rather than embedding-space fidelity.

## The race

Two repos share this tree. They are written and operated by adversarial agents:

- **`llgan/`** — LLNL (Claude Opus). The original LSTM-WGAN evolved into a multi-track suite: phase-conditioned PMF atlas, neural conditional-transition atlas, IRD-renewal, post-hoc LRU stack decoders, chunk-ensemble optimizer, and trace-bootstrap baseline.
- **`altgan/`** — LANL (ChatGPT). An explicit cache-locality stack model — `StackAtlasModel` / `NeuralAtlasModel` / `NeuralMarkRuntime` — that treats stack distance, action class, and reuse rank as first-class generation variables.

Each turn, one team commits a **measured multi-seed claim**, a **closes-NEGATIVE** that prunes the next-move list, or an **infrastructure fix**. Single-seed numbers are scouting probes, not claims. A claim requires four seeds (e.g. 42/80/81/82), with mean and range reported. The ground truth is a Rust six-policy cache simulator at `tools/cachesim/`. See [LEADER-BOARD.md](LEADER-BOARD.md) for current standings and [LLNL-OPERATING-CHARTER.md](LLNL-OPERATING-CHARTER.md) / [LANL-OPERATING-CHARTER.md](LANL-OPERATING-CHARTER.md) for the operating contracts.

## Generation tracks

```
                                ┌─────────────────────────────┐
                                │  real .zst / .csv traces    │
                                └──────────────┬──────────────┘
                                               │ fit
              ┌────────────────┬───────────────┼────────────────┬────────────────┐
              ▼                ▼               ▼                ▼                ▼
       ┌──────────┐   ┌────────────────┐  ┌──────────┐  ┌──────────────┐  ┌──────────┐
       │ Atlas    │   │ Neural Atlas   │  │  IRD     │  │ GAN (legacy) │  │ Trace    │
       │ Phase /  │   │ cond → state   │  │ renewal  │  │ LSTM-WGAN-SN │  │ bootstrap│
       │ Markov / │   │ MLP +          │  │ empirical│  │ + auxiliary  │  │ chunk    │
       │ Stack    │   │ rank PMF       │  │ IRD/IRM  │  │ losses       │  │ shuffle  │
       └────┬─────┘   └────────┬───────┘  └─────┬────┘  └──────┬───────┘  └────┬─────┘
            │                  │                │             │                │
            └─────── per-stream rollout ────────┴──── post-hoc shaping ────────┘
                              (LRU stack decoder, chunk ensemble,
                               hot-pool / tail-reuse / recent-pool knobs)
                                              │
                                              ▼
                                     synthetic .csv trace
                                              │
                       ┌──────────────────────┴──────────────────────┐
                       ▼                                             ▼
              ┌────────────────┐                          ┌────────────────────┐
              │ tools/cachesim │  six policies: LRU,     │  per-policy / per- │
              │  (Rust crate)  │  ARC, FIFO, SIEVE,      │   capacity HRC-MAE │
              │                │  SLRU, CAR              │   vs real          │
              └────────────────┘  (+ LFU, LIRS for       └────────────────────┘
                                   8-policy CloudPhysics)
```

- **Phase / Markov / Stack atlas** (`llgan/phase_pmf_atlas.py`, `llgan/markov_atlas.py`, `llgan/stack_atlas.py`, `altgan/model.py:StackAtlasModel`). Reservoir sample a per-bucket PMF over (time/size/phase/action) state; roll out by sampling from that PMF. Bit-counted; no learning.
- **Neural conditional-transition atlas** (`llgan/neural_atlas.py`, `altgan/neural_atlas.py`). Six-state space (1 NEW + 5 stack-distance buckets) with a small MLP `P(next_state | prev_state, cond)`; rank PMF per state. Inline-cond beats JSONL-cond by 4× on alibaba. LANL's PhaseAtlas adds a forced-phase schedule and reservoir-driven event marks (opcode/tenant/dt/size).
- **IRD-renewal** (`llgan/ird_renewal.py`, `altgan/ird_renewal.py`). Empirical IRD distribution from real, optionally rank-conditioned; generate by sampling IRD and reading the LRU stack at that depth. Strong on KV-style corpora (CloudPhysics, Wikipedia, Meta KV) where neural-atlas scale-sharpening over-concentrates rank=0.
- **GAN (legacy LLGAN)** (`llgan/train.py`, `llgan/model.py`). Four-phase curriculum: AE pretrain → supervisor pretrain → G warm-up → joint WGAN-SN. Auxiliary losses (multi-scale critic, PCF, retrieval-memory, chunk-stitching, reuse-rate hinge). Closed-bounded on the cachesim surface; retained for ablation and per-window distributional metrics.
- **Trace-bootstrap** (`llgan/trace_bootstrap.py`, `altgan/trace_bootstrap.py`). Methodology baseline: chunk-shuffle of the real trace at chunk_size ≫ max cache. Sits at 0.0000-class HRC-MAE because the chunk boundaries rarely intersect cache-relevant scales. Useful for leaderboard occupation; not a generative claim.
- **Chunk-ensemble post-hoc** (`llgan/chunk_ensemble.py`, `altgan/launch_chunk_surface_multiseed.py`). Greedy guard pass: for each chunk of a base fake, try swapping the obj-id (and optionally size) column with each donor; keep the swap that reduces cachesim mean HRC-MAE against real. Pure synthesis; does not see real ids.

## Evaluation

The headline metric is `llgan.cachesim_eval`:

```bash
python3 -m llgan.cachesim_eval \
    --fake YOUR_FAKE.csv \
    --real /tiamat/zarathustra/llgan-output/refs/<corpus>_real.csv \
    --cache-sizes 32,128,512,2048,8192 \
    --policies lru,arc,fifo,sieve,slru,car
```

This calls `tools/cachesim/target/release/cachesim` (Rust, 30 simulations / ~3 s wall on vinge) and aggregates the per-(policy, capacity) miss-ratio MAE against a real reference. CloudPhysics also runs an 8-policy surface (`+ lfu,lirs`, plus a `32768` cache size). Match the corpus's official surface; do not shop for the easiest one.

**Multi-seed protocol.** A claim requires 4 seeds, mean and range reported. Single-seed wins are not promotable. See `RESPONSE-LLNL.md` / `RESPONSE-LANL.md` for the per-claim posting format (literal `cachesim mean HRC-MAE across policies` lines plus exact JSON means).

**Frozen GAN-bundle protocol** (legacy). `llgan/eval.py --eval-real-seed 42` returns MMD², β-recall, PRDC, DMD-GEN, Context-FID, autocorrelation distance, and a per-window HRC-MAE at hand-picked cache fractions of footprint. Useful as a sanity panel for short-window distributional realism; superseded by `cachesim_eval` for race claims.

**2DIO comparison class.** A third-party EuroSys '26 entrant (2DIO) reports per-trace HRC-MAE 0.02–0.05 on alibabaBlock_521 and CloudPhysics traces by directly parameterising the IRD distribution. Per-trace memoization is a different metric class than corpus generalization, but it sets the bar: see `LEADER-BOARD.md § 2DIO comparison class`.

## Downloading the traces

The training data is the public **libCacheSim cache-dataset** corpus (Juncheng Yang et al.), hosted in a public AWS S3 bucket. No credentials required.

| Dataset | Period | Format files | Size | Character |
|---------|--------|-------------|------|-----------|
| Tencent Block 2020 | 9 days | 382 oracleGeneral | 152 GB | 512B–32KB blocks, 83–94% reads |
| Alibaba Block 2020 | 31 days | 242 oracleGeneral | 93 GB | 4KB-aligned, mixed read/write |
| Tencent Photo 2018 | — | 2 oracleGeneral | 52 GB | Object store, large objects |
| Twitter 2020 | — | 54 oracleGeneral | 142 GB | CDN/KV cache, high fan-out |
| Meta KV 2022 | — | 3 oracleGeneral | 29 GB | Key-value cache, TTL-heavy |
| Meta CDN 2022 | — | 3 oracleGeneral | 2.1 GB | CDN edge cache |
| CloudPhysics 2015 | — | 106 oracleGeneral | 8.6 GB | VM disk I/O, varied |
| MSR Cambridge 2007 | — | 14 oracleGeneral | 1.5 GB | Enterprise disk traces |
| Baleen / Wikipedia | — | various | — | Storage cache / web replay |
| Tencent Block LCS | 9 days | 4,482 lcs | 662 GB | Same corpus, native LCS format |
| Alibaba Block LCS | 31 days | 428 lcs | 292 GB | Same corpus, native LCS format |

Quick start:

```bash
# Tencent Block 2020 — oracleGeneral format (~152 GB)
aws s3 sync s3://cache-datasets/cache_dataset_oracleGeneral/2020_tencentBlock/ \
    ./traces/2020_tencentBlock/ --no-sign-request

# Or: just one size bucket (~10–20 GB)
aws s3 sync s3://cache-datasets/cache_dataset_oracleGeneral/2020_tencentBlock/1M/ \
    ./traces/2020_tencentBlock/ --no-sign-request
```

S3 prefixes:

| S3 prefix | Format flag | Notes |
|-----------|-------------|-------|
| `s3://cache-datasets/cache_dataset_oracleGeneral/` | `oracle_general` | 24-byte binary; fastest |
| `s3://cache-datasets/cache_dataset_lcs/` | `lcs` | Versioned binary with op/tenant; 4× more files |
| `s3://cache-datasets/cache_dataset_txt/` | `csv` | Plain-text; largest |

**oracleGeneral** (24 bytes/record): `uint32 ts | uint64 obj_id | uint32 size | int32 next_access_vtime (drop) | int16 op | int16 tenant`.

**lcs v2** (28 bytes/record): adds a packed `op|tenant` field; v1 (24 bytes) omits it. Header at offset 0 has magic `0x123456789abcdef0` and a stats block.

Both formats are parsed natively by `llgan/dataset.py` and `altgan/model.py`. For other analysis the [libCacheSim](https://github.com/1a1a11a/libCacheSim) C library and Python bindings provide a full reader.

If you use these traces, please cite:

> Juncheng Yang, Yao Yue, K.V. Rashmi. *"A large-scale analysis of hundreds of in-memory key-value cache clusters at Twitter."* ACM TOS 2021.
>
> Juncheng Yang, Ziyue Qiu, Yazhuo Zhang, Yao Yue, K.V. Rashmi. *"FIFO queues are all you need for cache eviction."* SOSP 2023.

## Quick start

### 1. Build the cache simulator

```bash
cd tools/cachesim
cargo build --release
# binary at tools/cachesim/target/release/cachesim
```

### 2. Score a fake against a real reference

```bash
python3 -m llgan.cachesim_eval \
    --fake my_fake.csv \
    --real /tiamat/zarathustra/llgan-output/refs/alibaba_real.csv \
    --cache-sizes 32,128,512,2048,8192 \
    --policies lru,arc,fifo,sieve,slru,car
```

### 3. Fit and generate from a neural atlas (LLNL track)

```bash
# Fit
python -m llgan.neural_atlas fit \
    --trace-dir /tiamat/zarathustra/traces/alibaba/ \
    --char-file traces/characterization/trace_characterizations.jsonl \
    --files-per-fit 237 --records-per-file 50000 \
    --hidden 96 --epochs 600 --seed 137 \
    --inline-cond --cond-noise-std 0.05 \
    --output atlases/llnl_neural_atlas_alibaba.pkl.gz

# Generate
python -m llgan.neural_atlas generate \
    --model atlases/llnl_neural_atlas_alibaba.pkl.gz \
    --manifest /tiamat/zarathustra/llgan-output/manifests/alibaba_real_seed42_1M.json \
    --hot-pool-prob 0.45 --hot-pool-k 75 --hot-pool-min-age 8 \
    --tail-reuse-prob 0.10 --tail-reuse-min-frac 0.5 \
    --recent-pool-prob 0.15 --recent-pool-window 16 \
    --max-stack-depth 524288 --seed 42 \
    --output fakes/alibaba_v276_seed42.csv
```

### 4. Fit and generate from a PhaseAtlas (LANL track)

```bash
ssh vinge.local 'cd ~/Zarathustra && python3 -m altgan.train_neural_atlas \
    --trace-dir /home/darrell/traces/2020_tencentBlock/ \
    --n-phase-bins 8 --n-time-bins 4 --n-size-bins 4 \
    --hidden 128 --epochs 600 \
    --output /tiamat/zarathustra/checkpoints/altgan/tencent_phaseatlas.pkl.gz'

ssh vinge.local 'cd ~/Zarathustra && python3 -m altgan.evaluate_neural_atlas \
    --model /tiamat/zarathustra/checkpoints/altgan/tencent_phaseatlas.pkl.gz \
    --condition-from-real-manifest \
    --transition-blend 0.575 --local-prob-power 0.70 \
    --force-phase-schedule \
    --stack-rank-phase-scales 1.0,1.0,1.1,1.1 \
    --stack-hot-pool-prob 0.40 --stack-hot-pool-k 100 --stack-hot-pool-window 10000 \
    --fake-output fakes/tencent_lanl.csv --real-output reals/tencent_real.csv \
    --cachesim-bin tools/cachesim/target/release/cachesim'
```

### 5. IRD-renewal (corpus-agnostic, KV-friendly)

```bash
python3 -m llgan.ird_renewal \
    --real-csv /tiamat/zarathustra/llgan-output/refs/wiki_real.csv \
    --n-records 1000000 \
    --independent-prob 0.10 --ird-scale 32 \
    --seed 42 --output fakes/wiki_irdrenewal_seed42.csv
```

### 6. Multi-seed claim

A promotable claim runs four seeds and reports the mean and range:

```bash
for seed in 42 80 81 82; do
    python -m llgan.neural_atlas generate ... --seed $seed --output fake_$seed.csv
    python -m llgan.cachesim_eval --fake fake_$seed.csv --real REAL --output result_$seed.json
done
# Then post the four cachesim mean lines + JSON means to RESPONSE-LLNL.md
```

## Repository layout

```
Zarathustra/
├── llgan/                         # LLNL track
│   ├── train.py / model.py / config.py / dataset.py   # Legacy GAN curriculum
│   ├── neural_atlas.py            # 6-state cond-MLP atlas (R170+)
│   ├── phase_pmf_atlas.py         # Phase-conditioned PMF atlas
│   ├── markov_atlas.py            # Bit-counted compound-state Markov (R155)
│   ├── stack_atlas.py             # Per-object Markov state generator
│   ├── ird_renewal.py             # Empirical IRD + IRM renewal generator
│   ├── chunk_ensemble.py          # Cache-surface chunk-swap optimizer
│   ├── trace_bootstrap.py         # Chunk-shuffle baseline (methodology)
│   ├── lru_stack_decoder.py       # Post-hoc LRU-rank rewrite of obj_id seq
│   ├── cachesim_eval.py           # Headline metric: invokes Rust simulator
│   ├── cachesim_3way.py           # LLNL-vs-LANL-vs-real per-policy table
│   ├── eval.py / frozen_sweep.py  # Legacy frozen-bundle ★ panel
│   ├── long_rollout_eval.py       # 100k-record HRC + reuse + stack panel
│   ├── ird_diag.py / diag_hrc.py  # Per-stream IRD + per-cache-size diagnostics
│   ├── burst_probe.py             # IDEA #67 burst-injection sweep
│   ├── retrieval_memory.py / chunk_stitching.py / pcf_loss.py / ...   # GAN aux
│   └── generate.py                # GAN inference: checkpoint → synthetic CSV
├── altgan/                        # LANL track
│   ├── model.py                   # StackAtlasModel — canonical event frame
│   ├── neural_stack.py            # Older conditioned marginal stack model
│   ├── neural_atlas.py            # NeuralAtlasModel.generate (current champion)
│   ├── neural_marks.py            # Mark sidecar (opcode/tenant/dt/size)
│   ├── ird_renewal.py             # CP / Wikipedia / Meta-KV breakthrough path
│   ├── train_neural_atlas.py      # PhaseAtlas trainer
│   ├── evaluate_neural_atlas.py   # Long-rollout panel + cachesim hand-off
│   ├── cachesim_compare.py        # Standalone peer-trace scorer
│   ├── trace_bootstrap.py         # Chunk-shuffle baseline (LANL impl)
│   ├── launch_*_multiseed.py      # 4-seed claim launchers per corpus
│   ├── sweep_*.py                 # Knob brackets (chunk-ensemble, hot-pool, …)
│   └── RESULTS.md                 # LANL detailed result ledger
├── tools/cachesim/                # Rust 6-policy cache simulator
│   ├── src/                       # FIFO, LRU, SLRU, ARC, CAR, SIEVE
│   ├── tests/                     # 40 tests inc. paper-cited invariants
│   └── README.md
├── traces/characterization/       # Per-file workload descriptors
├── characterizations/             # Aggregate corpus-level characterizations
├── R-scripts/                     # R analyses (Hurst, changepoints, regimes)
├── parsers/                       # Trace-format parsers (SNIA, ETW, …)
├── scripts/                       # Operational + ad-hoc shell scripts
├── paper/ / pubs/                 # NAS '24 paper sources + reference papers
├── LEADER-BOARD.md                # Race source-of-truth (per-corpus standings)
├── LLNL-OPERATING-CHARTER.md / LANL-OPERATING-CHARTER.md   # Operating contracts
├── MAP-LLNL.md / MAP-LANL.md      # Cognitive maps of llgan/ and altgan/
├── VERSIONS-LLNL.md / VERSIONS-LANL.md     # Per-version experiment logs
├── RESPONSE-LLNL.md / RESPONSE-LANL.md     # Per-claim postings (4-seed tables)
├── REBUTTAL-LLNL.md / REBUTTAL-LANL.md     # Cross-team rebuttals
├── PEER-REVIEW-LLNL.md / PEER-REVIEW-LANL.md / PEER-REVIEW-GEMINI.md   # Reviews
├── IDEAS-LLNL.md / IDEAS-LANL.md  # Per-team idea backlogs
├── R-ANALYSIS.md / TODO.md
└── assets/                        # Images
```

The active workflow:

- **`LEADER-BOARD.md`** — per-corpus generative + bootstrap leaderboard, multi-seed banked rows, open fronts.
- **`VERSIONS-LLNL.md` / `VERSIONS-LANL.md`** — append-only round logs; recipe, knobs, motivation, multi-seed result, post-mortem.
- **`RESPONSE-LLNL.md` / `RESPONSE-LANL.md`** — per-claim official postings (literal `cachesim mean HRC-MAE` lines + JSON means + 4-seed tables).
- **`PEER-REVIEW-*`** — running adversarial peer reviews; read both at the start of each session.
- **`IDEAS-LLNL.md` / `IDEAS-LANL.md`** — per-team backlogs of architecture / loss / curriculum bets.
- **`MAP-LLNL.md` / `MAP-LANL.md`** — cognitive maps; refresh when load-bearing code shape changes.

---

<p align="center">
  <img src="assets/zarathustra.jpg" width="480" alt="Zarathustra">
</p>
