# Zarathustra

> *"I am Zarathustra the godless — I cook every chance in my pot. And only when it has been thoroughly cooked do I welcome it as my food."*
> — Gathas, Yasna 43 (attrib. Zarathustra / Zoroaster, c. 1500–1000 BCE)

> *"Iron sharpeneth iron; so a man sharpeneth the countenance of his friend."*
> — Proverbs 27:17

The project takes its name from the ancient Iranian prophet Zarathustra — Zoroaster in Greek — who articulated the first great dualist cosmology: Ahura Mazda, the Wise Lord of light and truth, in eternal struggle against Angra Mainyu, the destructive spirit of darkness and deception. From that primal opposition all creation emerges and all meaning is made.

Here the duality is literal: a **generator** and a **critic**, forever opposed, each made stronger by the other's resistance. The generator reaches toward the light of real data; the critic is the darkness that refuses to be fooled. Neither can exist without the other. What emerges from their adversarial tension is something neither could produce alone: synthetic I/O traces that are statistically indistinguishable from production reality.

Zarathustra generates synthetic I/O workload traces that are statistically indistinguishable from production traces. The goal is not simulation in the classical sense — hand-tuned Markov models or parameterized distributions — but learned generation: a model that has internalized the full joint distribution of request arrivals, object sizes, access patterns, and read/write ratios from hundreds of real-world storage traces.

## Motivation

Storage system researchers and engineers need realistic workloads for benchmarking, capacity planning, and stress testing. Real production traces are often proprietary, enormous, and tied to a single point in time. A generative model trained on a broad corpus of traces can produce arbitrary volumes of synthetic workload data that reflects the statistical character of real systems — burstiness, temporal correlations, size distributions — without exposing sensitive production data.

This project is inspired by the approach introduced in:

> H. Zhang, Z. Yang, Y. Xie, Y. Wu, J. Li, D. Feng, A. Wildani, and D. Long,
> *"Accurate Generation of I/O Workloads Using Generative Adversarial Networks,"*
> in *Proceedings of the 17th IEEE International Conference on Networking, Architecture, and Storage (NAS 2024)*, Guangzhou, China, Nov. 2024. ([PDF](pubs/Accurate_Generation_of_I_O_Workloads_Using_Generative_Adversarial_Networks.pdf))

That paper established that LSTM-based GANs (LLGANs) can learn and reproduce the statistical properties of block I/O traces — inter-arrival times, request sizes, and opcode distributions — more faithfully than classical models. Zarathustra extends that foundation toward generalization: rather than fitting one model to one trace, the goal is a single model that captures the workload distribution across an entire corpus.

## Approach

The original LLGAN paper used a single-layer LSTM Generator and Critic trained under Wasserstein loss. Zarathustra has since grown into a **four-phase curriculum** with a latent autoencoder, a teacher-forcing supervisor, optional FiLM/GMM/regime/retrieval/state-space modules, multi-scale critics, and a Path-Characteristic-Function adversarial loss. The Generator and Critic are still the heart of the system; everything else exists to put them in a good basin and keep them honest.

### Pipeline

```
   real traces                       latent space                feature space
  ─────────────                     ───────────────             ─────────────
   raw windows  ──Encoder──►    H_real ── Critic / GAN  ◄──   fake windows
   (B, T, F)                    (B, T, L)                     ▲
                                       ▲                      │
                                  z_global ────► Generator    │
                                  z_local                     │
                                  (cond, regime, …)           │
                                       │                      │
                                       └──Recovery─►   fake_decoded
                                                       (B, T, F)
```

- **Autoencoder (E + R)**: maps raw feature windows to a compact latent space (`latent_dim`, default 24). The Generator never sees raw features; it produces latent windows that the Recovery decoder maps back to feature space. This decouples generation from preprocessing.
- **Generator**: produces latent windows from `(z_global, z_local)`. `z_global` is sampled once per stream (via optional GMM prior conditioned on workload descriptors); `z_local` is per-step innovation noise. Default backbone is a single-layer LSTM; `--ssm-backbone` swaps in a Mamba-lite SelectiveDiagonalSSM. Optional modules: `--film-cond` (FiLM modulation per step), `--gmm-components` (mixture prior over workload regimes), `--var-cond` (variational conditioning), `--n-regimes` (Gumbel-Softmax regime sampler), `--retrieval-memory` (per-window object memory with reuse gate), `--gp-prior` (GP prior on `z_local`).
- **Supervisor**: a small auxiliary network trained alongside the autoencoder to predict the next latent step from the previous one (teacher forcing). Provides a stable supervisory signal in the early phases before the GAN takes over.
- **Critic**: spectrally-normalized LSTM that scores latent (and optionally decoded) windows. `--multi-scale-critic` adds critics at scales `T`, `T/2`, `T/4` for hierarchical realism. `--pcf-loss-weight` adds an adversarial Path-Characteristic-Function head that matches the empirical characteristic function of fake to real on each batch.
- **Recovery**: a small decoder mirroring the Encoder; turns latent windows back into feature space at training time (for feature-space losses) and at generation time.

### Four-phase curriculum

1. **AE pretrain** (50 ep) — Encoder + Recovery learn to round-trip real windows. Pure reconstruction loss.
2. **Supervisor pretrain** (50 ep) — Supervisor learns to predict next-step latent from real trajectories.
3. **G warm-up** (100 ep) — Generator learns to imitate the supervisor's outputs (no Critic gradients yet); stabilises against the latent geometry before adversarial training begins.
4. **Joint GAN** (200 ep) — Full WGAN-SN training of G vs Critic with all auxiliary losses live. Best checkpoint selected by combined metric on a held-out val bundle every 5 epochs (`★`).

Hot-starting from a prior pretrain checkpoint via `--resume-from` is the norm; only the wall-time-cheap Phase 4 is repeated when sweeping recipes.

### Conditioning

Per-file workload characterisations live in `traces/characterization/trace_characterizations.jsonl` (30,628 files, derived offline from R analysis: write_ratio, burstiness_cv, reuse_ratio, stride stats, IAT quantiles, …). At training time, each window's `cond` vector is the precomputed file-level descriptor; at generation time, the user supplies a target descriptor (or samples one from the corpus distribution).

Classifier-Free Guidance (`--cond-drop-prob`, default 0.25) randomly nulls the conditioning during training so the Generator learns both conditional and unconditional distributions.

### Evaluation

Training-time scoring uses **MMD² + 0.2·(1−β-recall)** on a held-out validation bundle every 5 epochs; the lowest-combined checkpoint is saved as `best.pt` (★). Lower is better. β-recall is the mode-coverage component of PRDC and prevents the model from gaming MMD² by collapsing to a few modes.

**Frozen-bundle protocol** (Round 15 P1 fix): pass `--eval-real-seed 42` to `eval.py` to pin the same 4 real files across runs. The original moving-bundle protocol conflated model variance with benchmark variance; under the frozen protocol numbers are roughly 2× higher than the legacy "5-run avg" reports. **All ATB-claiming evals must use the frozen protocol.** Current frozen-bundle bests:

| Corpus | Best frozen ATB | Version | Recipe |
|---|---|---|---|
| Alibaba | 0.176 | v114 | multi-scale critic + continuity loss |
| Tencent | 0.178 | v136 | multi-scale critic + PCF |

Full evaluation (`eval.py`) additionally reports **PRDC** (precision/recall/density/coverage), **DMD-GEN** (dynamic-mode-decomposition similarity), **Context-FID**, **autocorrelation distance**, and **HRC-MAE** (hit-ratio-curve fidelity — a cache-native metric).

## Downloading the traces

The training data is part of the public **libCacheSim cache-dataset** corpus, maintained by Juncheng Yang et al. and hosted in a public AWS S3 bucket.  No account or credentials are required.

### Available datasets

| Dataset | Period | Format files | Size | Character |
|---------|--------|-------------|------|-----------|
| **Tencent Block 2020** | 9 days | 382 oracleGeneral | 152 GB | 512B–32KB blocks, 83–94% reads |
| **Alibaba Block 2020** | 31 days | 242 oracleGeneral | 93 GB | 4KB-aligned, mixed read/write |
| **Tencent Photo 2018** | — | 2 oracleGeneral | 52 GB | Object store, large objects |
| **Twitter 2020** | — | 54 oracleGeneral | 142 GB | CDN/KV cache, high fan-out |
| **Meta KV 2022** | — | 3 oracleGeneral | 29 GB | Key-value cache, TTL-heavy |
| **CloudPhysics 2015** | — | 106 oracleGeneral | 8.6 GB | VM disk I/O, varied workloads |
| **MSR Cambridge 2007** | — | 14 oracleGeneral | 1.5 GB | Enterprise disk traces |
| **Meta CDN 2022** | — | 3 oracleGeneral | 2.1 GB | CDN edge cache |
| **Tencent Block LCS** | 9 days | 4,482 lcs | 662 GB | Same corpus, native LCS format |
| **Alibaba Block LCS** | 31 days | 428 lcs | 292 GB | Same corpus, native LCS format |

The `oracleGeneral` format (24 bytes/record) is the quickest to get started with.  The `lcs` files contain the same workloads with richer metadata (op codes, tenant IDs) and are split into more files, giving finer-grained per-volume models.

### Quick start (AWS CLI)

```bash
# Install AWS CLI if needed:  brew install awscli  or  pip install awscli

# Tencent Block 2020 — oracleGeneral format (~152 GB)
aws s3 sync s3://cache-datasets/cache_dataset_oracleGeneral/2020_tencentBlock/ \
    ./traces/2020_tencentBlock/ --no-sign-request

# Alibaba Block 2020 — oracleGeneral format (~93 GB)
aws s3 sync s3://cache-datasets/cache_dataset_oracleGeneral/2020_alibabaBlock/ \
    ./traces/2020_alibabaBlock/ --no-sign-request

# Twitter 2020 — oracleGeneral format (~142 GB; CDN workload, very different character)
aws s3 sync s3://cache-datasets/cache_dataset_oracleGeneral/2020_twitter/ \
    ./traces/2020_twitter/ --no-sign-request

# Tencent Block — native LCS format (~662 GB; 4,482 files, more per-volume detail)
aws s3 sync s3://cache-datasets/cache_dataset_lcs/tencentBlock/ \
    ./traces/lcs/tencentBlock/ --no-sign-request
```

If you only want a manageable sample to get started, download one size bucket:

```bash
# ~10–20 GB subset: volumes with 100K–1M unique objects
aws s3 sync s3://cache-datasets/cache_dataset_oracleGeneral/2020_tencentBlock/1M/ \
    ./traces/2020_tencentBlock/ --no-sign-request
```

### S3 format prefixes

| S3 prefix | Format flag | Notes |
|-----------|-------------|-------|
| `s3://cache-datasets/cache_dataset_oracleGeneral/` | `oracle_general` | 24-byte binary; fastest to load |
| `s3://cache-datasets/cache_dataset_lcs/` | `lcs` | Versioned binary with op/tenant; 4× more files |
| `s3://cache-datasets/cache_dataset_txt/` | `csv` | Plain-text; largest on disk |
| `s3://cache-datasets/cache_dataset_parquet/` | — | Parquet; not yet supported |

### File formats

**oracleGeneral** (24 bytes/record):
```
uint32  timestamp            (seconds since epoch)
uint64  object_id            (hash of key or LBA)
uint32  object_size          (bytes)
int32   next_access_vtime    (forward pointer; dropped at load time)
int16   op                   (0=read, 1=write)
int16   tenant_id
```

**lcs** (native libCacheSim binary, versioned):
```
Header: 8192 bytes
  [0:8]   magic   = 0x123456789abcdef0
  [8:16]  version (uint64: 1 or 2; determines record size)
  [16:]   trace statistics (n_req, n_obj, size histograms, …)

v1 record (24 bytes):
  uint32  clock_time   (seconds)
  uint64  obj_id
  uint32  obj_size     (bytes)
  int64   next_access_vtime   (dropped at load time)

v2 record (28 bytes, adds op and tenant):
  uint32  clock_time
  uint64  obj_id
  uint32  obj_size
  uint32  packed       (bits 0-7 = op, bits 8-31 = tenant_id)
  int64   next_access_vtime

op codes (libCacheSim enum.h): OP_READ=12, OP_WRITE=13
```

Zarathustra's `dataset.py` reads both formats natively (flags `--fmt oracle_general` and `--fmt lcs`); no external tools needed.  For other analysis the [libCacheSim](https://github.com/1a1a11a/libCacheSim) C library and Python bindings provide a full reader.

### Citation

If you use these traces, please cite the libCacheSim paper:

> Juncheng Yang, Yao Yue, and K.V. Rashmi.  "A large-scale analysis of hundreds of in-memory key-value cache clusters at Twitter."  *ACM Transactions on Storage*, 2021.

and the cache-dataset collection paper:

> Juncheng Yang, Ziyue Qiu, Yazhuo Zhang, Yao Yue, and K.V. Rashmi.  "FIFO queues are all you need for cache eviction."  *SOSP 2023*.

---

## Usage

### Pretrain once, sweep many

The autoencoder, supervisor, and warm-up Generator (Phases 1–2.5) are corpus-level work; only the Joint-GAN phase changes between recipes. Build a Phase-1+2+2.5 checkpoint once per corpus, then `--resume-from` it for every Phase-3 sweep.

```bash
# One-time pretrain (alibaba)
python -u llgan/train.py \
    --trace-dir /path/to/2020_alibabaBlock --fmt oracle_general \
    --files-per-epoch 12 --records-per-file 20000 \
    --char-file traces/characterization/trace_characterizations.jsonl \
    --epochs 0 \
    --checkpoint-dir checkpoints/alibaba_pretrain
```

### Training a recipe (Phase-3 only)

A representative production launch — multi-scale critic + PCF + chunk-stitching, with frozen-bundle eval pinned for ATB tracking:

```bash
python -u llgan/train.py \
    --trace-dir /path/to/2020_alibabaBlock --fmt oracle_general \
    --char-file traces/characterization/trace_characterizations.jsonl \
    --resume-from checkpoints/alibaba_pretrain/pretrain_complete.pt \
    --files-per-epoch 12 --records-per-file 20000 \
    --epochs 200 \
    --multi-scale-critic --multi-scale-weight 0.5 \
    --pcf-loss-weight 0.3 \
    --continuity-loss-weight 0.2 \
    --boundary-smoothness 0.1 \
    --cond-drop-prob 0.25 \
    --eval-real-seed 42 \
    --checkpoint-dir checkpoints/alibaba_v118
```

Frequently-used capability flags (full list in `train.py --help`):

| Flag | Effect |
|---|---|
| `--multi-scale-critic` (+ `--multi-scale-weight`) | Hierarchical critics at T, T/2, T/4 |
| `--pcf-loss-weight W` | Path-Characteristic-Function adversarial head |
| `--continuity-loss-weight W` | Penalises latent jumps between consecutive steps (alibaba bread-and-butter) |
| `--boundary-smoothness W` | Chunk-stitching loss for cross-window continuity |
| `--retrieval-memory` | Per-window object memory + reuse gate (tencent recipe) |
| `--ssm-backbone` (+ `--ssm-state-dim`) | Replace Generator LSTM with Mamba-lite SelectiveDiagonalSSM |
| `--film-cond` / `--var-cond` | FiLM modulation / variational conditioning |
| `--gmm-components K` / `--n-regimes K` | GMM prior / Gumbel regime sampler |
| `--gp-prior` | Gaussian-process smoothness prior on z_local |
| `--cond-drop-prob P` | Classifier-Free Guidance dropout (default 0.25) |
| `--eval-real-seed 42` | **Frozen-bundle protocol — required for ATB-claiming runs** |

### Full evaluation

```bash
python -u llgan/eval.py \
    --checkpoint checkpoints/alibaba_v118/best.pt \
    --trace-dir /path/to/2020_alibabaBlock --fmt oracle_general \
    --char-file traces/characterization/trace_characterizations.jsonl \
    --eval-real-seed 42 \
    --report eval_reports/alibaba_v118.json
```

Reports: MMD², β-recall + combined ATB, PRDC, DMD-GEN, Context-FID, autocorrelation distance, HRC-MAE.

### Generating synthetic traces

```bash
python -u llgan/generate.py \
    --checkpoint checkpoints/alibaba_v118/best.pt \
    --char-file traces/characterization/trace_characterizations.jsonl \
    --n-records 1000000 \
    --output synthetic.csv
```

### Supported trace formats

| Format | Description |
|--------|-------------|
| `oracle_general` | libCacheSim 24-byte binary (oracleGeneral); fastest |
| `lcs` | libCacheSim native binary v1/v2 (.lcs.zst); includes op codes and tenant |
| `spc` | ASU SPC-1 CSV |
| `msr` | Microsoft Research Cambridge traces |
| `k5cloud` | K5Cloud block traces |
| `systor` | Systor block traces |
| `exchange_etw` | Windows ETW disk traces (.csv.gz; MSR SNIA Exchange Server) |
| `csv` | Generic CSV (numeric columns auto-detected) |

## Repository layout

```
Zarathustra/
├── llgan/                       # Core training + generation library
│   ├── config.py                # Hyperparameter dataclass
│   ├── dataset.py               # Multi-format trace loading, sliding-window dataset
│   ├── model.py                 # Generator (LSTM/SSM) + Critic + Encoder/Recovery + Supervisor
│   ├── ssm_backbone.py          # SelectiveDiagonalSSM (Mamba-lite) generator backbone
│   ├── retrieval_memory.py      # Per-window object memory + reuse gate
│   ├── chunk_stitching.py       # Cross-window boundary smoothness loss
│   ├── pcf_loss.py              # Path-Characteristic-Function adversarial head
│   ├── multi_scale_critic.py    # Hierarchical critics at T, T/2, T/4 (in model.py)
│   ├── cache_descriptor.py      # 8-dim cache-native descriptor monitor
│   ├── precompute_descriptors.py# Offline corpus-wide descriptor precomputation
│   ├── timing_head.py           # MTPP timing head (standalone, pending wiring)
│   ├── hybrid_diffusion.py      # Diffusion refinement head (standalone)
│   ├── trace_profile.py         # Per-trace statistical profiler
│   ├── mmd.py                   # MMD² + β-recall combined metric
│   ├── eval.py                  # Full eval: MMD², PRDC, DMD-GEN, Context-FID, AutoCorr, HRC-MAE
│   ├── compare.py               # Side-by-side checkpoint comparison
│   ├── lcs_survey.py            # LCS corpus inventory tool
│   ├── train.py                 # Training loop (4-phase curriculum)
│   └── generate.py              # Inference: checkpoint → synthetic CSV
├── traces/
│   └── characterization/        # Per-file workload descriptors (30,628 files)
├── characterizations/           # Aggregate corpus-level characterizations
├── R-scripts/                   # R analyses (Hurst, changepoints, regime detection)
├── parsers/                     # Trace-format parsers (SNIA, ETW, etc.)
├── scripts/                     # Operational + ad-hoc shell scripts
├── paper/                       # NAS 2024 paper sources + revisions
├── pubs/                        # Reference papers
├── assets/                      # Images
├── VERSIONS.md                  # Per-version experiment log (the run book)
├── IDEAS.md                     # Backlog of architecture / loss / curriculum ideas
├── PEER-REVIEW.md               # Darrell's peer-review notes
└── PEER-REVIEW-GEMINI.md        # Gemini peer-review notes
```

The active workflow lives in three files at the repo root:

- **`VERSIONS.md`** — append-only log of every Phase-3 launch: recipe, hyperparameters, motivation, training-★ trajectory, frozen-bundle ATB on completion, post-mortem.
- **`IDEAS.md`** — numbered backlog of pending architecture / loss / curriculum bets, with status (queued / wired / running / closed).
- **`PEER-REVIEW.md`** + **`PEER-REVIEW-GEMINI.md`** — the two outside reviewers' running feedback. Read both at the start of each session.

---

<p align="center">
  <img src="assets/zarathustra.jpg" width="480" alt="Zarathustra">
</p>
