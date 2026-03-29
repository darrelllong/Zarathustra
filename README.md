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

### Architecture

The core model is an **LLGAN** (LSTM + GAN):

- **Generator**: a single-layer LSTM that maps random noise vectors to sequences of I/O requests. Each timestep receives the same noise vector; the LSTM hidden state carries temporal context forward. Output is passed through a tanh projection to produce normalized `(timestamp_delta, object_id, object_size, opcode)` tuples.
- **Critic**: a single-layer LSTM with a spectrally-normalized output projection. Trained under Wasserstein loss to score sequences as real or generated. Spectral normalization enforces the Lipschitz constraint without gradient penalty, which avoids second-order autograd limitations on Apple MPS hardware.

### Training

Training uses **Wasserstein loss** with spectral normalization (WGAN + SN), which is more stable than the original BCE formulation and well-suited to sequence data with heavy-tailed distributions.

The key preprocessing choices:

- **Delta-encoded timestamps**: inter-arrival times rather than absolute timestamps make the distribution stationary and far easier for an LSTM to model. Inverse-transform via cumsum reconstructs absolute times at generation time.
- **Opcode binarization**: read → +1, write → −1, aligning with tanh output range.
- **Min-max normalization** to [−1, 1] per column, fit on training data only.

### Multi-file streaming training

Rather than training on a single trace file, Zarathustra samples a random subset of files from the trace corpus each epoch. This forces the model to learn the shared statistical structure across workloads rather than overfitting to one volume's access pattern. A preprocessor is fitted once on a seed sample of files and held fixed; normalization statistics remain consistent across all epochs and at generation time.

### Trace corpus

Training data comes from the [libCacheSim cache-datasets](https://github.com/1a1a11a/libCacheSim) corpus, hosted in a public AWS S3 bucket.  Both `oracleGeneral` and native `lcs` binary formats are supported.  See [**Downloading the traces**](#downloading-the-traces) below for instructions.

### Evaluation

Quality is tracked during training using **Maximum Mean Discrepancy (MMD²)** combined with **β-recall** (mode coverage) against a held-out validation set.  The combined score `MMD² + 0.2·(1−recall)` is used for checkpoint selection, which prevents saving mode-collapsed checkpoints that achieve low MMD² by ignoring rare burst events.  Lower is better; the target is statistical indistinguishability.

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

### Training (single file)

```bash
cd llgan
python train.py \
    --trace /path/to/trace.oracleGeneral.zst \
    --fmt oracle_general \
    --loss wgan-gp \
    --epochs 300
```

### Training (multi-file streaming, recommended)

```bash
# oracleGeneral format (fastest; 382 files for tencentBlock)
python train.py \
    --trace-dir /path/to/cache_dataset_oracleGeneral/2020_tencentBlock \
    --fmt oracle_general \
    --files-per-epoch 8 \
    --records-per-file 15000 \
    --epochs 300 \
    --checkpoint-dir checkpoints/tencent_v14

# LCS format (richer metadata; 4,482 files for tencentBlock)
python train.py \
    --trace-dir /path/to/cache_dataset_lcs/tencentBlock \
    --fmt lcs \
    --files-per-epoch 8 \
    --records-per-file 15000 \
    --epochs 300 \
    --checkpoint-dir checkpoints/tencent_lcs_v1
```

### Generating synthetic traces

```bash
python generate.py \
    --checkpoint checkpoints/tencent_multifile/final.pt \
    --n 1000000 \
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
├── llgan/
│   ├── config.py       # Hyperparameter dataclass
│   ├── dataset.py      # Trace loading, preprocessing, sliding-window dataset
│   ├── model.py        # Generator and Critic (LSTM + spectral norm)
│   ├── mmd.py          # MMD² training-time evaluation
│   ├── eval.py         # Full evaluation: MMD², PRDC, DMD-GEN, Context-FID, AutoCorr
│   ├── train.py        # Training loop (single-file and multi-file modes)
│   └── generate.py     # Inference: checkpoint → synthetic CSV
├── pubs/               # Reference papers
└── assets/
```

---

<p align="center">
  <img src="assets/zarathustra.jpg" width="480" alt="Zarathustra">
</p>
