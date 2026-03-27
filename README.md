# Zarathustra

> *"I am Zarathustra the godless — I cook every chance in my pot. And only when it has been thoroughly cooked do I welcome it as my food."*
> — Gathas, Yasna 43 (attrib. Zarathustra / Zoroaster, c. 1500–1000 BCE)

The project takes its name from the ancient Iranian prophet Zarathustra — Zoroaster in Greek — who articulated the first great dualist cosmology: Ahura Mazda, the Wise Lord of light and truth, in eternal struggle against Angra Mainyu, the destructive spirit of darkness and deception. From that primal opposition all creation emerges and all meaning is made.

Here the duality is literal: a **generator** and a **critic**, forever opposed, each made stronger by the other's resistance. The generator reaches toward the light of real data; the critic is the darkness that refuses to be fooled. Neither can exist without the other. What emerges from their adversarial tension is something neither could produce alone: synthetic I/O traces that are statistically indistinguishable from production reality.

Zarathustra generates synthetic I/O workload traces that are statistically indistinguishable from production traces. The goal is not simulation in the classical sense — hand-tuned Markov models or parameterized distributions — but learned generation: a model that has internalized the full joint distribution of request arrivals, object sizes, access patterns, and read/write ratios from hundreds of real-world storage traces.

## Motivation

Storage system researchers and engineers need realistic workloads for benchmarking, capacity planning, and stress testing. Real production traces are often proprietary, enormous, and tied to a single point in time. A generative model trained on a broad corpus of traces can produce arbitrary volumes of synthetic workload data that reflects the statistical character of real systems — burstiness, temporal correlations, size distributions — without exposing sensitive production data.

This project is inspired by the approach introduced in:

> D. Haskins, *"Accurate Generation of I/O Workloads Using Generative Adversarial Networks,"*
> IEEE MASCOTS 2024. ([PDF](pubs/Accurate_Generation_of_I_O_Workloads_Using_Generative_Adversarial_Networks.pdf))

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

Training data comes from the [libCacheSim cache-datasets](https://github.com/1a1a11a/libCacheSim) corpus in `oracleGeneral` binary format, covering:

- **Tencent Block** (2020): ~191 volumes, 217-hour traces, mixed 512B–32KB requests
- **Alibaba Block** (2020): ~238 volumes, 744-hour traces, 4KB-aligned block I/O

### Evaluation

Quality is tracked during training using **Maximum Mean Discrepancy (MMD²)** against a held-out validation set. Lower MMD² indicates the generated distribution is closer to real. The target is statistical indistinguishability — MMD² near zero across all marginal and joint distributions of the trace fields.

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
python train.py \
    --trace-dir /Volumes/Archive/Traces/s3-cache-datasets/cache_dataset_oracleGeneral/2020_tencentBlock \
    --fmt oracle_general \
    --files-per-epoch 8 \
    --records-per-file 15000 \
    --epochs 300 \
    --checkpoint-dir checkpoints/tencent_multifile
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
| `oracle_general` | libCacheSim 24-byte binary (oracleGeneral) |
| `spc` | ASU SPC-1 CSV |
| `msr` | Microsoft Research Cambridge traces |
| `k5cloud` | K5Cloud block traces |
| `systor` | Systor block traces |
| `csv` | Generic CSV (numeric columns auto-detected) |

## Repository layout

```
Zarathustra/
├── llgan/
│   ├── config.py       # Hyperparameter dataclass
│   ├── dataset.py      # Trace loading, preprocessing, sliding-window dataset
│   ├── model.py        # Generator and Critic (LSTM + spectral norm)
│   ├── mmd.py          # MMD² evaluation metric
│   ├── train.py        # Training loop (single-file and multi-file modes)
│   └── generate.py     # Inference: checkpoint → synthetic CSV
├── pubs/               # Reference papers
└── assets/
```

---

<p align="center">
  <img src="assets/zarathustra.jpg" width="480" alt="Zarathustra">
</p>
