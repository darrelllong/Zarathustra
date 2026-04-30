# MAP-LLNL.md — Cognitive Map of `llgan/`

LLNL's GAN-based synthetic I/O trace generator. This is a navigational map,
not a tutorial: it tells you where things are, how they connect, and what
load-bearing invariants you must respect when changing them.

Last refreshed: 2026-04-29 23:35 PDT (during v234d Phase 2.5 G-warm).

---

## 1. Top-level data flow

```
real .zst trace files
       │
       ▼
┌──────────────────────┐
│  dataset.py          │   parse oracle_general / spc / msr / k5cloud
│  TracePreprocessor   │   ─ delta-encode ts; signed-log obj_size
│                      │   ─ split obj_id → (obj_id_reuse, obj_id_stride)
│                      │   ─ auto-drop zero-variance cols
│                      │   ─ IDEA #97: obj_id_reuse can be LRU-hit-at-K
│                      │     instead of legacy consecutive (--lru-cache-depth)
└──────────────────────┘
       │ (per-file TraceDataset)
       ▼
┌──────────────────────┐
│  train.py            │   4-phase curriculum:
│                      │     Phase 1 — Autoencoder pretrain (Encoder+Recovery)
│                      │     Phase 2 — Supervisor pretrain (teacher-forcing)
│                      │     Phase 2.5 — Generator warm-up (imitate Sup)
│                      │     Phase 3 — Joint GAN (WGAN-SN + aux losses)
│                      │   Saves epoch_*.pt + best.pt (training-time selector)
│                      │   On --w-stop-threshold breach (3 consec epochs):
│                      │     stop, write final.pt
└──────────────────────┘
       │ (checkpoints)
       ▼
┌──────────────────────┐  ┌──────────────────────┐
│  frozen_sweep.py     │  │  generate.py         │
│  ★ = MMD² +          │  │  emit synthetic CSV  │
│      0.2·(1 − recall)│  │  with carried hidden │
│  on 4-file frozen    │  │  AND retrieval state │
│  bundle, seed=42     │  │  across windows      │
│  THIS IS THE ATB     │  │                      │
└──────────────────────┘  └──────────────────────┘
       │                         │
       ▼                         ▼
┌──────────────────────┐  ┌──────────────────────┐
│  eval.py             │  │  long_rollout_eval.py│
│  MMD² / PRDC / DMD-  │  │  HRC-MAE / reuse_    │
│  GEN / Context-FID / │  │  access / stack      │
│  AutoCorr / HRC      │  │  median+p90 /        │
│                      │  │  footprint / drift / │
│                      │  │  mark_score          │
└──────────────────────┘  └──────────────────────┘
                                 (parity panel for LANL)
```

---

## 2. File index, grouped by role

### Core training pipeline (always touched)

| File | Lines | Role |
|---|---|---|
| `train.py` | 3270 | 4-phase curriculum, all loss assembly, W-stop guard, IDEA #115 LRU diagnostic. Single `train()` entry point at L304; CLI at `parse_args()` L2632. |
| `model.py` | 1160 | All nn.Modules: Encoder, Recovery, Supervisor, LatentDiscriminator, CondEncoder, RegimeSampler, GMMPrior, GPPrior, **Generator**, Critic, MultiScaleCritic. |
| `config.py` | 260 | `Config` dataclass — all hyperparameters with defaults. CLI flags map onto config attrs in `train.py:parse_args`. |
| `dataset.py` | 1135 | File readers (`_READERS` for oracle_general/spc/msr/k5cloud/csv), `TraceDataset`, `TracePreprocessor`. **IDEA #97 lives in `_apply_obj_locality`.** |

### Training-time auxiliary modules (loaded conditionally on flags)

| File | Lines | Activated by | Role |
|---|---|---|---|
| `retrieval_memory.py` | 294 | `--retrieval-memory` (IDEA #17) | persistent attention memory; M=32 default; **bank not saturated during T=12 training, only at long-rollout — Gemini Round 3 P1 #2 / IDEA #117 open** |
| `ssm_backbone.py` | 229 | `--ssm-backbone` (IDEA #19) | selective diagonal SSM as alternative to LSTM body |
| `chunk_stitching.py` | 221 | `--boundary-smoothness-weight`, `--overlap-consistency-*` (IDEA #21) | BS sub-loss (latent-space) + OC sub-loss (feature-space). Palindrome bug in `boundary_latent_smoothness` was patched (Gemini Round 3 P1 #1). |
| `boundary_critic.py` | 145 | `--boundary-critic-*` (IDEA #36) | learned critic over adjacent-window pairs |
| `pcf_loss.py` | 115 | `--pcf-loss-weight` | path characteristic function loss; multi-scale freq matching |
| `timing_head.py` | 178 | `--mtpp-timing` (IDEA #20) | marked temporal point process timing head |
| `cache_descriptor.py` | 294 | `--cache-descriptor` (IDEA #18) | distilled cache-descriptor target |
| `hybrid_diffusion.py` | 322 | `--hybrid-diffusion-aux` (IDEA #22) | latent diffusion stage aux head |
| `tail_strata.py` | 196 | `--tail-strata` (IDEA #34) | tail-heaviness-stratified eval bundle |

### Eval / scoring (downstream of training)

| File | Lines | Role |
|---|---|---|
| `eval.py` | 755 | Frozen-bundle protocol (`--eval-real-seed 42`); MMD²/PRDC/DMD-GEN/Context-FID/AutoCorr/HRC. **`_per_window_hrc` padding bug fixed; conditioning bypass fixed; reuse-rate column index now dynamic — all per Gemini Round 1.** |
| `frozen_sweep.py` | 254 | Sweeps every checkpoint via `eval.py`, ranks by ★, promotes `frozen_best.pt`. Round 15 protocol — `--eval-real-seed 42` is **mandatory** for ATB claims. |
| `mmd.py` | 368 | Multi-scale RBF MMD² (unbiased, diagonal excluded). DMD-GEN via pseudo-inverse. Combined-score helper used by training-time best.pt selection (mis-ranks vs frozen ★). |
| `long_rollout_eval.py` | 1020 | The **panel LANL uses**: HRC-MAE, reuse_access, stack median+p90, footprint, drift, mark_score over 100k chained-window generation against `tencent_stackatlas.json` real manifest. LLNL has been delinquent on running this for v229 ★=0.039. |
| `compare.py` | 602 | File-by-file fidelity: load real trace vs synthetic CSV, score side-by-side. |
| `eval_pregenerated.py` | 109 | Score a pre-generated CSV without re-generating. IDEA #48 experiments. |

### Generation-time path

| File | Lines | Role |
|---|---|---|
| `generate.py` | 538 | Loads checkpoint, threads `(h_n, c_n)` AND `retrieval_state` across windows for long traces. Optional `--lru-stack-decoder` post-hoc rewrite. |
| `lru_stack_decoder.py` | 556 | Post-hoc LRU-rank rewrite of obj_id sequence (IDEA #48). |
| `calibrate_lru_per_stream.py` | 435 | Per-stream LRU-PMF calibration. |
| `phase_pmf_atlas.py` | 692 | LLNL Phase-Conditioned PMF Atlas (IDEA #65) — alibaba ATB ★=0.001937 came from this path. |
| `stack_atlas.py` | 494 | Per-object Markov state generator (IDEA #64). |
| `compute_markov_atlas.py` | 165 | Empirical Markov transitions over stack-rank from real traces. |
| `compute_cond_pmf.py` | 233 | Conditional stack-distance PMF P(bucket | dt_bin). |
| `precompute_descriptors.py` | 146 | Precompute cache descriptors per real trace → JSONL. |

### Profiling / utilities (rarely touched)

| File | Lines | Role |
|---|---|---|
| `trace_profile.py` | 589 | Per-file characterization (Hurst, changepoints, regime stats). |
| `lcs_survey.py` | 357 | LCS trace-format survey tool (one-shot). |

---

## 3. Generator forward — the load-bearing path

`model.py:Generator` (L524–928) is the most-touched class. Its forward signature:

```python
G(z_global, z_local,
  hidden=None, return_hidden=False,
  retrieval_state=None, return_retrieval_state=False,
  cond=None) -> H  (or H, h_carry  or  H, h_carry, r_carry)
```

`z_global` is per-stream (workload identity); `z_local` is per-step innovation
(B, T, noise_dim). In **training** (`train.py`), z_global is constructed via
`_make_z_global()` (L87) routing `cond` through `cond_encoder` →
`regime_sampler` → noise sampling. In **eval** (`eval.py:_sample_fake`) and
**generate** (`generate.py`), the same routing path must be used or
checkpoints with regime/var-cond/gmm components evaluate as garbage (Gemini
Round 1 P1 #3 — fixed; do not regress).

`hidden` carries LSTM state across windows; `retrieval_state` carries the
retrieval-memory bank read-state across windows. **In training, the main
critic/G forward DOES NOT thread `retrieval_state`** — only the BS/OC
sub-losses do (`train.py:2124-2195`). Eval and long-rollout generation DO
thread it. **This is the IDEA #117 / Gemini Round 3 P1 #2 gap.**

---

## 4. Phase 3 loss assembly (train.py L1500–2300, simplified)

For each batch in Phase 3:

```
G_loss = wgan_g_loss
       + supervisor_consistency
       + moment_loss_weight        × moment_match           (--moment-loss-weight)
       + pcf_loss_weight           × pcf_loss               (--pcf-loss-weight)
       + reuse_rate_loss_weight    × (reuse_rate − target)² (--reuse-rate-loss-weight,
                                                             --reuse-rate-target;
                                                             hybrid surrogate via IDEA #79)
       + boundary_smoothness_weight × bs_sub_loss           (--boundary-smoothness-*)
       + overlap_consistency_weight × oc_sub_loss           (--overlap-consistency-*)
       + acf_chain_weight           × acf_chain_loss        (--acf-chain-*)
       + long_chain_weight          × long_chain_loss       (--long-chain-*; IDEA #116; closed-failed v233)
       + boundary_critic_weight     × adv loss vs D_bc      (--boundary-critic-*)
       + mtpp_timing_weight         × mtpp loss             (--mtpp-timing-*)
       + hybrid_diffusion_weight    × diffusion aux         (--hybrid-diffusion-aux)
       + cache_descriptor_weight    × distillation          (--cache-descriptor-*)

C_loss = wgan_critic_loss            (multi-scale if --multi-scale-critic; IDEA #15)
W_estimate = E[C(real)] − E[C(fake)] → tracked for W-stop guard
```

Default flags ON in race recipes: `--retrieval-memory`, `--multi-scale-critic`,
`--mixed-type-recovery`, `--pcf-loss-weight 0.5`, `--moment-loss-weight 0.5`,
`--reuse-rate-loss-weight 10.0`, `--reuse-rate-target 0.70`. v229's tencent
ATB was built on this set (no long-chain, no boundary-critic).

---

## 5. Training selectors and gates (where decisions are made)

| Decision | Where | Rule |
|---|---|---|
| Best epoch (training-time) | `train.py` ★ block ~L2400 | combined = MMD²_EMA + 0.2·(1−recall_EMA) [+ optional dmd_ckpt_weight·DMD-GEN]; **mis-ranks vs frozen ★ by up to 5×** |
| W-stop | `train.py:~L2440` | W > w_stop_threshold for 3 consecutive epochs → stop, write final.pt |
| Frozen ATB | `frozen_sweep.py` | run `eval.py --eval-real-seed 42 --eval-fake-seed 42` over all epoch_*.pt + best.pt; promote `frozen_best.pt` symlink |
| Kill threshold (race-mode) | external (cron prompt) | 30 epochs stale from best ★ |

---

## 6. Critical invariants — DO NOT BREAK

1. **`--eval-real-seed 42`** is mandatory for ATB claims. Without it the eval resamples a random 4-file real bundle, conflating model variance with benchmark variance (Round 15).
2. **train.py must be invoked from `llgan/` directly**: `cd /home/darrell/Zarathustra/llgan && python train.py …`. `python -m llgan.train` fails because `train.py:43` does `from config import Config` (bare import). The IDEA #115 LRU diagnostic also fails this way (`from llgan.lru_stack_decoder` requires package context); `lru_eval=ERR(ModuleNotFoundError)` in v232/v233/v234 logs are this bug. **Pending fix per LANL R1.**
3. **`--lru-cache-depth K` (IDEA #97) shifts `obj_id_reuse` semantics**: legacy (K=0) is consecutive-same-obj (~3% positive rate for tencent); K>0 is LRU-hit-at-K (~61.5% positive rate at K=15000). v229 was trained pre-IDEA #97 (legacy); reusing v229's pretrain with `--lru-cache-depth 15000` does not crash but feeds wildly different reuse distribution to weights expecting legacy → unknown effect (so far inconclusive — see Round 138-140).
4. **`retrieval_state` must be threaded** at long-rollout time (`generate.py`, `long_rollout_eval.py`); training-time main forward currently does NOT thread it (IDEA #117 gap).
5. **The frozen-bundle protocol is short-window (T=12)**; a good ★ does NOT imply good HRC-MAE on 100k chained windows. Promotion must include the long-rollout panel (Darrell PEER-REVIEW-LANL.md Round 45 P0).
6. **Preprocessor schema is part of the experiment definition.** `_fit_prep_on_files` samples `n_seed = files_per_epoch` files via a seeded `random.sample`. Min/max bounds, auto-dropped zero-variance columns, and feature ordering all depend on which files were sampled — Gemini Round 1 P1 #5. Pretrain checkpoints carry an implicit binding to the preprocessor that produced them.

---

## 7. Open-bug summary (race-blocking)

| Bug | Source | Status |
|---|---|---|
| LRU diagnostic import error in train.py | LANL R1, Sandia VERSIONS-Sandia L94 | open |
| LRU diagnostic doesn't thread retrieval_state | LANL R1 | open |
| Main training doesn't thread retrieval_state (IDEA #117) | Gemini R3 P1 #2 | open — needs `--retrieval-train-carry` flag |
| No durable training log in checkpoint dirs | LANL R1 | open — fix in launcher |
| v229 ATB not reproducible from cloned pretrain | self-observed Rounds 138-140 | testing v234d (fresh pretrain) |
| LLNL has not run long-rollout panel for v229 ATB | Darrell R45 P0 | blocked on v234d ep10 |

---

## 8. Where the bodies are buried (round-log highlights)

- **v164 alibaba ★=0.03457**: legacy buggy-BS code (palindrome bug now patched). NOT reproducible under current code. Kept as numeric threshold but tainted.
- **v195 ep110 alibaba ★=0.001937 (current ATB)**: from PhaseAtlas (`phase_pmf_atlas.py`) path, NOT the GAN. Different generation pipeline than v229.
- **v229 ep10 tencent ★=0.039 (current ATB)**: GAN path (Phase 3 of train.py), seed=5. ATB-claiming run that has NOT been cross-seed validated. v234* attempts to reproduce keep failing (W-stop or ★=0.26).
- **v233 (closed-failed)**: IDEA #116 long-chain loss with seed=7. Frozen ★=0.26224. Hypothesis rejected.
- **v234, v234b, v234c (closed-failed)**: v229 recipe with v229's pretrain cloned. All three W-blew up. Round 139's `--lru-cache-depth` schema-mismatch hypothesis was rejected by v234c being bit-identical to v234b at ep1-2.
- **v234d (running)**: v229 recipe with FRESH pretrain. AE+Sup converged better than v229's. Phase 3 ep1 ETA ~23:35 PDT.

---

## 9. How to update this map

When you change anything load-bearing:
- Add or remove a file → update §2.
- Change Generator forward signature → update §3.
- Add a flag-gated loss → update §4.
- Change a selector / gate → update §5.
- Land a fix or open a new gap → update §6 / §7.

Keep the map under 350 lines. If a section grows past 50 lines, factor the
detail into a separate doc and link it from here.
