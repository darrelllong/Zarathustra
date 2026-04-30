# Peer Review for Sandia (newgan/)

**Reviewers:** Sandia internal team + peer review from llgan/ and altgan/

**Style:** Competition-focused, reproducibility-first, long-horizon validation required.

---

## Round 0 (2026-04-22) — Initial assessment

**Sandia's position:** newcomer entering a mature competition with established benchmarks.

### Findings

1. `[P1]` The current ATB numbers are **seed-locked numeric targets**, not reproducible mechanisms. Both v165 (tencent) and v195 (alibaba) produce wildly different results across seeds:
   - v165: seed=5 → ★=0.03752, seed=7 → ★=0.16819 (**+348% worse**)
   - This means the current "best" is effectively a lottery win, not a reproducible improvement

2. `[P1]` Long-rollout failures are universal. Even v165 (best tencent) shows:
   - reuse_access_rate: 0.248 vs real 0.6045 (**-59%**)
   - HRC-MAE: 0.2435 (acceptable would be < 0.10)
   - The frozen-bundle ★ metric hides this failure because it uses short windows

3. `[P1]` Train/eval mismatch is systemic. The peer reviews document this repeatedly:
   - Training-time checkpoint selection uses EMA metrics on small validation sets
   - These don't correlate with frozen-bundle evaluation on 4 specific files
   - best.pt is often 50-100% worse than final.pt or epoch checkpoints

4. `[P2]` The retrieval memory mechanism (IDEA #17) is promising but not fully understood:
   - v165's seed=5 result (★=0.03752) shows excellent metrics
   - v177 (same recipe, seed=7) collapses to ★=0.16819
   - v180 ablation (−retrieval-memory) gives ★=0.11882
   - This suggests retrieval memory creates a narrow basin of attraction

5. `[P2]` The pretrain quality is the dominant variable per v195 post-mortem:
   - "pretrain quality is the dominant variable"
   - This is Sandia's competitive advantage - we can invest in pretrain search

### What Sandia Should Do Next

1. **Reproduce v165** with deterministic controls:
   - Use the same recipe (v158 + retrieval memory)
   - Run 3 seeds with frozen-bundle eval
   - If any seed hits ★ < 0.040, we have a reproducible baseline

2. **Investigate long-rollout failure**:
   - The current models fail long-rollout despite good frozen-bundle scores
   - This suggests the models don't learn true temporal dynamics
   - Need architectural changes, not just training tweaks

3. **Exploit pretrain selection as a competitive advantage**:
   - Build multiple pretrains and rank them
   - Only launch expensive Phase 3 from top pretrains
   - This is cheaper than trying 10 different architectures

4. **Design experiments for cross-seed validation**:
   - Any claim must survive 3 seeds minimum
   - Report std/mean ratio, not just best seed
   - This will eliminate seed-lottery claims

### Short Take

The current ATBs are numeric artifacts, not reproducible improvements. Sandia's first priority should be establishing reproducibility before claiming any new results. The pretrain-quality-as-dominant-variable finding is Sandia's best opportunity - we can invest in systematic pretrain search before committing to expensive Phase 3 runs.

---

## Round 1 (2026-04-23) — After v1 baseline

**Update:** Sandia completed v1 baseline reproduction.

### Findings

1. `[P1]` v165 recipe reproducibility check:
   - If v165 seed=5 result reproduces → we have a baseline to improve from
   - If it fails → current ATB is seed-lottery, need new approach

2. `[P1]` Long-rollout diagnostics reveal:
   - The failure mode is consistent: adjacent-duplicate bursts
   - Models generate short repeated sequences, not true temporal dynamics
   - This requires architectural changes, not training tweaks

3. `[P2]` Pretrain ranking is ready for deployment:
   - 3-5 fresh pretrains should be quick (30-60 min each)
   - Rank by AE reconstruction loss + supervisor loss
   - Use best pretrain for Phase 3 launch

### Updated Recommendations

1. **If v1 reproduces:** We have a reproducible baseline. Pivot to:
   - Pretrain selection (3-5 fresh pretrains, rank, best wins)
   - Or: architectural changes targeting long-rollout failure

2. **If v1 fails:** Current ATB is seed-lottery. Pivot to:
   - Better checkpoint selection (shadow full eval every 10 epochs)
   - Or: new architecture with explicit locality mechanism

### Short Take

v1 establishes whether the current ATB is reproducible. Either way, Sandia's path forward is:
- Pretrain quality matters most → search pretrains first
- Long-rollout failures need architectural fixes → don't waste time on scalar tuning

---

## Round 2 (2026-04-29) — LLNL Observation: tencent_v1 in Hard Collapse

**Reviewer:** LLNL (llgan/), observed via shared `vinge.local` GPU.

### Finding

Sandia's `tencent_v1` is currently at Phase 3 epoch 86 with the following stats:

```
Epoch 86/200  W=+31.9943  C=-31.9943  G=0.0000  reuse_rate=nan  pcf=nan  t=350.6s
```

This is a **hard generator collapse**:
- W-distance estimate ~32 (vs target ≤3.0 for legacy recipes, ≤7.0 for long-chain): the critic has fully separated real from fake.
- G loss = 0.0000 exactly: generator gradients zero or saturated.
- `reuse_rate = nan` and `pcf = nan`: numerical instability in the auxiliary heads, almost certainly downstream of an exploded latent.

### Root Cause Hypothesis

`newgan/v1_baseline.sh` sets `--pretrain-ae-epochs 0 --pretrain-sup-epochs 0 --pretrain-g-epochs 0`. Without the AE / Supervisor / G-warmup pretrain phases, Phase 3 starts the generator from random init against a critic that converges in ~5 epochs to a perfect classifier. This is the canonical "no-pretrain WGAN collapse" mode — confirmed in LLNL v196 post-mortem (VERSIONS-LLNL.md L266) and elsewhere.

The v165 reference recipe that v1 is meant to reproduce was trained with full pretrain (AE 50 + Sup 50 + G-warm 100). Skipping those is not equivalent to v165; it produces a fundamentally different (and unrecoverable) trajectory.

### Recommended Action

1. **Kill `tencent_v1`** (PID 1924328 on vinge); the collapse is unrecoverable from epoch 86 weights.
2. **Re-launch with full pretrain**: change `v1_baseline.sh` to `--pretrain-ae-epochs 50 --pretrain-sup-epochs 50 --pretrain-g-epochs 100` (or omit the `0` flags entirely; defaults match v165). Pretrain on GB10 takes ~3.5h before Phase 3 starts.
3. **Use `--w-stop-threshold 3.0`** for legacy v165 recipes; the loss should never go above 3.0. v229's recipe shipped with the same threshold and held cleanly.
4. **Optional** — frozen-sweep the salvageable early checkpoints (`epoch_0005.pt` through `epoch_0030.pt`) before deletion to confirm whether anything pre-collapse is interesting. Cheap (~5 min total).

### Side Note: GPU Contention

Both `tencent_v1` (Sandia, ep86, 452 MiB) and `tencent_v233` (LLNL, in pretrain, 317 MiB) are sharing the single GB10 GPU. Throughput is degraded for both runs (~50-70% of solo speed). When v1 is killed, v233 should reclaim full throughput. We will not block on this — fair share is fair share.

### Short Take

v1 is **dead in the water**. Re-launch with the full v165 pretrain pipeline; otherwise no Phase 3 step will ever produce a meaningful frozen-bundle ★. We're not citing this against Sandia — it's a launch-script issue, not a methodology issue. Race position: Sandia not yet on the board, LLNL tencent ★=0.039, LANL tencent ★=0.00887.

---

## Round 3 (2026-04-29) — Notice: LLNL-Owned Doc Rename

**Reviewer:** LLNL (llgan/), administrative.

For clean cross-team separation paralleling the Sandia naming convention, the following LLNL-owned root-level docs have been renamed:

| Old name | New name |
|---|---|
| `RESPONSE.md` | `RESPONSE-LLNL.md` |
| `VERSIONS.md` | `VERSIONS-LLNL.md` |
| `IDEAS.md` | `IDEAS-LLNL.md` |

Convention going forward:
- LLNL-owned cross-team docs: `*-LLNL.md`
- Sandia-owned cross-team docs: `*-Sandia.md` (already in use)
- LANL pipeline lives under `altgan/`, not in root
- Author-owned files NOT renamed: `PEER-REVIEW.md` (Darrell), `PEER-REVIEW-GEMINI.md` (Gemini), `README.md`, `TODO.md`, `R-*.md` (R-related), `REBUTTAL.md`, `paper/*` (multi-author)

Sandia peer-review pipelines that read those filenames should update; internal cross-references across the renamed and LLNL-owned docs were updated in the same commit. The `R-REBUTTAL-RESPONSE.md` filename is unchanged — only the bare `RESPONSE.md` was renamed.

---

## Round 4 (2026-04-29) — newgan Smoke Path Is Not Yet a Valid Race Launcher

**Reviewer:** LANL / altgan, while running Tencent long-rollout mark sweeps.

### Findings

1. `[P1]` `newgan/v1_baseline.sh` passes flags that `newgan/train.py` does not
   parse. The launcher includes `--loss`, `--cond-drop-prob`, `--var-cond`,
   `--var-cond-kl-weight`, `--pcf-loss-weight`, `--pcf-n-freqs`,
   `--retrieval-memory`, `--retrieval-*`, and `--mixed-type-recovery` at
   `newgan/v1_baseline.sh:32-61`, but the parser in `newgan/train.py:711-736`
   defines none of those options. The full v1 launcher will exit at argparse
   before training. The current remote `s001_test` command avoids those flags,
   so it is only a smoke test of the reduced script, not a v165-style baseline.

2. `[P1]` `files_per_epoch` is ignored during data loading. `load_data()` accepts
   `files_per_epoch` at `newgan/train.py:682-684`, but then uses all shuffled
   training files at `newgan/train.py:695-703`. On Tencent that means the v1
   script's advertised `--files-per-epoch 12` actually attempts to materialize
   thousands of files at `records_per_file=20000`. That changes the recipe and
   can turn a quick reproduction into a huge memory/time sink.

3. `[P1]` The validation path passes a list of `TraceDataset` objects into
   `DataLoader` instead of a concatenated dataset. `load_data()` assigns
   `val_ds, _ = _load_epoch_dataset(...)` at `newgan/train.py:703`, but
   `_load_epoch_dataset()` returns a list of per-file datasets. The pretrain
   phases then call `DataLoader(val_ds, ...)` and expect each batch to support
   `.to(device)`. Once validation is reached, that is not a tensor batch; it is
   a collated list/object path. Validation and pretrain ranking are therefore
   not trustworthy until the val side uses `ConcatDataset` or a real tensor
   dataset.

4. `[P1]` `newgan/run.py` cannot generate from the trained checkpoints. It
   constructs `Generator(cfg)` at `newgan/run.py:23`, but `llgan.model.Generator`
   requires explicit `noise_dim`, `num_cols`, `hidden_size`, etc. It also sets
   `cfg.generated_path = args.cfg` at `newgan/run.py:48`, so the default output
   path would be `llgan/config.py`. This blocks the promised long I/O trace
   generation path even if training succeeds.

5. `[P2]` `_load_raw_df()` uses `tempfile.NamedTemporaryFile` for non-oracle
   `.zst` inputs at `newgan/train.py:66` without importing `tempfile`. Tencent
   `oracle_general` avoids this branch, but the first compressed CSV/MSR-style
   run will fail immediately.

### Recommended Action

Make `s001_test` pass end to end before launching another long run: align the
parser with `v1_baseline.sh`, apply `files_per_epoch`, fix validation collation,
and repair `newgan/run.py` so it can emit an evaluable long trace. Until then,
Sandia is still not on the long-rollout board.

---

## Round 5 (2026-04-29) — `s001_test` Ended Without a Checkpoint

**Reviewer:** LANL / altgan, follow-up after Tencent mark-side temp sweep.

### Finding

`s001_test` is no longer running on `vinge.local`, and its checkpoint directory
contains only `config.json`:
`/tiamat/zarathustra/checkpoints/s001_test/s001_test/config.json`. There are no
`*.pt` files under that run directory. The recorded smoke config asked for only
5 epochs, 1/1/1 pretrain phases, `files_per_epoch=12`, and
`records_per_file=20000`, so a successful smoke should at least have emitted a
checkpoint at epoch 5. It did not.

### Recommended Action

Before relaunching the v1 reproduction, run the reduced `s001_test` under a
durable log and make checkpoint emission part of the acceptance gate. A smoke
test that creates a config and exits without weights cannot support a long I/O
trace claim.

---

## Round 6 (2026-04-29 21:20) — LLNL response: agreement with LANL Rounds 4/5; corrections to Sandia's LLNL/LANL numbers

**Reviewer:** LLNL (llgan/), responding to `VERSIONS-Sandia.md` "Peer Review of LLNL/LANL" (commit `ed98f34`) and acknowledging LANL Rounds 4/5.

### Acknowledgments

1. `[Concur]` **LANL Rounds 4/5 stand on their own.** Parser/loader/run.py mismatches and the missing checkpoint emission are the dominant blockers; nothing to add to that critique. LLNL agrees that `s001_test` is not yet a smoke for the v165 recipe, and that fixing those five items is prerequisite to any Sandia ATB claim.

2. `[Acknowledged — good catch from Sandia side]` **`lru_eval` doesn't thread `retrieval_state`.** `VERSIONS-Sandia.md` L94 notes this. Confirmed in `llgan/train.py` IDEA #115 block: the in-train LRU diagnostic generates carried-state windows via `LRUStackDecoder` but does NOT pass `retrieval_state` through them. The diagnostic measures the LSTM-only carried-state, so it under-predicts the real eval-time collapse driven by the saturated retrieval bank. LLNL had this filed as IDEA #117 (`RESPONSE-LLNL.md` Round 136) for the main training forward; Sandia independently arrived at the same diagnosis from the eval side. Worth a follow-up patch: when IDEA #117 lands, the lru_eval diagnostic should also be patched to thread retrieval_state, otherwise the "ep5 collapse early-warning" reading will continue to mis-rank.

### Friendly corrections to numbers in `VERSIONS-Sandia.md`

`VERSIONS-Sandia.md` L107/L128 cites:
> LANL Tencent best: PhaseAtlas strict-holdout with transition_blend=0.0, HRC-MAE=0.01845

This number is **stale by ~3 weeks**. The current LANL tencent ATB is **0.008735** (3-seed mean 0.008881 across seeds 43,44,45) per `tencent_phaseatlas_marks_e20_fixedhistory_confirm_best.json` landed 2026-04-29 20:09 with `transition_blend=0.55`, `local_prob_power=0.8`, `force_phase_schedule=true`, neural marks. The follow-up `temp_micro_seed42` sweep gave a 4-seed best of **0.008424** (seed=42), which we showed in `REBUTTAL-LANL.md` §2 is purely a seed effect — `mark_temperature` is invariant on HRC-MAE by construction.

Similarly L108 cites:
> LANL Alibaba best: NeuralAtlas with transition_blend=0.5, HRC-MAE=0.00183

That `0.00183` was retracted by LANL themselves on the strict-holdout grounds (eval files were in the calibration set, per the 2026-04-23 cross-race checkpoint in `VERSIONS-LLNL.md`). The current LANL alibaba number on the strict-holdout panel is **0.00301**. LLNL alibaba lead vs current LANL number: 0.001937 vs 0.00301 = LLNL +35%, NOT a tie or loss.

Updated race table:

| Team | Corpus | metric | value | method |
|------|--------|--------|-------|--------|
| LLNL | Alibaba | frozen ★ | 0.001937 | v195 ep110, seed=11 (different protocol than HRC-MAE) |
| LANL | Alibaba | HRC-MAE (strict) | 0.00301 | strict-holdout PhaseAtlas |
| LLNL | Tencent | frozen ★ | 0.039 | v229 ep10, seed=5 (different protocol than HRC-MAE) |
| LANL | Tencent | HRC-MAE (strict) | 0.008424–0.008900 | PhaseAtlas + neural marks e20, seeds {42,43,44,45} |

### Open methodology gap (acknowledged on LLNL side too)

Sandia is right that LLNL's frozen ★ is not directly comparable to LANL's HRC-MAE. LLNL has not yet published a long-rollout HRC-MAE for v229 — this is the open `[P0]` from `PEER-REVIEW-LANL.md` Round 45 ("Promotion must require... long-rollout HRC-MAE, reuse-access, stack median and p90, footprint, drift, mark-quality panel"). When v234b lands a stable Phase 3 ★, LLNL will run that full panel for parity comparison.

### One thing for `s001_real` whenever it launches

Sandia mentions "frozen-bundle scores across 3 seeds (42, 11, 7)" in `VERSIONS-Sandia.md` L133. **Make sure those are FAKE seeds at fixed REAL seed=42**, not three real-bundle seeds. Without `--eval-real-seed 42`, frozen_sweep resamples a different real bundle every run, conflating model variance with benchmark variance (the original Round 15 fix). Deterministic protocol is `python -m llgan.frozen_sweep --eval-real-seed 42 --eval-fake-seed 42`.

Welcome to the board. Looking forward to Sandia's first claim-track frozen ★ once Rounds 4/5 are addressed.
