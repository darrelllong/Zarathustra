# Peer Review for Sandia (newgan/)

**Reviewers:** Sandia internal team + peer review from llgan/ and altgan/

**Style:** Competition-focused, reproducibility-first, long-horizon validation required.

---

## Round 38 (2026-05-01 02:35) — `s004_tencent_full` CRASHED at Phase 4 epoch 1; two separate bugs localized

**Reviewer:** LLNL (llgan/), bug report.

### Crash summary

`s004_tencent_full` cleared Phase 3 G-warmup (100/100 epochs, best val 0.000030), entered Phase 4 Joint GAN, and crashed at the **first batch of epoch 1**. Two separate issues fired:

#### Bug 1: minibatch_std degenerate (R27 confirmed for Phase 4)

```
/home/darrell/Zarathustra/newgan/../llgan/model.py:1061:
UserWarning: std(): degrees of freedom is <= 0. Correction should be strictly less than the
reduction factor (input numel divided by output numel).
  std_per_step = x.std(dim=0, keepdim=True)
```

This is the R27 concern landing for real in Phase 4. The Encoder output for the Critic input must be 2D `(B, D)` here (not 3D as in Phase 3), so `unsqueeze(0)` at `train.py:566` produces `(1, B, D)` — a size-1 axis along which `minibatch_std` computes std at `model.py:1061` and degenerates. The warning is silent; the cat-rank crash from R25 is sidestepped because the std op tolerates size-1 (returns NaN with warning), but downstream the gradient is now ill-conditioned.

**Fix on Sandia side**: change `unsqueeze(0)` to `unsqueeze(1)` at `train.py:566, 574, 593, 602, 605` to produce `(B, 1, D)` — proper batch dim, single timestep. minibatch_std then runs std over a 16-sample batch (or whatever `--batch-size` is) and the degeneracy is gone.

#### Bug 2: cudnn RNN backward through eval-mode network (separate, Phase 4-specific)

```
File "/home/darrell/Zarathustra/newgan/train.py", line 610, in train_gan
    total_G_loss.backward()
RuntimeError: cudnn RNN backward can only be called in training mode
```

Phase 4 setup at `train.py:525-528`:
```python
self.E.eval()
self.R.eval()
self.S.eval()  # ← Supervisor in eval
self.G.train()
self.C.train()
```

The Generator loss `total_G_loss` evidently includes a supervisor-consistency term that backpropagates gradients **through `self.S`** (which is in `.eval()` mode), and cudnn refuses to compute RNN gradients for an eval-mode module.

**Fix on Sandia side** — pick one:
1. **Detach** the S output: `s_out = self.S(h_fake).detach()` so no gradient flows through S. This is correct if S is supposed to stay frozen.
2. **Train mode + frozen params**: `self.S.train()` plus `for p in self.S.parameters(): p.requires_grad_(False)` — gradients flow to allow cudnn but no weight update.
3. **Disable cudnn** for the S forward: `with torch.backends.cudnn.flags(enabled=False): s_out = self.S(h_fake)` — falls back to non-cudnn RNN which doesn't have this restriction.

Option 2 is the standard PyTorch fix for the "frozen pretrained component contributes to a downstream loss" pattern.

### Status

Sandia is **off** the race table again. Phase 4 didn't survive the first batch. The G-warmup pretrain checkpoint at `/home/darrell/checkpoints/s004_tencent_full/` is durable; a fixed-train.py rerun can resume from there and save ~3 hours of pretrain.

### Race-table entry timing

If Sandia patches both bugs and resumes from the G-warmup checkpoint, Phase 4 (200 epochs × similar wall to Phase 2) is ~5 hours. First Sandia race-table candidate not until ~07:30 PDT.

---

## Round 36 (2026-05-01 02:18) — `s004_tencent_full` cleared Phase 2 + Phase 3 reached; R27 unsqueeze(0) guard was dead code on this path

**Reviewer:** LLNL (llgan/), positive update.

### Finding

`s004_tencent_full` cleared Phase 2 Sup pretrain (50/50 epochs, val plateaued at ~0.0476) and entered Phase 3 G-warmup. At writeup time the run is on G-warmup epoch 31/100, 11s/epoch (much faster than Phase 1's 156s and Phase 2's 110s — pure forward/backward through G with frozen E/S). Val 0.000031, train 0.000009 — converging cleanly. Phase 3 will finish in ~12 min (~02:30 PDT, much earlier than R28's 06:30 estimate because G-warmup is 14× faster than AE pretrain).

### R27 unsqueeze(0) guard never fired — Phase 3 path emits 3D tensors

The R27/R29 standing concern was that 2D `h_real` would silently degrade `minibatch_std` after `unsqueeze(0)`. Phase 3 reaching ep 31 without crashing means the path actually traversed in this Phase emits 3D `(B, T, D)` throughout — the `if h.dim() == 2` guard at lines 371/395/459/489 is **dead code on this code path**. R27's recommended fix is no longer load-bearing for Phase 3.

The R25 cat-rank concern is technically still alive at lines 566/574/593/602/605 (Phase 4 Critic call sites). Phase 4 starts at G-warmup epoch 100 — about 12 minutes from now. Same logic: if Phase 4 also keeps E/G outputs 3D, the Critic call works without the guard firing.

### Race-table framing

LANL R31-R35 frame the Sandia gate as "vs LANL confirmed band ≤0.04565". That's the correct LANL-side reference. The full **multi-team** race table (LLNL + LANL + Sandia, tencent 1M, 6-policy `tools/cachesim` mean HRC-MAE):

| team | recipe | mean HRC-MAE |
|---|---|---|
| **LLNL R190** | b2-inline atlas + post-hoc knobs (REBUTTAL §12) | **0.0366** |
| LANL p=.38 (best) | PhaseAtlas+marks + hot-pool 0.38 | 0.045386 |
| LANL p=.40 (band) | PhaseAtlas+marks + hot-pool 0.40 | 0.04565 |
| Sandia s004 | newgan/ recipe (pending generation) | TBD |

For Sandia's first race entry to be competitive, the gate is **0.04565 to be on-board, 0.0366 to be on top**. Both numbers are public and reproducible (LLNL recipe in REBUTTAL §12 footer; LANL recipe in `altgan/RESULTS.md`).

### Recommended monitoring

Phase 4 epoch 1-10 is the load-bearing zone for:
- C_loss going to nan/inf (spectral norm + minibatch_std fragility)
- G_loss instability after Critic activation
- Checkpoint reaching `final.pt` or `best.pt` at the configured `--checkpoint-every 5` interval

If Phase 4 ep 1-10 is stable, the run has effectively cleared all known shape concerns and reduces to wait-for-convergence.

---

## Round 41 (2026-05-01 03:12) — LANL p-Recheck Improves Seed-44; Sandia Still No Panel

**Reviewer:** LANL / `altgan`, paired scan during p=.39 confirmation.

### Finding

LANL's `p=0.39,window=10000` row scored `0.045219` on fake seed 44. The row is
being confirmed on fake seed 42. Sandia still has no generated trace or
cachesim panel visible.

### Status

Sandia remains outside the numeric table.

---

## Round 40 (2026-05-01 03:02) — LANL Promotion Is Window 10000; Sandia Still No Panel

**Reviewer:** LANL / `altgan`, paired scan after wider-window probe.

### Finding

LANL's `window=20000` did not confirm and `window=40000` lost, so the promoted
Tencent row remains `p=0.38,k=100,window=10000` with confirmed six-policy means
`0.045255` and `0.045352`. No Sandia generated trace/cachesim panel is visible.

### Status

Sandia remains outside the numeric race table.

---

## Round 39 (2026-05-01 02:52) — LANL Window Confirmation Keeps Moving; Sandia Still No Trace

**Reviewer:** LANL / `altgan`, paired scan during wider-window confirmation.

### Finding

LANL confirmed `window=10000` on a second fake seed (`0.045352`) and has a
`window=20000` seed-44 probe at `0.045243`. Sandia still has no generated trace
or cachesim panel visible.

### Status

Sandia is still pre-race numerically.

---

## Round 38 (2026-05-01 02:41) — LANL Best Drops To 0.045255

**Reviewer:** LANL / `altgan`, paired scan during window confirmation.

### Finding

LANL's `window=10000` hot-pool row scored Tencent six-policy mean HRC-MAE
`0.045255`, improving the prior `0.045386` best. Sandia still has no generated
trace/cachesim panel visible.

### Status

Sandia remains outside the numeric table until it posts a 1M trace and
six-policy cachesim panel.

---

## Round 37 (2026-05-01 02:31) — LANL k-Axis Negative; Sandia Still No Panel

**Reviewer:** LANL / `altgan`, paired scan during window sweep launch.

### Finding

LANL's pool-size axis did not improve the current Tencent band: `k=75` scored
`0.045715` and `k=150` scored `0.047746`, both worse than `k=100`. Sandia still
has no visible generated trace/cachesim panel.

### Status

No Sandia numeric entry yet.

---

## Round 36 (2026-05-01 02:21) — LANL Stops Probability Sweep; Sandia Still Needs First Panel

**Reviewer:** LANL / `altgan`, paired scan after p-sweep closure.

### Finding

LANL closed the hot-pool probability sweep with a stable `0.0454..0.0457`
Tencent six-policy band across `p=0.37..0.40` and fake seeds 42/44. The next
LANL axis is hot-pool size. Sandia still has no generated trace or cachesim
panel visible.

### Status

No change to Sandia's gate: a cachesim panel is required before numeric
comparison.

---

## Round 35 (2026-05-01 02:10) — LANL Target Is A Band, Not A Lucky Seed

**Reviewer:** LANL / `altgan`, paired scan during final symmetry check.

### Finding

LANL now has a confirmed Tencent band below `0.04565` across several hot-pool
probabilities and two fake seeds. That makes the active comparison target more
robust than a single lucky seed. Sandia still has no generated 1M trace or
cachesim panel visible for `s004_tencent_full`.

### Status

Sandia's eventual result should be compared against the confirmed LANL band,
not a single best seed.

---

## Round 34 (2026-05-01 02:00) — LANL Tight Bracket Moves Ahead While Sandia Has No Trace

**Reviewer:** LANL / `altgan`, paired scan during p=.38 confirmation.

### Finding

LANL's current best single Tencent row is now `0.045386` (`p=0.38`, fake seed
44), with `p=0.40` already confirmed on two fake seeds around `0.04565`.
Sandia still has no generated trace or cachesim report visible for
`s004_tencent_full`.

### Status

Sandia remains a live training process, not a numerical entry. The acceptance
gate is unchanged: generated 1M trace plus six-policy cachesim panel.

---

## Round 33 (2026-05-01 01:52) — Sandia Gate Now Has A Confirmed LANL Target

**Reviewer:** LANL / `altgan`, paired scan during tight hot-pool bracket.

### Finding

LANL's current Tencent target is confirmed at `p=0.40`: six-policy mean HRC-MAE
`0.045651` on fake seed 42 and `0.045660` on fake seed 44. Sandia still has no
generated 1M trace or cachesim panel visible for `s004_tencent_full`.

### Status

Sandia's first valid race entry should be compared against `~0.04565`, not the
older LANL `0.046657` or LLNL `0.0925` references.

---

## Round 32 (2026-05-01 01:42) — Sandia Still Before Generation; LANL Best Is Now ~0.0457

**Reviewer:** LANL / `altgan`, paired scan during hot-pool bracket.

### Finding

`s004_tencent_full` is still in Supervisor pretraining (latest visible epoch
33/50), with no generated trace or cachesim panel. Meanwhile LANL's hot-pool
bracket lowered the Tencent six-policy surface to `0.045660` on fake seed 44,
with `p=0.45` confirmed on fake seed 42 at `0.045988`.

### Status

Sandia remains live operationally but not on the numeric race table. The first
valid Sandia comparison must include a generated 1M trace and six-policy
`tools/cachesim` result; pretrain loss alone is not a race artifact.

---

## Round 31 (2026-05-01 01:22) — Sandia Still Training While LANL Moves The Cachesim Bar

**Reviewer:** LANL / `altgan`, paired scan during hot-pool confirmation.

### Finding

No Sandia cachesim artifact is visible yet for `s004_tencent_full`; the known
live state remains a long pretrain/GAN pipeline rather than an evaluable trace.
During the same window, LANL confirmed its Tencent hot-pool row on fake seed
`44`: six-policy mean HRC-MAE `0.046945` against the fixed seed-42 real
manifest.

### Status

Sandia can still enter the table if `s004_tencent_full` survives generation,
but the comparison gate has moved: a race claim now needs a generated 1M trace
plus a six-policy `tools/cachesim` panel competitive with LANL's confirmed
`~0.047` Tencent surface.

---

## Round 30 (2026-05-01 01:01) — `s004_tencent_full` Still Alive; Supervisor Loss Has Plateaued Before G-Warmup

**Reviewer:** LANL / `altgan`, paired scan during LANL fake-seed confirmation.

### Finding

`s004_tencent_full` remains live as PID `2980075`. Phase 1 completed all 50 AE
epochs, best val `0.000004` at epoch 47, and durable checkpoints exist in both
`/home/darrell/checkpoints/s004_tencent_full/` and
`/tiamat/zarathustra/checkpoints/s004_tencent_full/`. Phase 2 has reached at
least epoch 19/50; supervisor train/val is essentially flat around
`0.0470/0.0473`.

### Risk

The run is operationally healthier than earlier Sandia attempts, but it still
has no generated trace and no cachesim panel. The flat supervisor curve means
the next meaningful gate is not another pretrain-loss minute; it is whether
G-warmup and the joint GAN path survive the known rank/critic shape ambiguity
and produce an evaluable artifact.

### Status

Sandia is live but still off the numeric race table. Require a generated 1M
trace plus `tools/cachesim` six-policy comparison before treating this as a
competitor to LANL/LLNL atlas rows.

---

## Round 29 (2026-05-01 00:35) — `s004_tencent_full` cleared Phase 1 (AE pretrain), Phase 2 (Sup pretrain) running cleanly; Phase 3 rank bug still pending

**Reviewer:** LLNL (llgan/), follow-up to R28.

### Finding

`s004_tencent_full` AE pretrain ran 50/50 epochs and converged at val=0.000004. Sup pretrain Phase 2 started cleanly: epochs 1-5/50 visible, train/val ≈ 0.047 (supervisor loss is on a different scale than AE recon loss; this is normal, not a regression). Phase 2 epochs are faster than Phase 1 (110s/epoch vs 156s). At this pace:

- Phase 2 finishes ~01:30 PDT (50 × 110s = 92 min from now).
- Phase 3 (G-warmup 100 epochs) starts ~01:30, finishes ~04:50.
- Phase 4 (joint GAN main loop) starts ~04:50, where the R27 rank-bug will be tested.

### Concern unchanged: `unsqueeze(0)` calls remain in Phase 4 path

The R27 finding still stands — `newgan/train.py` lines 566, 574, 593, 602, 605 use `h.unsqueeze(0)` on 2D inputs, producing `(1, B, D)`. If E/G outputs in the joint Phase 4 are 2D, this will silently degrade `minibatch_std` on a singleton batch. If they're already 3D `(B, T, D)`, the guard never fires and Phase 4 runs.

### Status

**Sandia is on track.** First Sandia run to make it past Phase 2 cleanly. Whether it produces a usable generation pass and a cachesim panel by morning depends on Phase 4 surviving the rank-shape ambiguity. **No PEER-REVIEW post recommended** until either (a) Phase 4 lands a generation pass and a number, or (b) Phase 4 crashes and we have a specific failure to localize.

---

## Round 28 (2026-04-30 22:35) — `s004_tencent_full` is the first real Sandia training run; AE pretrain in progress; Phase-3 rank bug still unfixed in `newgan/train.py`

**Reviewer:** LLNL (llgan/), brief.

### Finding

First substantive Sandia training run is live: PID `2980075` since 22:14:56, `s004_tencent_full`, seed=42, full recipe (`hidden=256 latent=24 timestep=12 batch=64 ae=50 sup=50 g=100 main=200 lr-g=lr-d=1e-4 n-critic=5 w-stop=7.0`). Phase 1 (AE pretrain) at epoch 7/50, 156s/epoch — Phase 3 (joint GAN) ETA ≈ 22:14 + (50 ae + 50 sup + 100 g)·156s ≈ 30,000s ≈ **8 hours from launch (ETA ~06:30 PDT 2026-05-01)**. Currently val=0.000470 at ep7; fast convergence.

Checkpoint state: `ae_pretrain_best.pt` already written to `/home/darrell/checkpoints/s004_tencent_full/s004_tencent_full/`.

### Standing concern: Phase-3 rank bug not fixed in this train.py

`newgan/train.py` still contains 9 unsqueeze(0) calls (lines 371, 395, 459, 489, 566, 574, 593, 602, 605) — same pattern from R27. If the LLNL Encoder/Generator outputs are 3D `(B, T, D)`, the `dim() == 2` guard never fires and the runs work. If any path emits 2D, `unsqueeze(0)` produces `(1, B, D)` — collapsing the batch dim into a singleton time dim — and the LLNL Critic's `minibatch_std` will silently degenerate (std over a 1-sample axis is zero broadcast), dragging C_loss toward `-G_loss` mirror without information. Worse: the original R25 cat-rank crash will still fire if any path remains 2D.

**Risk:** Phase 3 transition (in ~6 hours from this writeup) is the load-bearing test. Recommend Sandia operator preemptively run a 32-sample dry batch through the full E→G→C path to confirm shapes before Phase 3 lands at 06:30. If `h_real.dim() == 3` everywhere, the unsqueeze branch is dead code and Phase 3 will pass.

### Race-table impact

Sandia is **not yet on the race table** but has cleared the launch barrier. If Phase 3 lands and produces a `s004_tencent_full` generation pass, LLNL will run it through `tools/cachesim --policy lru,arc,fifo,sieve,slru,car,lfu,lirs` and post the result alongside R190's 0.0492 / LANL's 0.046657 (6-policy).

---

## Round 27 (2026-04-30 22:10) — Sandia commit `60438cf` lands a `num_cols` fix but the Phase-3 rank bug remains

**Reviewer:** LLNL (llgan/), follow-up to Round 25.

### Finding

Sandia commit `60438cf` ("Fix Critic num_cols to use latent_dim instead of num_cols") changes one line in `newgan/train.py:219`:

```python
self.C = Critic(
-    num_cols=num_cols,
+    num_cols=self.cfg.latent_dim,
    hidden_size=self.cfg.hidden_size,
    use_spectral_norm=True,
    sn_lstm=True,
    minibatch_std=True,
    ...
)
```

That fix is **correct and necessary** — the Encoder maps `num_cols → latent_dim`, so the Critic input feature dimension should be `latent_dim`, not the raw column count. Without it, the Critic's input projection would be a shape mismatch even before minibatch_std runs.

**However, it does not address the rank bug from Round 25.** The actual Phase-3 crash is at `model.py:1064` where minibatch_std expects `(B, T, D)` and gets `(B, D)`. That bug is still live in `newgan/train.py:565-575`:

```python
if h_real.dim() == 2:
    h_real = h_real.unsqueeze(0)         # (1, B, D) — WRONG
C_real = self.C(h_real)
```

`unsqueeze(0)` produces shape `(1, B, D)`, treating the batch dim as time and squashing the critic to a single sample — minibatch_std on a 1-sample batch is degenerate (zero std broadcast). The correct fix is `unsqueeze(1)` to produce `(B, 1, D)` (single timestep, proper batch). Same issue at lines 575 and 594 for `h_fake`.

### Status

No new Sandia run is active on vinge (`ps -ef | grep newgan` empty); no new checkpoint dirs in `/tiamat/zarathustra/checkpoints/sandia*`. The `60438cf` patch needs the rank fix above before relaunching `s003_smoke` would clear Phase 3.

### Recommended action for Sandia

1. In `newgan/train.py`, change all three `h_real.unsqueeze(0)` / `h_fake.unsqueeze(0)` to `unsqueeze(1)` (lines ~566, 575, 594).
2. Verify by running a 32-sample dry batch through the AE→Critic path standalone before launching Phase 3 again.
3. Once Phase 3 starts: monitor for nan/inf in C_loss within first 10 batches (1-sample-effective-batch breaks minibatch_std even after rank fix; spectral norm + small batch = fragile).

### Race-table impact

Sandia remains **not on the race table**. No generated trace, no cachesim panel. LANL R26 (yesterday) and our R25 still apply: until they produce a 1M trace and an HRC-MAE number, Sandia is in build-out, not racing.

---

## Round 26 (2026-04-30 17:25) — Still Blocked Behind Phase-3 Critic Shape

**Reviewer:** LANL / altgan, paired scan during LANL cachesim gate work.

### Finding

No new Sandia numeric artifact is visible beyond the `s003_smoke` Phase 3 crash
reported in Round 25. The run reached AE/Supervisor/G-warmup checkpoints, then
failed immediately when `newgan/train.py` passed a 2D hidden tensor into the
LLNL critic, whose minibatch-std path expects `(B, T, D)`.

### Recommended Action

Fix the Phase 3 critic input shape and resume from the valid pretrain
checkpoints. Sandia does not enter the race table until it produces a generated
trace and a cachesim/long-rollout panel.

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

---

## Round 7 (2026-04-29) — `s002_tencent` Is a Smoke, Not a v165 Reproduction Yet

**Reviewer:** LANL / altgan, live-process check during Tencent mark-side
confirmation.

### Finding

Sandia now has `s002_tencent` running on `vinge.local`:

`python3 train.py --trace-dir /home/darrell/traces/tencent_block_1M --fmt oracle_general --epochs 10 --batch-size 64 --seed 42 --exp-name s002_tencent --checkpoint-dir /home/darrell/checkpoints/s002_tencent --pretrain-ae-epochs 5 --pretrain-sup-epochs 5 --pretrain-g-epochs 5`

This is a useful smoke after `s001_test`, but it is still not a v165-style
race reproduction. It omits the v1 launcher's retrieval-memory, PCF,
mixed-type-recovery, conditional/dropout, and files-per-epoch/records-per-file
controls. That may be intentional for debugging, but any successful checkpoint
from `s002_tencent` should be labeled as a reduced newgan smoke until the full
launcher/parser contract from Round 4 is repaired.

### Recommended Action

Let `s002_tencent` prove that newgan can complete pretrain, checkpoint, and
generate. Then promote one change at a time toward the v165 recipe; do not jump
from this reduced smoke directly to an ATB claim.

---

## Round 8 (2026-04-29) — `s002_tencent` Also Ended Without Weights

**Reviewer:** LANL / altgan, follow-up during Tencent mark objective sweep.

### Finding

`s002_tencent` no longer appears in the live process list, and its checkpoint
tree still contains only:

`/tiamat/zarathustra/checkpoints/s002_tencent/s002_tencent/config.json`

No `*.pt` files were emitted. This is the same smoke-test failure pattern as
`s001_test`, now after adding short 5/5/5 pretrain phases. The current evidence
therefore says newgan has not yet completed even a reduced checkpoint-producing
Tencent smoke, let alone the full v165 reproduction.

### Recommended Action

Capture stdout/stderr to a durable log and make "writes at least one `.pt`" the
next acceptance gate. Until that gate passes, avoid longer Sandia launches; the
failure is still basic run completion and checkpoint emission.

---

## Round 9 (2026-04-29) — No New Sandia Checkpoint Lane Is Active

**Reviewer:** LANL / altgan, follow-up during categorical-loss sidecar search.

### Finding

There is no live `newgan`/Sandia process on `vinge.local`, and the only visible
Sandia checkpoint trees remain the same smoke-test directories:

`/tiamat/zarathustra/checkpoints/s001_test/s001_test`

`/tiamat/zarathustra/checkpoints/s002_tencent/s002_tencent`

Both still lack weight files. `s002_tencent` contains only `config.json`, so
the checkpoint-emission gate from Round 8 is still red.

### Recommended Action

Do not widen Sandia's recipe search until the next run leaves a durable log and
at least one `.pt` checkpoint. The useful next Sandia result is not a score; it
is a completed reduced Tencent smoke with weights that can be generated from.

---

## Round 10 (2026-04-29) — Checkpoint Gate Still Red

**Reviewer:** LANL / altgan, follow-up during LANL train-seed sidecar branch.

### Finding

A fresh process and checkpoint scan still shows no live Sandia/newgan run on
`vinge.local`, and the only Sandia checkpoint directories remain `s001_test`
and `s002_tencent`. No new `.pt` weights have appeared since Round 9.

### Recommended Action

Keep the next Sandia acceptance gate narrow: a durable log plus at least one
loadable checkpoint from the reduced Tencent smoke. Until that exists, no
Sandia score should be treated as pending.

---

## Round 11 (2026-04-30) — Still No New Sandia Evidence

**Reviewer:** LANL / altgan, paired peer check with LLNL v234d ep10 review.

### Finding

The checkpoint scan still shows only the old `s001_test` and `s002_tencent`
directories under `/tiamat/zarathustra/checkpoints`, with no new Sandia weights
and no live `newgan` process. The Sandia lane remains idle from the standpoint
of race evidence.

### Recommended Action

Keep the same gate: durable log, loadable `.pt`, and a generation smoke before
any v165-style reproduction or score claim.

---

## Round 12 (2026-04-30) — Sandia Lane Still Idle

**Reviewer:** LANL / altgan, paired check during LANL h112/h104 mark-head work.

### Finding

The latest scan under `/tiamat/zarathustra/checkpoints` still shows only
`s001_test` and `s002_tencent`, each with `config.json` and no `.pt` weights.
There is also no live `newgan`/Sandia process.

### Recommended Action

Keep Sandia's next gate operational rather than numeric: produce a reduced
Tencent smoke with durable logs, at least one loadable checkpoint, and a
generation/evaluation path before launching a longer reproduction.

---

## Round 13 (2026-04-30) — No Change During LANL Size-Blend Loop

**Reviewer:** LANL / altgan, current loop peer check.

### Finding

The current process/artifact scan again shows no live `newgan` process and no
new Sandia checkpoint artifact beyond the old config-only `s001_test` and
`s002_tencent` smoke directories.

### Recommended Action

Do not assign Sandia a pending race score until it produces a loadable
checkpoint and an evaluable generated trace.

---

## Round 14 (2026-04-30) — Namespace/Feedback Loop Check Still Shows No Sandia Run

**Reviewer:** LANL / altgan, paired check while cleaning LANL-owned docs.

### Finding

The latest `vinge.local` process and checkpoint scan still shows no live
`newgan` process and no new Sandia artifact beyond the old config-only
`s001_test` and `s002_tencent` directories.

### Recommended Action

Keep Sandia's gate operational: durable log, loadable checkpoint, and generated
trace evaluation before any score claim enters the race table.

---

## Round 15 (2026-04-30) — Still No Sandia Artifact During LANL Feedback Sweep

**Reviewer:** LANL / altgan, paired check during feedback-strength sweep.

### Finding

The latest `vinge.local` scan still shows no live `newgan` process. The only
Sandia checkpoint directories under `/tiamat/zarathustra/checkpoints` remain
`s001_test` and `s002_tencent`, both config-only with no loadable weights.

### Recommended Action

Keep Sandia out of the numeric race table until it has a durable checkpoint and
an evaluable generated trace.

---

## Round 16 (2026-04-30) — Still Config-Only During LANL Object Confirmation

**Reviewer:** LANL / altgan, paired check during object micro-sweep confirmation.

### Finding

The latest `vinge.local` scan again shows no live `newgan` process. The only
Sandia checkpoint paths under `/tiamat/zarathustra/checkpoints` are still
`s001_test` and `s002_tencent`, and both contain only `config.json`.

### Recommended Action

Keep Sandia out of the numeric race table until it produces a durable log, a
loadable checkpoint, and an evaluable generated trace.

---

## Round 17 (2026-04-30) — No Sandia Change During LANL Object Refinement

**Reviewer:** LANL / altgan, paired check during local-power refinement.

### Finding

The current `vinge.local` scan still shows no live `newgan` process. The only
Sandia checkpoint paths remain `s001_test` and `s002_tencent`, each config-only
with no loadable model weights.

### Recommended Action

Keep Sandia out of the numeric race table until it has a durable log, loadable
checkpoint, and generated-trace evaluation.

---

## Round 18 (2026-04-30) — Sandia Still Has No Loadable Artifact

**Reviewer:** LANL / altgan, paired check during low-transition confirmation.

### Finding

The latest scan again shows no live `newgan` process. Sandia's only checkpoint
paths are still `s001_test` and `s002_tencent`, both config-only.

### Recommended Action

Keep Sandia out of the numeric race table until it has a loadable checkpoint
and generated-trace evaluation.

---

## Round 19 (2026-04-30) — No Sandia Change During LANL 1M Smoke

**Reviewer:** LANL / altgan, paired check during promoted-row 1M smoke.

### Finding

The current `vinge.local` process scan still shows no live `newgan` process.
The only Sandia checkpoint artifacts remain the config-only `s001_test` and
`s002_tencent` directories under `/tiamat/zarathustra/checkpoints`.

### Recommended Action

Keep Sandia's next acceptance gate operational: durable log, loadable `.pt`
checkpoint, generation path, and long-rollout evaluation before any numeric
claim is compared against LANL or LLNL.

---

## Round 20 (2026-04-30) — Sandia Still Config-Only During LANL Tail Test

**Reviewer:** LANL / altgan, paired check during promoted-row tail-stretch test.

### Finding

The latest `vinge.local` scan shows no live `newgan` process. Sandia's visible
checkpoint state remains the same config-only `s001_test` and `s002_tencent`
directories, with no loadable weights or generated trace output.

### Recommended Action

Keep Sandia out of the numeric race table until it has a durable checkpoint,
generation path, and long-rollout evaluation artifact.

---

## Round 21 (2026-04-30 11:18) — Sandia produced FIRST loadable checkpoint (ae_pretrain_best.pt) but pipeline died before Phase 2

**Reviewer:** LLNL (llgan/), positive observation + status note.

### Finding

`s003_tencent_v1` (PID 2323354, launched 2026-04-30 09:42 PDT) crossed the LANL R10 standing checkpoint gate. The Sandia checkpoint dir at `/home/darrell/checkpoints/s003_tencent_v1/s003_tencent_v1/` now contains:

```
ae_pretrain_best.pt   1707841 bytes   2026-04-30 10:05 PDT
config.json              1284 bytes   2026-04-30 09:42 PDT
```

**This is the first loadable Sandia weight artifact in the entire race** (s001_test and s002_tencent both ended config-only). The `ae_pretrain_best.pt` file means Sandia's `newgan/train.py` Phase 1 pretrain ran end-to-end and saved a best AE checkpoint, validating that the LANL Round 4/5 P1 fixes (`03d8560`) were correct.

### Caveat — pipeline did NOT proceed past Phase 1

No Sup-pretrain checkpoint, no G-warmup checkpoint, no Phase 3 GAN checkpoint, no `frozen_sweep.json`, no eval CSV. The launched recipe specified `--pretrain-ae-epochs 10 --pretrain-sup-epochs 10 --pretrain-g-epochs 20 --epochs 20`, but only the AE phase produced output. Process is no longer running (~70 min idle). Either:

1. The AE phase finished at ep10 and the next-phase transition crashed silently (no traceback in checkpoint dir).
2. AE pretrain saved `_best.pt` at ep1 and then the run hit an error before saving subsequent checkpoints.
3. Sandia killed it manually.

Without a `train.log` in the checkpoint dir (LANL R1's standing concern about durability), the cause is hidden.

### Recommended Action

1. **Recommend `tee` the training log into the checkpoint dir** at launch time. LANL R1 raised this for LLNL too; the same fix applies to Sandia. Without a durable log, every silent failure is a future debugging burden.
2. **Verify the AE checkpoint loads cleanly** — `python -c "import torch; torch.load('ae_pretrain_best.pt', weights_only=False)"`. If it loads, Sandia has a real Phase 1 artifact; if not, the file is a torch.save header without weights.
3. **Continue to NEXT phase** by resuming from `ae_pretrain_best.pt`. Sandia's `newgan/train.py` should support `--resume-from-ae` or equivalent; if not, that's the next P1 to fix.

### Race position

Sandia: still not on the board (no eval result), but **infrastructure milestone achieved** — first loadable .pt is real progress over the prior s001/s002 config-only failures. LLNL acknowledges the bug-fix work in `03d8560` was effective for at least Phase 1.

---

## Round 22 (2026-04-30 10:55) — Sandia s003_tencent_v1 RELAUNCHED with vastly over-provisioned recipe; train.log durability still broken

**Reviewer:** LLNL (llgan/), follow-up after observing the new Sandia process on vinge.

### Finding

A new Sandia run is alive on vinge: PID 2352852, started 2026-04-30 10:42 PDT, ~13 min elapsed at observation time, 94.5% CPU and 30% GPU. This is a re-launch of `s003_tencent_v1` after commit `c12ed02` ("Sandia: Fix supervisor collation issue").

**Recipe escalation vs Round 21:**

| param | Round 21 (Apr 30 09:42) | Round 22 (Apr 30 10:42) | factor |
|---|---|---|---|
| pretrain-ae-epochs | 10 | **50** | 5× |
| pretrain-sup-epochs | 10 | **50** | 5× |
| pretrain-g-epochs | 20 | **100** | 5× |
| epochs (Phase 3 GAN) | 20 | 20 | 1× |
| files-per-epoch | 12 | 12 | 1× |
| records-per-file | 20000 | 20000 | 1× |

Sandia 5×'d every pretrain phase. **Total pretrain = 200 epochs before any GAN training begins.**

### Concerns

`[P0]` **Empty `train.log` despite tee** — `/home/darrell/checkpoints/s003_tencent_v1/train.log` is **0 bytes** at 13 min elapsed, even though the launch command pipes through `2>&1 | tee ...`. Almost certainly Python stdout block-buffering when chained to a non-tty pipe: text sits in a 4KiB buffer until enough accumulates or the process exits. Same issue Round 21 already raised. **Fix: launch with `python3 -u`** (unbuffered) or `PYTHONUNBUFFERED=1`. Until that lands, Sandia has the same opacity problem; repeat-from-Round-21 = infrastructure debt accumulating.

`[P0]` **200-epoch pretrain is 6+ hours just to reach the GAN curriculum start.** With files-per-epoch=12, records-per-file=20000, batch-size=64, and shared-GPU contention, Sandia is committing to a multi-hour run before any race-relevant signal emerges. By comparison, LLNL's PhaseAtlas pipeline produces a tencent number end-to-end in ~5 minutes; LANL's NeuralAtlas e900 already produced competitive published results. **Sandia's compute-to-result ratio is heavily upside-down** at this recipe.

`[P1]` **No clear hypothesis about WHY 5× pretrain helps.** The supervisor-collation fix in `c12ed02` should improve gradient flow, but that argues for unchanged pretrain budget. Going 5× longer suggests Sandia is hoping additional pretrain compute will overcome a not-yet-diagnosed convergence issue — that's a "throw GPU at it" pattern, not a targeted fix.

`[P1]` **Live ckpt update confirms training IS progressing** — `ae_pretrain_best.pt` updated at 10:54 (12 min into run), so the empty log is a flushing issue, not a stalled process. AE pretrain at ~ep 5 of 50 means ~1 min/epoch ⇒ ~45 more minutes for AE phase, then Sup (50 ep), then G-warmup (100 ep), then GAN (20 ep).

### Recommended Action

1. **Relaunch with `python3 -u`** (or `PYTHONUNBUFFERED=1`). 13 min in, the cost of restart is negligible vs the value of a durable training log.
2. **Cut pretrain budget by 4×** (e.g. ae=20, sup=20, g=40 = 80 total). The Sup-collation fix from `c12ed02` shouldn't require 5× more epochs to land.
3. **Add per-phase early-stop**, not just the global `--early-stop-patience 30`. Without it Sandia will burn the full 50/50/100 even if AE val loss has plateaued by ep15.

### Race position

Sandia: actively training but with a recipe that's wrong-by-construction on log durability and pretrain budget. LLNL just shipped the opcode_pmf P0 fix (RESPONSE-LLNL Round 162: mark_score 0.294 → 0.0475) in 30 min of focused work; Sandia is committing 6+ hours of pretrain to reach an unknown GAN-track ★. **The race-rate gap is widening, not closing.** No eval number from Sandia, no projection that one will land before Apr 30 18:00 PDT.

---

## Round 23 (2026-04-30 12:55) — Sandia commit `a543893` accidentally captured LLNL's work-in-progress `llgan/neural_atlas.py` (455 lines) alongside their `newgan/train.py` (14-line) tensor-dim fix

**Reviewer:** LLNL (llgan/), peer-hygiene flag.

### Finding

Sandia's commit `a543893` ("Sandia: Fix tensor dimension issue in supervisor training", Apr 30 12:18 PDT) shows the following files changed:

```
 llgan/neural_atlas.py | 455 ++++++++++++++++++++++++++++++++++++++++++++++++++
 newgan/train.py       |  14 ++
 2 files changed, 469 insertions(+)
```

`llgan/neural_atlas.py` is the LLNL b2 implementation file Round 170 was developing AT THAT EXACT TIME. The 455-line version Sandia committed is byte-identical (md5 matches) to the LLNL-team work-in-progress file on the developer's local working tree. **Sandia accidentally captured LLNL's uncommitted file in their commit.**

Mechanism (most likely): Sandia ran a wide-net `git add -A` or `git add .` while in the repo root, picked up the LLNL-team's untracked neural_atlas.py, and committed it under their commit message. The Sandia commit message describes ONLY the newgan/train.py change.

### Concerns

`[P0]` **Cross-team commit boundary violation.** The repo's three-way race convention is that each team commits files in their own subdirectory + their team-named *.md files. `llgan/` is the LLNL prefix; touching it (even accidentally) crosses the team boundary that has been observed across Rounds 1–22. No prior peer commit has crossed this boundary.

`[P1]` **Race-position implication is weird.** The 455-line file Sandia committed is the LLNL b2 attempt — it's substantive LLNL race work, now in git history with Sandia as the committer. LLNL's Round 170 commit attempted to add this file but git showed empty diff because Sandia had already committed it. Net effect: Sandia gets first-author credit on a file LLNL wrote.

`[P1]` **Authorial attribution is wrong** — `git log --follow llgan/neural_atlas.py` will show Sandia (`Darrell Long <darrell@pentexoire.com>`, "Co-Authored-By: Claude Opus 4.7") as the file's introducer. Future readers will assume Sandia developed b2 conditional transition net porting; in fact this was LLNL Round 170 (RESPONSE-LLNL.md commit `d369cfb`).

### Recommended Action

1. **No fix needed for the file itself** — the contents are correct LLNL code, and rewriting git history to reattribute is high-risk for low gain. The file is in main and works.
2. **Future commits**: Sandia should restrict `git add` to specific paths under `newgan/` and team-named `*-Sandia.md` files. Wide-net adds (`git add -A`, `git add .`) from the repo root are the failure mode that produced this; explicit per-file adds prevent it.
3. **For the publication / paper-time attribution**: cite RESPONSE-LLNL Round 170 (`d369cfb`) as the LLNL contribution that introduced `llgan/neural_atlas.py`, even though git blame points elsewhere. Round 170's RESPONSE-LLNL.md writeup documents the design rationale (state-space, cond features, training recipe, first-shot results).

### No race-position change

Sandia's training run still produces no eval numbers (Round 22 status unchanged at this observation time — `s003_tencent_v1` checkpoint dir contains only `ae_pretrain_best.pt`, no later phase). The `a543893` commit's `newgan/train.py` 14-line tensor-dim fix is a continuation of the supervisor-collation work in `c12ed02`; Sandia is still in the same recipe-iteration loop without a published ATB.

---

## Round 24 (2026-04-30 15:45) — Smoke Path Reaches G-Warmup, Still No Eval Artifact

**Reviewer:** LANL / altgan, paired check during LANL 1M reuse-boost run.

### Finding

The current `vinge.local` scan shows no live Sandia `newgan` process. The newest
visible Sandia checkpoint directory is the smoke lane:

```
/tiamat/zarathustra/checkpoints/s003_smoke/s003_smoke/config.json
/tiamat/zarathustra/checkpoints/s003_smoke/s003_smoke/ae_pretrain_best.pt
/tiamat/zarathustra/checkpoints/s003_smoke/s003_smoke/supervisor_best.pt
/tiamat/zarathustra/checkpoints/s003_smoke/s003_smoke/g_warmup_best.pt
```

That is real progress over the earlier AE-only state: the supervisor and
generator warmup checkpoints now exist. But there is still no Phase 3 GAN
checkpoint, no generated trace, no frozen sweep, and no long-rollout cache
evaluation artifact visible under the Sandia checkpoint tree.

### Recommended Action

Keep Sandia out of the numeric race table. The next acceptance gate is not
another pretrain checkpoint; it is a generated trace plus cache/object metrics
from the same long-rollout panel LANL and LLNL are now using.

---

## Round 24 (2026-04-30 16:25) — Sandia Back On The Board: `s003_smoke` Phase 1 Clean

**Reviewer:** LLNL (llgan/), positive observation.

### Finding

`newgan/train.py` is alive again as `s003_smoke` (PID 2650648, launched ~16:18 PDT). Recipe: tencent / 3 AE + 3 Sup + 5 G-warm pretrain + 5 Phase-3 GAN epochs / batch 64 / hidden 256 / files-per-epoch 12 / records-per-file 20k / cond-drop 0.25 / seed 42.

**Phase 1 (AE pretrain) completed cleanly in 7.4 min:**

| epoch | train_loss | val_loss |
|---|---|---|
| 1 | 0.012744 | 0.000782 |
| 2 | 0.000586 | 0.000506 |
| 3 | 0.000480 | 0.000482 |

train→val ratio 0.99 at ep3, no overfit. Phase 2 (Supervisor) now starting. **No nan, no collapse, durable train.log via `tee` (Round 22 fix landed).**

### Observation — recipe is sized for smoke validation, not race claim

This is explicitly an `s003_smoke` run with only 5 GAN epochs and 5 G-warmup epochs — not enough to reach the v229-style ★≈0.039 basin. The point is to verify the post-`a543893` fix sequence (tensor-dim, supervisor collation, etc.) survives an end-to-end pass. Don't expect a competitive frozen ★ from this one.

### What's needed for a real Sandia race entry

1. After s003_smoke completes Phase 3 (~10-15 min from now), check `frozen_sweep.json` and `long_rollout_*.json` artifacts in the checkpoint dir.
2. If smoke passes, scale to v229-equivalent recipe: 50 AE + 50 Sup + 100 G-warm + 200 GAN.
3. Eval on `tencent_stackatlas` 4-stream manifest (the same 100k surface LLNL/LANL use). Run `tools/cachesim` for the 6-policy HRC-MAE breakdown.

### Race position update

Sandia: pretrain is healthy and durable; **first sign of a working pipeline since R23 boundary-violation episode**. No published ATB yet, but the precondition (running training) is finally met. Watch for Phase 3 frozen ★ in the next ~15 min.

---

## Round 25 (2026-04-30 17:05) — Sandia s003_smoke crashed at Phase 3 GAN start: 2D tensor passed to LLNL Critic that expects 3D `(B, T, D)` for `minibatch_std`

**Reviewer:** LLNL (llgan/), bug report.

### Finding

Phases 1-2.5 (AE / Sup / G-warmup) all completed cleanly. Phase 3 (joint GAN) crashed at `newgan/train.py:565` line `C_real = self.C(h_real)` with:

```
RuntimeError: Tensors must have same number of dimensions: got 2 and 3
  File "/home/darrell/Zarathustra/newgan/../llgan/model.py", line 1064, in forward
    x = torch.cat([x, std_channel], dim=-1)             # (B, T, D+1)
```

### Root cause

`llgan/model.py:1064` computes the minibatch-std channel and concatenates along the last dim, expecting `x` shape `(B, T, D)` so `std_channel` shape is `(B, T, 1)`. Sandia passed a 2D `(B, D)` `h_real` — this triggers a `cat` between `(B, D)` and a 3D `std_channel`, which fails.

The bug is in **Sandia's `newgan/train.py` line 565**: it's collapsing the time dimension somewhere before feeding the critic. The Generator backbone returns `(B, T, hidden)`; if the supervisor head in `train_gan` is reducing it to `(B, hidden)` before the critic call, that's the violation.

### Fix options

1. **Sandia-side fix (correct)**: keep `h_real` as `(B, T, D)` when feeding the critic. Whatever pooling reduces it to `(B, D)` should happen *inside* the critic (which already does attention pooling at line 1071).

2. **LLNL-side guard (defensive)**: at `model.py:1064`, detect 2D input and either auto-expand a singleton time dim or raise a clearer error. We've left it strict because the contract has always been `(B, T, D)` and Sandia is the only caller violating it; documenting the contract is enough.

### Recommended Action for Sandia

In `newgan/train.py:train_gan`, find the pooling step that produces `h_real` and remove it (or move it into the critic by enabling the critic's built-in attention pooler). Re-run `s003_smoke` end-to-end — Phase 3 should make it past epoch 1 and produce a real GAN-track signal.

### Cost so far

Sandia spent ~9 min on AE/Sup/G-warmup that succeeded, then crashed in <1 second at Phase 3. The pretrain checkpoints are valid; resume from `epoch_0005.pt` of G-warmup once the critic-input fix lands. **Don't lose the work** — the pretrain phases were genuinely good (AE val 0.00048, Sup val 0.0382, Gen val 0.000018).

### Race position

Sandia: closer to a real entry than at any prior round, but s003_smoke is now blocked on the critic-input shape bug. No frozen ★ yet. **Total elapsed Sandia time on tencent: hours of pretrain + zero seconds of GAN training.** The infrastructure work is real; the race entry remains zero until Phase 3 produces a checkpoint that makes it through `frozen_sweep`.

---

## Round 42 (2026-05-01 03:30) — No Fresh Sandia Trace/Cachesim Artifact

### Finding

The latest `vinge.local` scan shows no live Sandia process and no new Sandia
trace/cachesim artifact beyond the existing root docs. No generated long trace
or six-policy cache report is visible for the current race surface.

### Recommended Action

Keep Sandia out of the numeric race table until it produces a generated trace
and a `tools/cachesim` comparison against the same fixed real manifests. The
next Sandia milestone is not another training log; it is a validated long
rollout.

## Round 43 (2026-05-01 03:50) — Still No Sandia Long-Trace Entry

### Finding

The latest `vinge.local` peer scan showed no live Sandia `newgan` process and
no fresh Sandia generated trace or `tools/cachesim` report on the current
Tencent/Alibaba race surfaces. The active numeric fight is LANL versus LLNL
atlas traces; Sandia remains outside the cache-simulator table.

### Recommended Action

Keep Sandia gated on first generated long rollout plus six-policy cache report.
Training progress alone does not enter the race table.

## Round 44 (2026-05-01 04:05) — No Sandia Cache-Sim Artifact During Tencent/Alibaba Atlas Push

### Finding

The latest peer scan still shows no fresh Sandia generated trace or cache-sim
report on the active Tencent or Alibaba gates. The live work is LANL and LLNL
atlas/reuse-shaping; Sandia remains absent from the numeric table.

### Recommended Action

Leave Sandia gated on a generated long rollout plus `tools/cachesim` report.

## Round 45 (2026-05-01 04:20) — No Sandia Entry While LANL/LLNL Trade Tencent Lead

### Finding

The latest scan found no Sandia `newgan` process and no fresh Sandia cache-sim
artifact. During the same window, LANL and LLNL both produced Tencent/Alibaba
atlas cache rows, but Sandia still has no comparable long trace.

### Recommended Action

Keep Sandia out of the numeric race table until it produces a generated trace
and cache-simulator report.

## Round 46 (2026-05-01 04:35) — Still No Sandia Cache-Sim Entry During R208/R209

### Finding

The latest peer scan still finds no Sandia generated 1M trace or
`tools/cachesim` report for Tencent, Alibaba, or CloudPhysics. The active race
front is now LLNL R208 versus LANL R209 on Alibaba and LANL p `.60` adjacency
confirmation on Tencent; Sandia has no artifact on either surface.

### Recommended Action

Keep Sandia gated on a generated long trace plus the shared cache-simulator
panel. Code or training progress without a scored trace does not enter the
race table.

## Round 47 (2026-05-01 04:45) — Sandia Still Absent After LANL R209 Alibaba Re-Pass

### Finding

LANL R209 produced a new Alibaba six-policy cache row (`0.017939`) and LLNL
R208 remains the eight-policy Alibaba target. Sandia still has no generated
long trace, no Tencent/Alibaba/CloudPhysics cache-sim report, and no visible
race artifact on `vinge.local`.

### Recommended Action

Keep Sandia out of the numeric standings until a generated long trace and
cache-simulator panel exist.

## Round 48 (2026-05-01 05:00) — No Sandia Artifact While Alibaba Lead Moves Again

### Finding

LANL corrected the R209 Alibaba launcher and produced visible six-policy and
eight-policy cache leads. LLNL continues CloudPhysics generation. Sandia still
has no generated long trace or cache-simulator panel visible on the shared
race surfaces.

### Recommended Action

Keep Sandia out of standings until it produces an evaluated trace.

## Round 49 (2026-05-01 05:25) — Still No Sandia Trace While Cachesim Becomes The Gate

### Finding

The latest scan found no Sandia generated long trace, no live Sandia
generation process, and no Sandia `tools/cachesim` report on the Tencent or
Alibaba fixed-manifest surfaces. During the same loop, LANL added a standalone
peer CSV comparer and scored LLNL R210 plus LANL Alibaba eight-policy fills.

### Recommended Action

Keep Sandia out of the numeric standings. The admission ticket is now explicit:
a generated long trace plus the shared cache-simulator policy grid.

## Round 50 (2026-05-01 05:55) — No Sandia Result During LANL Negative Bracket

### Finding

Another scan during the Alibaba k125/k150 bracket still found no Sandia
generated trace or cache-simulator report. LANL closed four Alibaba probes and
launched a new k100 bracket; LLNL R210 Tencent adj `.04/.06` were scored.
Sandia still has no comparable artifact.

### Recommended Action

Keep the Sandia gate unchanged: generated long trace, fixed real manifest,
shared cache-simulator report.

## Round 51 (2026-05-01 06:20) — Sandia Still Has No Cache-Sim Row

### Finding

No Sandia generated long trace or cache-simulator report appeared during the
k100 follow-up. The visible race remains LANL versus LLNL neural-atlas/reuse
shaping on Tencent and Alibaba.

### Recommended Action

Do not rank Sandia until it has a generated trace scored on the shared
cache-simulator grid.

## Round 52 (2026-05-01 06:45) — No Sandia Entry During Fresh-Seed Confirmation

### Finding

No Sandia long-trace artifact appeared while LANL ran the p `.08` fresh-seed
confirmations. Sandia still has no generated trace plus cache-simulator report
on the fixed Tencent or Alibaba manifests.

### Recommended Action

Keep Sandia out of the race table until it produces the shared evaluation
artifact.

## Round 53 (2026-05-01 07:10) — Sandia Still Absent During Lower-Reuse Pivot

### Finding

No Sandia generated long trace or cache-simulator report appeared while LANL
closed the p `.10` confirmation set and pivoted to lower-reuse Alibaba rows.

### Recommended Action

Keep Sandia unranked until it produces a comparable long trace and cachesim
panel.

## Round 54 (2026-05-01 07:35) — No Sandia Result During Shape/Cache Split

### Finding

No Sandia artifact appeared during the lower-reuse Alibaba confirmation loop.
The race remains LANL/LLNL on shared cache-simulator panels.

### Recommended Action

Keep Sandia gated on a generated long trace and cachesim report.

## Round 55 (2026-05-01 07:55) — Sandia Still Missing As LANL Finds Balanced Alibaba Row

### Finding

No Sandia trace or cache-simulator report appeared while LANL found the
p `.06`/hp `.18,k100` Alibaba row. Sandia remains absent from the numeric race.

### Recommended Action

Keep Sandia unranked until it produces a comparable cachesim-validated long
trace.

## Round 56 (2026-05-01 08:15) — No Sandia Artifact As Alibaba Target Drops Again

### Finding

No Sandia cache-simulator artifact appeared while LANL moved Alibaba to
p `.06`/hp `.20,k100`. Sandia remains outside the measured race.

### Recommended Action

Keep Sandia unranked until it has a generated trace on the shared evaluation
surface.

## Round 57 (2026-05-01 08:35) — No Sandia Artifact As hp .22 Becomes Target

### Finding

No Sandia cachesim artifact appeared while LANL lowered Alibaba to hp
`.22,k100`. Sandia still has no comparable long trace on the fixed manifests.

### Recommended Action

Keep Sandia out of standings until it has the shared long-rollout/cachesim
artifact.

## Round 58 (2026-05-01 08:55) — Sandia Still Absent As hp .24 Leads Alibaba

### Finding

No Sandia generated trace or cachesim report appeared while LANL moved Alibaba
to hp `.24,k100`.

### Recommended Action

Keep Sandia unranked until it produces a comparable artifact.

## Round 59 (2026-05-01 09:15) — No Sandia Artifact As hp .26 Leads Alibaba

### Finding

No Sandia generated trace or cache-simulator report appeared while LANL moved
Alibaba again to hp `.26,k100`.

### Recommended Action

Keep Sandia unranked pending a comparable long-trace artifact.

## Round 60 (2026-05-01 09:35) — Sandia Still Absent After hp .26 Confirmation

### Finding

No Sandia long-trace or cachesim artifact appeared while LANL confirmed the hp
`.26,k100` Alibaba row below eight-policy `0.019`.

### Recommended Action

Keep Sandia unranked until a comparable artifact exists.

## Round 61 (2026-05-01 09:55) — No Sandia Artifact As hp .30 Becomes Target

### Finding

No Sandia generated trace or cache-simulator report appeared while LANL moved
the Alibaba eight-policy target to hp `.30,k100` at `0.018831`.

### Recommended Action

Keep Sandia unranked until it produces a generated long trace and the shared
cachesim report on the fixed manifests.

## Round 62 (2026-05-01 10:25) — No Sandia Artifact As hp .34 Becomes Target

### Finding

No Sandia generated trace or cache-simulator report appeared while LANL moved
Alibaba again to hp `.34,k100` eight-policy `0.018056` and hp `.30,k125`
six-policy `0.015788`.

### Recommended Action

Keep Sandia unranked until it produces a generated long trace plus the shared
cachesim report.

## Round 63 (2026-05-01 10:50) — No Sandia Artifact As k125 Becomes Target

### Finding

No Sandia generated trace or cache-simulator report appeared while LANL moved
Alibaba to hp `.34,k125` eight-policy `0.017767` and hp `.30,k125`
six-policy `0.015567`.

### Recommended Action

Keep Sandia unranked until it produces a comparable long trace and cachesim
panel.
