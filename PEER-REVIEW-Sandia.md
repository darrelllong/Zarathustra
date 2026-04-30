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

`newgan/v1_baseline.sh` sets `--pretrain-ae-epochs 0 --pretrain-sup-epochs 0 --pretrain-g-epochs 0`. Without the AE / Supervisor / G-warmup pretrain phases, Phase 3 starts the generator from random init against a critic that converges in ~5 epochs to a perfect classifier. This is the canonical "no-pretrain WGAN collapse" mode — confirmed in LLNL v196 post-mortem (VERSIONS.md L266) and elsewhere.

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
