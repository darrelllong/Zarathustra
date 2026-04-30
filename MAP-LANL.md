# MAP-LANL.md - Cognitive Map of `altgan/`

LANL owns `altgan/`: the explicit cache-object-process track for the
Zarathustra race. This map is a navigation aid, not a tutorial. Keep it current
when code paths, promoted recipes, or long-rollout conclusions change.

Last refreshed: 2026-04-30, after the Tencent 1M tail check.

---

## Operating Contract

- Local workspace: `/Users/darrell/Zarathustra`
- Do not leave this tree locally.
- LANL write scope: `altgan/`, `MAP-LANL.md`, `PEER-REVIEW-*.md`,
  `IDEAS*.md`, `VERSIONS*.md`, `RESPONSE*.md`, and `REBUTTAL*.md`.
- Remote execution host: `vinge.local`
- Remote trace/artifact root: `/tiamat/zarathustra`
- Remote repo mirror: `/home/darrell/Zarathustra`
- Keep root team docs suffixed. No bare `RESPONSE.md`, `VERSIONS.md`,
  `IDEAS.md`, `RESULTS.md`, or generic LANL-owned `PEER-REVIEW.md`.

Before promotion or long runs, verify:

```bash
cd /Users/darrell/Zarathustra
python3 -m py_compile altgan/*.py
ssh vinge.local 'cd /home/darrell/Zarathustra && python3 -m py_compile altgan/*.py'
```

---

## Big Picture

`altgan/` is not another GAN loss branch. It makes cache-locality explicit:

```text
real oracle_general traces
        |
        v
canonical event frame
        |
        v
stack distance + action class + timing/size/opcode/tenant marks
        |
        +--> StackAtlas reservoir state model
        |
        +--> NeuralAtlas profile-conditioned transition model
                  |
                  v
          explicit LRU stack decoder
                  |
                  v
          synthetic long trace
                  |
                  v
          llgan long-rollout metrics:
          HRC-MAE, reuse access, stack median/p90, footprint, drift, mark score
```

The core bet: HRC, reuse, stack distance, and object identity should be
first-class generation variables, not accidental consequences of a normalized
neural decoder.

---

## Current Tencent Promotion

Checkpoint:

`/tiamat/zarathustra/checkpoints/altgan/tencent_phaseatlas_marks_e20_128files_h128_catw025.pkl.gz`

Object runtime:

- forced phase schedule
- `transition_blend=0.575`
- `local_prob_power=0.70`
- `stack_rank_phase_scales=1.0,1.0,1.1,1.1`

Mark runtime:

- neural opcode/tenant categoricals
- emitted numeric marks from reservoir: `mark_numeric_blend=0.0`
- feedback-only size log blend: `mark_feedback_numeric_blend=0.080`
- `mark_feedback_numeric_fields=size`
- `mark_numeric_noise=0.05`

100k seed panels promoted this row on HRC over the old `0.55/0.8` row. The 1M
panel changed the diagnosis:

| Row | HRC-MAE | fake reuse | real reuse | fake med | real med | fake p90 | real p90 | mark |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| promoted `0.575/0.70` | 0.05899 | 0.61286 | 0.72841 | 54 | 84 | 170 | 29150 | 0.03086 |
| old `0.55/0.8` | 0.05982 | 0.61385 | 0.72841 | 53 | 84 | 169 | 29150 | 0.03027 |
| promoted + tail scale 340 | 0.08607 | 0.61286 | 0.72841 | 54 | 84 | 24224 | 29150 | 0.03086 |

Conclusion: rank-tail stretching is real but insufficient. It can move p90,
but it cannot fix the upper HRC curve while total fake reuse stays near 0.613
and real reuse is 0.728. The next LANL code move should target controlled
new-to-reuse conversion plus long-rank selection, not only rank scaling.

1M artifacts:

- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_seed42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_old_tb055_lp080_seed42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_tailp84_tails340_seed42_eval_1M.json`

---

## File Index

### Core object-process model

| File | Role |
|---|---|
| `altgan/model.py` | Base `EventSample` and `StackAtlasModel`. Canonicalizes traces, computes stack distances, action classes, quantile bins, reservoir samples, and explicit stack generation. |
| `altgan/neural_stack.py` | Older conditioned marginal stack model. Useful baseline; not current champion. |
| `altgan/neural_atlas.py` | Current object-process engine. `NeuralAtlasModel.generate()` blends neural transitions with nearest-file empirical transitions, forces phase when requested, decodes object ids through an LRU stack, and optionally attaches neural marks. |

### Mark model

| File | Role |
|---|---|
| `altgan/neural_marks.py` | IDEA #53 mark sidecar. `NeuralMarkHead` stores metadata; `NeuralMarkRuntime` samples autoregressive opcode/tenant/dt/size marks and observes emitted or feedback-adjusted marks. |
| `altgan/mark_quality.py` | Mark score: normalized log-W1 for `ts_delta` and `obj_size`, plus TV distance for opcode and tenant. |

### Train and evaluate entry points

| File | Role |
|---|---|
| `altgan/train.py` | Trains base `StackAtlasModel`. |
| `altgan/train_neural.py` | Trains `NeuralStackModel`. |
| `altgan/train_neural_atlas.py` | Trains `NeuralAtlasModel`/PhaseAtlas. Use `--n-phase-bins 8` and `--exclude-manifest` for strict holdout. |
| `altgan/train_neural_marks.py` | Attaches or snapshots neural mark heads onto a frozen NeuralAtlas/PhaseAtlas checkpoint. Supports loss weights and snapshot epochs. |
| `altgan/evaluate.py` | Evaluates base StackAtlas checkpoints through long-rollout metrics. |
| `altgan/evaluate_neural.py` | Evaluates NeuralStack checkpoints. |
| `altgan/evaluate_neural_atlas.py` | Main evaluator for current LANL work. Computes fake/real/gap/mark-quality JSON and records runtime knobs. |
| `altgan/generate.py` | Minimal generator wrapper for base atlas checkpoints. |

### Sweep and report helpers

| File | Role |
|---|---|
| `altgan/sweep_phaseatlas_hrc.py` | HRC-first sweep over transition blend, phase mode, rank scale/cap schedules, and panels. |
| `altgan/sweep_mark_hybrids.py` | Current workhorse for paired mark/object sweeps. Supports exact `--object-candidates`, feedback-only mark grids, `--jobs`, and tail-rank knobs. |
| `altgan/report_phaseatlas_hrc.py` | Converts HRC sweep CSVs into markdown summaries. |
| `altgan/report_mark_hybrids.py` | Converts mark-hybrid sweep CSVs into markdown summaries and best-candidate notes. |
| `altgan/RESULTS.md` | Detailed LANL result ledger. Keep this more granular than `VERSIONS-LANL.md`. |
| `altgan/README.md` | Operator-facing quickstart. Should agree with this map on major knobs. |

---

## `NeuralAtlasModel.generate()` Mental Model

The hot path is `altgan/neural_atlas.py`.

```text
resolve conds
  |
  v
load tiny conditional transition net
  |
  v
precompute init/trans probabilities per stream
  |
  v
for each stream:
    choose nearest empirical reservoir by condition vector
    blend neural initial distribution with reservoir initial distribution
    initialize explicit LRU stack
    for each position:
        optionally force phase from stream position
        sample event marks from reservoir bucket for current state
        if action says reuse:
            calibrate sampled stack rank
            pick object from LRU stack and move to front
        else:
            allocate new object id using sampled stride
        optionally sample neural mark sidecar
        optionally blend numeric marks for emitted output
        optionally blend numeric marks only for feedback state
        observe feedback mark in mark runtime
        sample next state from neural/reservoir blend
```

Important knobs:

- `transition_blend`: neural transition weight; lower means more nearest-file
  empirical transition law.
- `local_prob_power`: sharpens or flattens empirical transition probabilities.
- `force_phase_schedule`: overwrites sampled phase with synthetic position
  phase.
- `stack_rank_scale`: global stack-rank multiplier.
- `stack_rank_phase_scales`: per-phase stack-rank multipliers.
- `stack_rank_max` / `stack_rank_phase_maxes`: rank caps; often hurt HRC when
  they truncate useful tail.
- `stack_rank_tail_pivot` / `stack_rank_tail_scale`: tail-only rank stretch.
  Current evidence: useful diagnostic, not enough alone.
- `mark_feedback_numeric_blend`: numeric blend used only as autoregressive mark
  feedback; preserves emitted reservoir numeric marks when `mark_numeric_blend`
  is `0.0`.

---

## Current Invariants

1. Do not train on eval real-manifest files when making strict holdout claims.
2. Use `--condition-from-real-manifest` for paired evals unless explicitly
   testing condition mismatch.
3. Paired controls must share seed, real manifest, n-records, n-streams, mark
   runtime, phase mode, and rank schedule except for the knob under test.
4. Single seed wins do not promote. Require paired panels across fresh fake
   seeds.
5. 100k HRC wins do not imply 1M trace realism. The Tencent 1M panel exposed a
   missing long-reuse tail and missing total reuse.
6. Feedback-only mark blending must be recorded separately from emitted numeric
   blending.
7. Do not sync stale local code over vinge without compiling and checking CLI
   flags.

---

## Common Remote Commands

Sync LANL code:

```bash
cd /Users/darrell/Zarathustra
rsync -az --delete --exclude '__pycache__/' altgan/ \
  vinge.local:/home/darrell/Zarathustra/altgan/
ssh vinge.local 'cd /home/darrell/Zarathustra && python3 -m py_compile altgan/*.py'
```

Check root doc namespace on vinge:

```bash
ssh vinge.local 'cd /home/darrell/Zarathustra && find . -maxdepth 1 -type f \
  \( -name "*LANL*.md" -o -name "*LLNL*.md" -o -name "PEER-REVIEW.md" \
     -o -name "RESPONSE.md" -o -name "VERSIONS.md" -o -name "IDEAS.md" \
     -o -name "RESULTS.md" \) -print | sort'
```

Run current promoted 1M row:

```bash
ssh vinge.local 'cd /home/darrell/Zarathustra && python3 -m altgan.evaluate_neural_atlas \
  --model /tiamat/zarathustra/checkpoints/altgan/tencent_phaseatlas_marks_e20_128files_h128_catw025.pkl.gz \
  --trace-dir /home/darrell/traces/tencent_block_1M \
  --fmt oracle_general \
  --char-file /tiamat/zarathustra/analysis/out/trace_characterizations.jsonl \
  --cond-dim 13 \
  --condition-from-real-manifest \
  --transition-blend 0.575 \
  --local-prob-power 0.70 \
  --temperature 1.0 \
  --stack-rank-scale 1.0 \
  --stack-rank-max -1 \
  --n-records 1000000 \
  --n-streams 4 \
  --seed 42 \
  --output /tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_seed42_eval_1M.json \
  --mark-categorical-source neural \
  --mark-numeric-blend 0.0 \
  --mark-numeric-blend-space log \
  --mark-numeric-fields size \
  --mark-temperature 1.0 \
  --mark-numeric-noise 0.05 \
  --mark-feedback-numeric-blend 0.08 \
  --mark-feedback-numeric-blend-space log \
  --mark-feedback-numeric-fields size \
  --force-phase-schedule \
  --stack-rank-phase-scales 1.0,1.0,1.1,1.1'
```

---

## Update Protocol

Refresh this file when:

- a new LANL runtime knob is added;
- a checkpoint or object/mark recipe is promoted;
- a long-rollout result changes the diagnosis;
- a file changes role or a new entry point appears;
- root doc ownership changes.

Minimum update set after a meaningful loop:

1. `MAP-LANL.md` for code shape and current mental model.
2. `altgan/RESULTS.md` for detailed metrics and artifact paths.
3. `VERSIONS-LANL.md` for promoted milestones.
4. `RESPONSE-LANL.md` for cross-team response posture.
5. `PEER-REVIEW-LLNL.md` and `PEER-REVIEW-Sandia.md` for each loop's peer scan.
