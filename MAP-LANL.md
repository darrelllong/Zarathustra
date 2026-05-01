# MAP-LANL.md - Cognitive Map of `altgan/`

LANL owns `altgan/`: the explicit cache-object-process track for the
Zarathustra race. This map is a navigation aid, not a tutorial. Keep it current
when code paths, promoted recipes, or long-rollout conclusions change.

Last refreshed: 2026-05-01, during the Tencent/Alibaba cache-sim race loop
after the LLNL R210 adj `.04` score and the LANL Alibaba eight-policy fill.

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

- Current Tencent cache best is p `.60`, k50, tail `.10`, adj `.015`, fake
  seed `58`: six-policy mean `0.030240`, evaluator HRC `0.008832`, median
  `81`, p90 `33757`, top100 `0.260899`, top1000 `0.402591`, adjdup
  `0.015475`. Confirmation seed `59` scored `0.030301`; adj `.010` seed `60`
  was slightly worse at `0.030386`. LLNL R210 adj `.04` scored `0.030856` and
  adj `.06` scored `0.030526`, so R210 does not retake the visible Tencent
  cache lead.

## Current Alibaba Gate

- Fixed real manifest:
  `/tiamat/zarathustra/altgan-output/alibaba_real_manifest_seed42_1M_manifest.json`
- Current LANL six-policy cache best: deep-reuse p `.08` plus hot-pool
  `.10,k125,window10000`, fake seed `62`, six-policy `0.017260`,
  eight-policy `0.022470`. Median is low (`257` versus real `276`), p90 is
  high (`59278` versus real `44829`).
- Current LANL eight-policy cache best: deep-reuse p `.08` plus hot-pool
  `.15,k100,window10000`, fake seed `61`, eight-policy `0.021637` and
  six-policy `0.017641`. p `.08`/hp `.12`/k125 seed `68` is the nearest
  bridge at six-policy `0.017100` and eight-policy `0.022172`.
- Seed-46 is not yet robust. The first follow-up launcher malformed decimal
  probabilities (`.10` as `.010`, `.25` as `.025`); those rows are rejected.
  Corrected hp `.10` seed `49` scored six-policy `0.017547` and eight-policy
  `0.022264`; hp `.15` seed `50` scored six-policy `0.018764` and eight-policy
  `0.022021`; hp `.12` seed `56` improved eight-policy to `0.021982`; hp
  `.10,k100` seed `57` improved six-policy to `0.017524`. hp `.20`, hp `.25`,
  and hp `.25` + tail `.30` lose.
- Current LLNL cache best: R208 adj `.00` at six-policy `0.019671`; adj `.02`
  has its best rescored eight-policy row at `0.022266`; LANL now leads both
  visible cache panels, with much lower adjacency debt.
- Closed negative: p `.08`/hp `.15`/k125 seed `70` (`0.018845` six,
  `0.023364` eight), p `.075`/hp `.12`/k125 seed `71` (`0.019830`,
  `0.024602`), p `.08`/hp `.10`/k150 seed `72` (`0.020138`, `0.025011`), and
  p `.07`/hp `.12`/k125 seed `73` (`0.019743`, `0.024429`).
- Also closed negative: k100 p `.08`/hp `.12` seed `74` (`0.020097` six,
  `0.024603` eight), p `.08`/hp `.13` seed `75` (`0.019734`, `0.024003`), and
  p `.085`/hp `.15` seed `76` (`0.018410`, `0.022816`).
- Fresh-seed p `.08` confirmations were weak: p `.08`/hp `.10,k125` seed `77`
  scored `0.020158` six and `0.024861` eight; p `.08`/hp `.15,k100` seed `78`
  scored `0.019030`/`0.023338`; p `.08`/hp `.12,k125` seed `79` scored
  `0.020151`/`0.024729`. Treat the p `.08` best rows as seed-fragile.
- p `.10` confirmations were also weak on fresh seeds: hp `.12,k75` seed `80`
  scored `0.019992` six and `0.024045` eight; hp `.10,k100` seed `81` scored
  `0.019997`/`0.024628`; hp `.10,k75` seed `82` scored `0.019926`/`0.024444`.
- Lower-reuse confirmations show a shape/cache split: p `.06`/hp `.10,k125`
  seed `83` scored `0.020902` six and `0.025591` eight with p90 `44240`;
  p `.065`/hp `.10,k125` seed `84` scored `0.020901`/`0.025524`; p
  `.06`/hp `.12,k125` seed `85` scored `0.020368`/`0.024968` with p90
  `43801`. These match reuse/p90 much better but lose policy curves.
- Low-reuse/high-hot-pool found the current best balanced Alibaba row:
  p `.06`/hp `.18,k100` seed `88` scored `0.018282` six-policy and `0.022144`
  eight-policy, with reuse `0.307590`, median `240`, p90 `43194`. This beats
  LLNL R208 eight-policy (`0.022266`) while keeping trace shape close.
  hp `.15,k100` seed `86` was weaker (`0.019406`/`0.023606`), and
  hp `.15,k125` seed `87` lost (`0.019937`/`0.024477`).
- The neighbor probe promoted p `.06`/hp `.20,k100` seed `90`: six-policy
  `0.017356`, eight-policy `0.020988`, reuse `0.306875`, median `240`, p90
  `43721`. Same hp `.18,k100` confirmed at seed `89` (`0.018010`/`0.022058`);
  hp `.18,k125` seed `91` scored `0.017992`/`0.022436`.
- hp `.24,k100` seed `96` is the new Alibaba target: six-policy `0.016666`,
  eight-policy `0.019718`, reuse `0.306815`, median `240`, p90 `43898`.
  hp `.22,k100` confirmed at seed `95` (`0.016740`/`0.019927`); hp `.22,k75`
  seed `97` scored `0.017604`/`0.020350`.
- Live LANL probes: hp `.24,k100` seed `99` confirmation, hp `.26,k100` seed
  `98`, and hp `.24,k75` seed `100`. Launch these with math-library thread
  caps; uncapped four-way launches got stuck in startup/CPU probing before
  opening model or trace files.

| Row | HRC-MAE | fake reuse | real reuse | fake med | real med | fake p90 | real p90 | mark |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| promoted `0.575/0.70` | 0.05899 | 0.61286 | 0.72841 | 54 | 84 | 170 | 29150 | 0.03086 |
| old `0.55/0.8` | 0.05982 | 0.61385 | 0.72841 | 53 | 84 | 169 | 29150 | 0.03027 |
| promoted + tail scale 340 | 0.08607 | 0.61286 | 0.72841 | 54 | 84 | 24224 | 29150 | 0.03086 |
| promoted + reuse boost 0.30, min-rank 84 | 0.05921 | 0.72971 | 0.72841 | 72 | 84 | 14132 | 29150 | 0.03265 |

Conclusion: rank-tail stretching is real but insufficient. It can move p90,
but it cannot fix the upper HRC curve while total fake reuse stays near 0.613
and real reuse is 0.728. Controlled new-to-reuse conversion fixes the reuse
total, but a low min-rank over-hits the low/mid cache curve. The active bracket
pushes injected reuse deeper (`min_rank=4096`) and emits a CSV for
`tools/cachesim`.

1M artifacts:

- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_seed42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_old_tb055_lp080_seed42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_tailp84_tails340_seed42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_reuseboost030_min84_pow2_seed42_eval_1M.json`

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
| `altgan/cachesim_compare.py` | Standalone peer-trace scorer for existing fake/real CSVs. Runs `tools/cachesim` on both sides and writes policy/capacity HRC-MAE JSON. |
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
        else if reuse-boost fires:
            convert NEW into a deep LRU reuse
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
- `stack_reuse_boost_prob`, `stack_reuse_boost_min_rank`,
  `stack_reuse_boost_rank_power`: converts some sampled NEW events into reuse
  events and samples the injected rank from the live LRU stack. Current
  evidence: `prob=0.30,min_rank=32768,power=2` fixes total reuse and much of
  the long tail when applied as a post-decode correction. The emitted action is
  passed to the mark runtime, but transition rollout stays on the originally
  sampled state; feeding injected reuses back into transition state amplified
  the wrong reuse cascade.
- `stack_adj_dup_prob`: optional rank-0 reuse injection for SIEVE/CLOCK style
  probes. Current Tencent 1M exact-slice diagnostic says LANL fake already has
  more adjacent duplicates than real (`0.00427` vs `0.00234`), so this is a
  diagnostic knob, not a promoted fix.
- `stack_hot_pool_prob`, `stack_hot_pool_k`, `stack_hot_pool_window`,
  `stack_hot_pool_weight_power`: redirects ordinary sampled reuses, not deep
  injected reuses, toward the recent hot set. Current evidence: `p=0.50,k=100`
  improves six-policy mean HRC-MAE from `0.054073` to `0.046657`, mainly by
  fixing SIEVE/FIFO/LRU, while top-100 share remains low (`0.1195` vs real
  `0.2640`). `p=0.60`/`0.70` both lost to `p=0.50`; frequency weight power
  `2.0` catastrophically over-concentrated and is closed negative. Wide-window
  `50000` attempts were killed as pathological under both exact lookup and a
  cached-rank experiment. Current promoted hot-pool row remains
  `p=0.50,k=100,window=5000,wpow=1`.
- `stack_hot_pool_max_search`: bounds hot-object lookup to a stack prefix and
  falls back to the normal sampled rank if the hot object is deeper. Added
  after seed-43 exact lookup ran 40+ minutes without a fake CSV. `max_search=8192`
  and `max_search=512` were also killed at 40+ minutes. Fixed real-manifest
  confirmation with fake RNG seed `43` was also killed as slow; fake RNG seed
  `44` is running.
- `_RankedLRUStack`: implicit-treap LRU stack used by generation. It preserves
  rank semantics while making `move_to_front(rank)` and hot-object `index()`
  logarithmic instead of scanning/moving a Python list across hundreds of
  thousands of objects. Randomized equivalence tests compare it against the old
  list behavior before using it for the next confirmation runs.
- `--progress-interval`: evaluator/generator logging knob for long 1M runs.
  Use it on confirmation launches so a silent log means startup or failure, not
  merely "still inside generation."
- Confirmed `p=0.50` hot-pool on fake seed `44` against the fixed seed-42 real
  manifest: six-policy mean `0.046945` versus seed-42 `0.046657`; top-100
  share `0.123185`, top-1000 `0.384067`, adjacent duplicate rate `0.004372`.
  Bracketing moved the live best lower: `p=.45` scored `0.045864` on fake seed
  `44` and `0.045988` on fake seed `42`; `p=.55` lost at `0.047347`; `p=.40`
  scored `0.045660` on fake seed `44` and `0.045651` on fake seed `42`;
  `p=.38` improved fake seed `44` again to `0.045386` and confirmed on seed
  `42` at `0.045648`; `p=.37` seed-44 is essentially tied at `0.045395`;
  `p=.37` seed-42 scored `0.045599`, `p=.39` seed-44 scored `0.045532`, and
  `p=.35`/`p=.42` lost on their checked seeds. Treat `p=.37..40` as the robust
  probability band. Pool-size sweep says keep `k=100`: `k=75` scored
  `0.045715`, `k=150` scored `0.047746`, both worse than `k=100` at
  `0.045386`. Window sweep says wider is useful up to at least `10000`:
  `window=2500` lost at `0.045842`, `window=10000` improved to `0.045255`.
  `window=10000` confirmed on seed `42` at `0.045352`; `window=20000` seed-44
  nudged to `0.045243` but did not confirm on seed `42` (`0.045465`), and
  `window=40000` lost at `0.045855`. Current promotion is `window=10000`.
  Probability recheck at that window moved the seed-44 best to `p=.39`
  (`0.045219`), but fake seeds `45`/`46` came back weaker at
  `0.045573`/`0.045627`. Do not promote `p=.39` yet. The p38/window10000
  variance pair landed at `0.045614`/`0.045511`, so the robust LANL Tencent
  band remains roughly `0.0455..0.0456`.
- Alibaba 1M cache gate is now open on
  `/tiamat/zarathustra/altgan-output/alibaba_real_manifest_seed42_1M_manifest.json`.
  Control is the existing `alibaba_phaseatlas_marks_e20.pkl.gz` checkpoint with
  reservoir marks, forced phase, `transition_blend=0.2`, and
  `local_prob_power=0.9`; it scores six-policy mean `0.020282`. The useful
  Alibaba knob is deep new-to-reuse injection (`min_rank=32768`,
  `rank_power=2.0`). Hot-pool concentration over-compresses the reuse tail.
  `p=0.10` is the current cache best (`0.019857`/`0.019892`), while `p=0.06`
  is the trace-shape compromise (`0.020009`/`0.020072`) because it matches
  reuse and p90 much more closely. LLNL R207 hp `0.40` narrowed Alibaba to
  `0.025387` but still trails LANL and carries adjdup `0.058331`.
- `stack_tail_reuse_prob`, `stack_tail_reuse_min_frac`,
  `stack_recent_pool_prob`, and `stack_recent_pool_window`: LANL-side ports of
  the LLNL R203 reuse-shaping levers. Tail reuse redirects sampled reuses into
  the deep half/tail of the live stack; recent-pool reuse redirects to the
  latest emitted objects. Both are evaluated through the normal LANL fake/real
  CSV and `tools/cachesim` path.
- Peer map: LLNL's active race path has moved from GAN checkpoints to
  `llgan.neural_atlas` b2 traces with hand-shaped reuse controls. Its visible
  Alibaba R204 k-axis scores `0.050148`, `0.033206`, and `0.029747` for
  k25/k75/k100 against LANL's fixed Alibaba real manifest; LANL remains ahead
  on that simulator surface. On Tencent, LLNL R203 k25 scores `0.038256` and
  LLNL R206 k50 adj `0.075` scores `0.030360`; adj `0.03` is close at
  `0.031474` with lower but still high adjacent duplicates, and R206 adj `0.00`
  scores `0.043287` with realistic adjacent duplicates but poor SIEVE. LANL's direct
  k25/adj `0.15` clone failed at `0.107924`, but the k50 tail clone worked:
  adj `0.05` scored `0.031461`, adj `0.00` scored `0.031040`, and adj `0.02`
  scored `0.030632` with adjdup `0.018330`. Tail `0.08` closed negative, and
  adj `0.03`/`0.04` were slightly worse (`0.030802`/`0.030963`). Raising
  hot-pool to `0.60` at adj `0.02` scored `0.030298`, edging LLNL's visible
  `0.030360` while keeping lower adjdup (`0.018463` vs LLNL `0.045438`).
  Live follow-ups are a same-recipe confirmation and hot-pool `0.60`/adj
  `0.015`; Sandia has no visible fresh trace/cachesim artifact in the latest
  scan.
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
7. `--fake-output`, `--real-output`, and `--cachesim-bin` should be used on 1M
   probes that need `tools/cachesim` comparison. The simulator gate is now part
   of the evaluator path, not a separate afterthought.
8. Do not sync stale local code over vinge without compiling and checking CLI
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
  --stack-rank-phase-scales 1.2,1.2,1.3,1.3 \
  --stack-reuse-boost-prob 0.30 \
  --stack-reuse-boost-min-rank 32768 \
  --stack-reuse-boost-rank-power 2.0 \
  --fake-output /tiamat/zarathustra/altgan-output/current_fake.csv \
  --real-output /tiamat/zarathustra/altgan-output/current_real.csv \
  --cachesim-bin tools/cachesim/target/release/cachesim'
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
