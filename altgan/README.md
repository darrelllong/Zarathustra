# altgan: explicit object-process challengers

`altgan` is a deliberately different model family for Zarathustra.  It is not
another boundary-critic weight, seed basin, or scalar loss.  It generates the
cache-relevant object stream explicitly.

The current LLGAN path normalizes raw `obj_id` into `obj_id_reuse` and
`obj_id_stride`, then asks a neural decoder to rediscover realistic object
locality indirectly.  Long-rollout review showed the failure mode: low
reuse-access, positional IRD near 1, and stack-distance near 0 even when
short-window `★` looks competitive.

The first StackAtlas model attacks that directly:

1. Fit a coarse regime Markov chain over real events.
2. Label each real access as new object, near reuse, mid reuse, or far reuse.
3. Store empirical mark samples by regime: inter-arrival time, size, opcode,
   tenant, new-object stride, and stack distance.
4. Generate with an explicit LRU stack.  Reuse events sample a stack-distance
   bucket and move that object to the top; new events allocate a new object.

This makes HRC, reuse-access, positional IRD, and stack-distance first-class
generation outputs rather than post-hoc diagnostics.

`NeuralStack` and `NeuralAtlas` are the trained follow-ups:

- `NeuralStack` maps a file/workload descriptor to new/reuse and stack-rank
  probabilities while retaining explicit LRU-stack generation.
- `NeuralAtlas` trains a profile-conditioned transition model over the same
  coarse StackAtlas states.  It can also blend the trained neural transition
  smoother with the nearest real file's fitted transition atlas; that blend is
  intentionally exposed because the long-rollout panel decides whether neural
  smoothing helps or hurts.

## Operating Rule

LANL owns `altgan/`. LLNL owns `llgan/`; read it for intelligence, but do not
edit it from this branch. When LANL changes code, results, reviews, or strategy,
commit and push before leaving the work unattended.

## Train

```bash
python -m altgan.train \
  --trace-dir /tiamat/zarathustra/traces/2020_alibabaBlock \
  --fmt oracle_general \
  --max-files 16 \
  --records-per-file 50000 \
  --output checkpoints/altgan/alibaba_stackatlas.pkl.gz
```

Use `--max-files 0` for the full directory once the smoke test looks right.

## Train NeuralAtlas

```bash
python -m altgan.train_neural_atlas \
  --trace-dir /tiamat/zarathustra/traces/2020_alibabaBlock \
  --fmt oracle_general \
  --char-file /home/darrell/traces/characterization/trace_characterizations.jsonl \
  --max-files 64 \
  --records-per-file 25000 \
  --epochs 900 \
  --hidden-dim 128 \
  --output checkpoints/altgan/alibaba_neuralatlas.pkl.gz
```

For a nonstationary phase-conditioned atlas, add `--n-phase-bins 8`. For a
strict long-rollout holdout, add `--exclude-manifest path/to/real_manifest.json`
so the real eval stream files cannot be used as routed source atlases.

## Attach Neural Marks

Freeze a trained NeuralAtlas/PhaseAtlas object process and add the IDEA #53 mark
sidecar:

```bash
python -m altgan.train_neural_marks \
  --model checkpoints/altgan/alibaba_phaseatlas.pkl.gz \
  --trace-dir /tiamat/zarathustra/traces/2020_alibabaBlock \
  --fmt oracle_general \
  --char-file /home/darrell/traces/characterization/trace_characterizations.jsonl \
  --exclude-manifest /home/darrell/long_rollout_manifests/alibaba_stackatlas.json \
  --max-files 64 \
  --records-per-file 25000 \
  --epochs 20 \
  --output checkpoints/altgan/alibaba_phaseatlas_marks.pkl.gz
```

The attached mark head conditions on workload profile, atlas state,
new/reuse-action class, stack-rank bucket, and previous emitted marks. It leaves
the explicit LRU object decoder untouched, so HRC/reuse/stack-distance can be
compared directly against the reservoir-mark PhaseAtlas checkpoint.
Use `altgan.evaluate_neural_atlas --disable-neural-marks` on an attached
checkpoint to run the paired reservoir-mark control.

## Generate

```bash
python -m altgan.generate \
  --model checkpoints/altgan/alibaba_stackatlas.pkl.gz \
  --n-records 100000 \
  --n-streams 4 \
  --seed 42 \
  --output outputs/alibaba_stackatlas.csv
```

## Evaluate

```bash
python -m altgan.evaluate \
  --model checkpoints/altgan/alibaba_stackatlas.pkl.gz \
  --trace-dir /tiamat/zarathustra/traces/2020_alibabaBlock \
  --fmt oracle_general \
  --n-records 100000 \
  --n-streams 4 \
  --seed 42 \
  --real-manifest /home/darrell/long_rollout_manifests/alibaba_stackatlas.json
```

For NeuralAtlas:

```bash
python -m altgan.evaluate_neural_atlas \
  --model checkpoints/altgan/alibaba_neuralatlas.pkl.gz \
  --trace-dir /tiamat/zarathustra/traces/2020_alibabaBlock \
  --fmt oracle_general \
  --char-file /home/darrell/traces/characterization/trace_characterizations.jsonl \
  --condition-from-real-manifest \
  --transition-blend 0.5 \
  --n-records 100000 \
  --n-streams 4 \
  --seed 42 \
  --real-manifest /home/darrell/long_rollout_manifests/alibaba_stackatlas.json
```

The evaluator reuses `llgan.long_rollout_eval`'s HRC/reuse/IRD/stack-distance
metrics, so it is directly comparable to the long-rollout panels in
`VERSIONS.md`.

The evaluator also writes a `mark_quality` block for timing, size, opcode, and
tenant realism. To compare two already-generated CSVs directly:

```bash
python -m altgan.mark_quality \
  --fake-csv outputs/fake.csv \
  --real-csv outputs/real.csv
```

## Why this is the bet

If the altgan family wins long-rollout HRC and stack-distance while losing some
short-window smooth-feature score, that is still useful: it proves the current
model family is missing an explicit object process.  If it loses everywhere,
the failure will be informative because the stack law has been isolated from
the neural mark model.
