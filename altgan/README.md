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

## TraceBootstrap (bootstrap ledger)

To publish/pin multi-seed TraceBootstrap rows on the official cachesim surface,
use the multi-seed launcher (prints pasteable literal cachesim mean lines + exact
JSON means for `altgan/RESULTS.md` and `RESPONSE-LANL.md`):

```bash
python -m altgan.launch_trace_bootstrap_multiseed \
  --corpus twitter \
  --trace-dir /tiamat/zarathustra/traces/twitter_cluster \
  --fmt oracle_general \
  --real-manifest /tiamat/zarathustra/llgan-output/manifests/twitter_cluster_stackatlas.json \
  --real-ref /tiamat/zarathustra/llgan-output/refs/twitter_cluster_real.csv \
  --mode shuffle \
  --chunk-size 65536 \
  --seeds 42,80,81,82 \
  --emit-markdown
```

If the run is interrupted, re-run with `--skip-existing` to resume without
recomputing completed seeds.

To save the paste-ready Markdown (and a machine-readable JSON summary) to files:

```bash
python -m altgan.launch_trace_bootstrap_multiseed \
  ... \
  --emit-markdown-to /tmp/tracebootstrap_twitter_shuffle.md \
  --emit-summary-json-to /tmp/tracebootstrap_twitter_shuffle.json
```

To publish/refresh the full 1M-corpus shuffle pack (Twitter / Meta KV / Meta CDN /
Wikipedia) in one go:

```bash
python -m altgan.launch_trace_bootstrap_shuffle_pack \
  --markdown \
  --skip-existing \
  --keep-going \
  --emit-markdown-dir /tmp/tracebootstrap_shuffle_snips \
  --emit-summary-json-dir /tmp/tracebootstrap_shuffle_snips
```

## IRD-renewal (Wikipedia / CloudPhysics)

The IRD-renewal sweep launcher can also emit/append paste-ready Markdown tables
with the per-seed literal cachesim mean line + exact JSON mean:

```bash
python3 -m altgan.launch_ird_renewal_sweep \
  --real /tiamat/zarathustra/llgan-output/refs/wiki_real.csv \
  --output-root /tiamat/zarathustra/altgan-output \
  --corpus wiki \
  --cache-sizes 32,128,512,2048,8192 \
  --policies lru,arc,fifo,sieve,slru,car \
  --seeds 42,80,81,82 \
  --spec "rb16_s28:ird_s=28,ip=0.10,rb=16" \
  --emit-markdown
```

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
  --device auto \
  --output checkpoints/altgan/alibaba_phaseatlas_marks.pkl.gz
```

The attached mark head conditions on workload profile, atlas state,
new/reuse-action class, stack-rank bucket, and previous emitted marks. It leaves
the explicit LRU object decoder untouched, so HRC/reuse/stack-distance can be
compared directly against the reservoir-mark PhaseAtlas checkpoint.
Use `altgan.evaluate_neural_atlas --disable-neural-marks` on an attached
checkpoint to run the paired reservoir-mark control.
Use `--mark-temperature` and `--mark-numeric-noise` to ablate sampling noise
without retraining the object process.
Use `--mark-numeric-blend` and `--mark-categorical-source reservoir` for the
hybrid control: keep reservoir opcode/tenant and blend only neural timing/size.
Add `--mark-numeric-blend-space log` to blend timing/size in the same log scale
used by the mark-quality score; the default `raw` mode preserves prior results.

To run the full paired hybrid sweep and write a CSV summary:

```bash
python -m altgan.sweep_mark_hybrids \
  --model checkpoints/altgan/alibaba_phaseatlas_marks.pkl.gz \
  --trace-dir /tiamat/zarathustra/traces/2020_alibabaBlock \
  --fmt oracle_general \
  --char-file /home/darrell/traces/characterization/trace_characterizations.jsonl \
  --real-manifest /home/darrell/long_rollout_manifests/alibaba_stackatlas.json \
  --output-dir /tiamat/zarathustra/altgan-output \
  --prefix alibaba_phaseatlas_marks_hybrid \
  --transition-blends 0.0,0.2 \
  --local-prob-powers 0.9,1.0 \
  --mark-numeric-blend-spaces raw,log \
  --include-reservoir-control \
  --skip-existing
```

The sweep writes both `*_summary.csv` and `*_best.json`; use `--skip-existing`
when resuming a remote run so completed eval JSONs are reused. Add
`--seeds 42,43,44,45` for the stability pass; the best JSON then includes
candidate means across seeds so a single lucky HRC draw does not promote a
weaker mark recipe.
Use `--transition-blends` and `--local-prob-powers` to run object-process
microblend controls in the same paired panel as raw/log neural-mark hybrids.

To turn a completed mark-hybrid sweep into a `RESULTS.md` section:

```bash
python -m altgan.report_mark_hybrids \
  --summary-csv /tiamat/zarathustra/altgan-output/alibaba_phaseatlas_marks_hybrid_summary.csv \
  --best-json /tiamat/zarathustra/altgan-output/alibaba_phaseatlas_marks_hybrid_best.json \
  --append-results altgan/RESULTS.md
```

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

For an HRC-first PhaseAtlas calibration sweep:

```bash
python -m altgan.sweep_phaseatlas_hrc \
  --model checkpoints/altgan/alibaba_phaseatlas.pkl.gz \
  --trace-dir /tiamat/zarathustra/traces/2020_alibabaBlock \
  --fmt oracle_general \
  --char-file /home/darrell/traces/characterization/trace_characterizations.jsonl \
  --real-manifest /home/darrell/long_rollout_manifests/alibaba_stackatlas.json \
  --output-dir /tiamat/zarathustra/altgan-output \
  --prefix alibaba_phaseatlas_hrc \
  --transition-blends 0.0,0.25,0.5,0.75,1.0 \
  --phase-modes natural,forced \
  --stack-rank-scales 0.75,1.0,1.25 \
  --stack-rank-maxes -1,512,1024 \
  --phase-stack-rank-scale-schedules "0.75,0.9,1.0,1.0,1.1,1.15,1.2,1.25" \
  --phase-stack-rank-max-schedules "-1,-1,1024,1024,768,768,512,512" \
  --panels 4x100000,8x50000 \
  --skip-existing
```

The sweep writes `*_summary.csv` and `*_best.json`, ranked by HRC-MAE first, so
interrupted remote runs can resume without losing completed cells. The stack
rank scale/max controls are HRC calibration knobs around the reuse tail. Phase
schedules let those knobs vary across PhaseAtlas bins; omit them and leave the
global knobs at `1.0` and `-1` for the unmodified PhaseAtlas baseline. When
phase schedules are supplied, the runner also includes the unscheduled global
baseline unless `--no-global-baseline` is set.

To turn a completed sweep CSV into a `RESULTS.md` section:

```bash
python -m altgan.report_phaseatlas_hrc \
  --summary-csv /tiamat/zarathustra/altgan-output/alibaba_phaseatlas_hrc_summary.csv \
  --top-n 5 \
  --append-results altgan/RESULTS.md
```

The evaluator also writes a `mark_quality` block for timing, size, opcode, and
tenant realism. To compare two already-generated CSVs directly:

```bash
python -m altgan.mark_quality \
  --fake-csv outputs/fake.csv \
  --real-csv outputs/real.csv
```

## Tencent cache-surface chunk selector

For Tencent, LANL also has a post-hoc cache-surface chunk selector that keeps an
atlas trace's timing/marks but swaps synthetic object streams in contiguous
chunks, accepting replacements only when they improve the official
`llgan.cachesim_eval` mean across the 6-policy surface.

Single-seed runs can be launched with `altgan.optimize_tencent_chunk_surface`.
For reproducible multi-seed pipelines (race protocol), use:

```bash
python -m altgan.launch_tencent_chunk_surface_multiseed \
  --real /tiamat/zarathustra/llgan-output/refs/tencent_stackatlas_real.csv \
  --base-template "/tiamat/zarathustra/altgan-output/TENCENT_BASE_seed{seed}_fake_100k.csv" \
  --donor-templates "/tiamat/zarathustra/altgan-output/DONOR_A_seed{seed}_fake_100k.csv,/tiamat/zarathustra/altgan-output/DONOR_B_seed{seed}_fake_100k.csv" \
  --tag-prefix tencent_chunksurf_rXXX \
  --pipeline 2048,1024,512,256 \
  --seeds 42,80,81,82
```

The launcher prints per-seed file paths plus a copy/pastable literal
`mean HRC-MAE across policies: ...` line and exact JSON mean. For direct
doc updates, use `--emit-markdown` (prints a ready-to-paste table + mean/range)
or `--append-markdown <path>` to append that snippet to `RESPONSE-LANL.md` or
`altgan/RESULTS.md`.

## Generic cache-surface chunk selector (any corpus)

Despite the module name, `altgan.optimize_tencent_chunk_surface` is corpus-agnostic
and is used across other corpora (Alibaba/MSR/Twitter/CloudPhysics/etc). For a
generic multi-seed runner (supports arbitrary policies/sizes and donor globs),
use:

```bash
python -m altgan.launch_chunk_surface_multiseed \
  --real /tiamat/zarathustra/llgan-output/refs/<CORPUS>_real.csv \
  --base-template "/tiamat/zarathustra/altgan-output/<BASE>_seed{seed}_fake_1000k.csv" \
  --donor-globs "/tiamat/zarathustra/altgan-output/<DONOR_PREFIX>*_seed{seed}_fake_1000k.csv" \
  --tag-prefix <TAG> \
  --pipeline 65536 \
  --cache-sizes 32,128,512,2048,8192 \
  --policies lru,arc,fifo,sieve,slru,car \
  --seeds 42,80,81,82 \
  --emit-markdown
```

CloudPhysics uses an 8-policy official surface (note the extra cache size and
policies). Example (8K tightening from the current r306 16K fakes):

```bash
python -m altgan.launch_chunk_surface_multiseed \
  --real /tiamat/zarathustra/llgan-output/refs/cloudphysics_stackatlas_real.csv \
  --base-template "/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r306_refine16_ck16384_seed{seed}_fake_1000k.csv" \
  --donor-globs "/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r306*_seed{seed}_fake_1000k.csv,/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r305*_seed{seed}_fake_1000k.csv,/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r304*_seed{seed}_fake_1000k.csv,/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r292*_seed{seed}_fake_1000k.csv" \
  --tag-prefix cloudphysics_chunksurf_rXXX_refine8 \
  --pipeline 8192 \
  --cache-sizes 32,128,512,2048,8192,32768 \
  --policies lru,arc,fifo,sieve,slru,car,lfu,lirs \
  --max-accepts 4 \
  --max-evals 120 \
  --seeds 42,80,81,82 \
  --emit-markdown
```

## Why this is the bet

If the altgan family wins long-rollout HRC and stack-distance while losing some
short-window smooth-feature score, that is still useful: it proves the current
model family is missing an explicit object process.  If it loses everywhere,
the failure will be informative because the stack law has been isolated from
the neural mark model.
