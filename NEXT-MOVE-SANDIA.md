# Sandia next move

The agent overwrites this file at the end of every turn. The next turn
reads it as the first hint of what to do. Format is fixed; do not add
sections beyond the four below.

## Environment is provisioned — STOP REFUSING TO WORK

The "broken environment" excuse is over. As of 2026-05-03 ~11:30 PT,
the Sandia stack on `baase.local` is provisioned and waiting:

- **Python venv**: `/home/darrell/sandia-venv/` on `baase.local`'s
  local NVMe (NOT NFS-shared with vinge — NFS small-file IO was too
  slow for pip extraction). Has `torch 2.11.0+cu130` (CUDA detected,
  1 GPU), `numpy`, `pandas`, `tqdm`, `scipy`, `pyarrow`.
  **Always invoke its python directly:**
  ```
  ssh baase.local '/home/darrell/sandia-venv/bin/python -m llgan.cachesim_eval ...'
  ssh baase.local '/home/darrell/sandia-venv/bin/python newgan/train.py ...'
  ```
  Do NOT call `python3` (that's the system python; PEP 668 will block
  any pip install, and it has no torch). Do NOT try to `source ... &&`
  the venv — non-interactive ssh doesn't reliably activate.
- **Rust toolchain**: `cargo`, `rustc`, `rustup` at `/usr/bin/` on
  `baase.local`. On default PATH. Use `ssh baase.local 'cargo build
  --release'` if you fork the cachesim. (Charter says you usually
  don't — the existing wrapper `python3 -m llgan.cachesim_eval`
  invokes the prebuilt binary already.)
- **Datasets**: `/tiamat/zarathustra/traces/` and the manifests at
  `/tiamat/zarathustra/llgan-output/manifests/...` exist. Run
  `ssh baase.local 'ls /tiamat/zarathustra/traces/'` to see them.
- **No sudo. None. Never.** If a step "needs sudo," it's the wrong step.

If your output starts with "I cannot run X because Y is missing,"
re-read this file — Y is not missing.

**You do not run `pip install`. Ever.** The venv is operator-managed.
If a Python import fails inside the venv, that is not your problem
to fix this turn — log the missing module name in your turn output
and the operator will install it. Two agents `pip install`-ing into
the same venv simultaneously corrupts both installs (already happened
once today; do not repeat). If you find yourself typing `pip install`
or `python -m pip`, stop and use the venv as-is.

## Last turn

(no real Sandia measurement has been shipped yet — the eight
`0.0000` rows in `RESPONSE-Sandia.md` from 2026-05-03 are fabricated
placeholder rows from the abandoned qwen3-coder:30b and nemotron-
cascade-2 agents. They have been or are being reverted; do not treat
them as real measurements.)

- Hypothesis tried: n/a
- Measurement: n/a
- Gap to LLNL: n/a (Sandia has zero real measurements)

## Next turn

- **Target corpus**: `msr_exchange`. Smallest dataset (1M records),
  fastest GPU iteration, cheapest pipeline check. LLNL bar 0.0253.
- **Hypothesis to try**: stand up the train→generate→eval pipeline
  end-to-end on `baase.local` using the provisioned venv. Goal of
  this turn is *the pipeline runs and emits a real cachesim mean* —
  not yet to beat LLNL. The first real number, even a bad one, beats
  the eight 0.0000 fabrications.
- **Concrete first commands** (run them; don't paraphrase them):
  ```
  ssh baase.local 'ls /tiamat/zarathustra/traces/msr_exchange/ | head'
  ssh baase.local '/home/darrell/sandia-venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"'
  ssh baase.local '/home/darrell/sandia-venv/bin/python -m llgan.trace_bootstrap --help' | head
  ```
  If those three pass, you have a working remote stack. Then run a
  real bootstrap measurement (not to win, just to prove the loop):
  ```
  ssh baase.local '/home/darrell/sandia-venv/bin/python -m llgan.trace_bootstrap \
      --mode shuffle \
      --trace-dir /tiamat/zarathustra/traces/msr_exchange \
      --fmt oracle_general \
      --real-manifest /tiamat/zarathustra/llgan-output/manifests/msr_exchange_stackatlas.json \
      --output /tiamat/zarathustra/sandia-output/sandia_msr_seed42.csv \
      --seed 42 --n-records 1000000 --chunk-size 65536'
  ssh baase.local '/home/darrell/sandia-venv/bin/python -m llgan.cachesim_eval \
      --fake /tiamat/zarathustra/sandia-output/sandia_msr_seed42.csv \
      --real /tiamat/zarathustra/llgan-output/refs/msr_exchange_stackatlas_real.csv \
      --cache-sizes 32,128,512,2048,8192 \
      --policies lru,arc,fifo,sieve,slru,car \
      2>&1 | tee /tiamat/zarathustra/sandia-output/sandia_msr_seed42.log'
  ```
  Then `ssh baase.local cat /tiamat/zarathustra/sandia-output/sandia_msr_seed42.log | grep "mean HRC-MAE"` → that is your real, measured mean. Append the row, drop the claim file pointing at the real /tiamat/ paths, commit, push.

## Pipeline reminders for next turn

- ssh from Mac to baase: `ssh baase.local` (the user's `~/.ssh/config`
  has it set up with the right user, key, and ProxyJump — no flags needed).
- All artifacts go under `/tiamat/zarathustra/sandia-output/` (NFS,
  visible from every host; create with `mkdir -p` on the remote).
- Log path: write to `/tiamat/zarathustra/sandia-output/<corpus>_<seed>.log`
  (NFS-shared) and reference THAT path in `.sandia-claim.json` — the
  Mac can `ssh baase.local cat <log>` if the structural pre-commit hook
  needs to verify it. Do NOT use `/tmp/foo.log` on baase — the Mac
  can't see /tmp on baase, and the hook will reject the claim.
- Pre-commit hook is structural only (no LLM). It checks: claim file
  exists, fake_csv path exists on disk (probes baase via ssh if path
  is `/tiamat/...`), log contains literal claimed_mean, RESPONSE row
  matches. Honor all four; fabrications are caught at commit time.
