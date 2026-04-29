# cachesim

Cache simulator (LRU, ARC) for evaluating real and synthetic I/O traces from
`llgan/generate.py` and `altgan/generate.py`. First Rust crate in the repo;
chosen over Python for speed on long-rollout simulations.

## Status

**Pre-staged skeleton** — compiles, CLI parses, LRU policy passes basic unit
tests. ARC, Mattson stack-distance HRC, real-vs-fake HRC-MAE, and grid loaders
land once a real Tencent / Alibaba `.zst` is available locally for validation
(see `TODO.md` → "Rust cache simulator").

## Build

```sh
cd tools/cachesim
cargo build --release
```

Toolchain pinned in `rust-toolchain.toml` (stable). All trace bytes go in
`tools/cachesim/testdata/`, which is gitignored.

## Run (current capability)

```sh
cargo run --release -- \
    --trace path/to/synthetic.csv \
    --policy lru \
    --cache-size 65536 \
    --out report.json
```

## Planned CLI (final v1)

```
cachesim
    --trace <path>
    --real  <path>                  # optional; enables HRC-MAE vs real
    --policy lru,arc
    --cache-size N | --cache-sizes N1,N2,... | --grid lanl-tencent
    --format auto|oracle|csv
    --n-streams 4 --seed 42
    --threads 0
    --out eval.json
```

## Output schema

JSON byte-identical to `llgan/long_rollout_eval.py` sidecar:

```json
{
    "policy": "lru",
    "hrc_mae_vs_real": 0.00887,
    "reuse_access_rate": 0.6145,
    "stack_distance_median": 60,
    "stack_distance_p90": 174,
    "footprint_mean_per_stream": 9627,
    "n_accesses": 100000,
    "per_cache_size": [{"size": 128, "miss_ratio": 0.41}, ...]
}
```

## Validation gates (pre-1.0)

1. Unit: LRU on Mattson textbook example; ARC on Megiddo–Modha 5-request example.
2. Real-vs-real: HRC-MAE(real, real) ≈ 0 on a fetched Tencent and Alibaba `.zst`.
3. Real-vs-fake: reproduce LANL Tencent **0.00887** on a paired altgan CSV.
