# cachesim

Cache simulator for evaluating real and synthetic I/O traces from
`llgan/generate.py` and `altgan/generate.py`. First Rust crate in the
repo; chosen over Python for speed on long-rollout simulations.

## Policies

| Policy | Reference                                  | Notes                                |
| ------ | ------------------------------------------ | ------------------------------------ |
| FIFO   | folklore                                   | Insert-time order, no hit reordering |
| LRU    | Belady-era textbook                        | Recency only                         |
| SLRU   | Karedla–Love–Wherry, IEEE Computer 1994    | 80/20 protected/probationary         |
| ARC    | Megiddo–Modha, FAST 2003                   | Adaptive recency/frequency           |
| CAR    | Bansal–Modha, FAST 2004                    | ARC over CLOCK lists, lazy promotion |
| SIEVE  | Zhang–Yang–Yue–Vigfusson, NSDI 2024        | One-bit FIFO with reverse hand       |

All policies share an O(1) doubly-linked arena (`policy::util::DList`):
hit, miss, eviction, and ghost manipulation are all amortised constant
time per access.

## Build

```sh
cd tools/cachesim
cargo build --release
```

Toolchain pinned in `rust-toolchain.toml` (stable). All trace bytes go in
`tools/cachesim/testdata/`, which is gitignored.

## Run

```sh
cargo run --release -- \
    --trace path/to/trace.oracleGeneral.zst \
    --policy lru,arc,fifo,sieve,slru,car \
    --cache-sizes 1024,4096,16384,65536,262144 \
    --out report.json
```

The trace is loaded once into memory; every (policy × cache_size) pair
is simulated in parallel via rayon. On vinge (20 cores) a 14M-access
oracleGeneral trace clears 30 simulations in ~3s wall-clock.

## Planned CLI extensions

```
cachesim
    --real  <path>                  # enables HRC-MAE vs real
    --grid  lanl-tencent            # named cache-size grids
    --format auto|oracle|csv
    --n-streams 4 --seed 42
```

## Output schema

JSON, schema-compatible with `llgan/long_rollout_eval.py` sidecar:

```json
{
    "policy": "arc",
    "hrc_mae_vs_real": null,
    "reuse_access_rate": 0.0,
    "stack_distance_median": null,
    "stack_distance_p90": null,
    "footprint_mean_per_stream": 0,
    "n_accesses": 13934482,
    "per_cache_size": [{"size": 1024, "miss_ratio": 0.4573}, ...]
}
```

`hrc_mae_vs_real`, the stack-distance fields, `reuse_access_rate`, and
`footprint_mean_per_stream` populate when the Mattson single-pass HRC and
the paired-trace HRC-MAE pass land.

## Validation gates (pre-1.0)

1. Unit: LRU on Mattson textbook example; ARC on Megiddo–Modha 5-request
   example. (Done.)
2. Real-vs-real: HRC-MAE(real, real) ≈ 0 on a fetched Tencent and
   Alibaba `.zst`. (Pending.)
3. Real-vs-fake: reproduce LANL Tencent **0.00887** on a paired altgan
   CSV. (Pending.)
