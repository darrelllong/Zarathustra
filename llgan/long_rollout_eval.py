"""
Long-rollout diagnostic sidecar for LLGAN checkpoints.

Round 20 peer-review P1 #2: the frozen-bundle evaluator is a short-window
selector. Persistent-memory / IRD / chained-window ideas (#28/#31/#32) need a
long-rollout diagnostic so a cache-fidelity win does not hide as a combined-
score wash AND a combined-score win does not hide over-copying.

This tool runs a deterministic long-rollout against a trained checkpoint and
computes:

  * HRC (LRU hit-ratio curve) per-stream and averaged.
  * Overall reuse rate.
  * Local (within-decile) reuse-rate drift first vs. last decile.
  * POSITIONAL inter-reference-distance (IRD) histogram — recurrence distance
    in access positions, NOT stack distance.
  * STACK-DISTANCE (reuse-distance) histogram — number of distinct intervening
    keys between successive occurrences; this is the quantity that governs
    LRU HRC and IDEA #32's cache-footprint target.
  * Per-stream first-half vs. second-half Wasserstein-1 drift on obj_size and
    ts_delta, averaged across streams (true temporal drift, not
    between-stream heterogeneity).

It emits both the fake metrics and a real-trace baseline, plus gap summaries.
Output is written as JSON next to the checkpoint and printed as a terminal
summary.

Round 21 peer-review fixes (this revision):
  * `--char-file` / `--source-traces` feed conditioning from the same
    workload-descriptor pool that frozen eval / generate.py use. Without a
    char-file the sidecar refuses to run unless `--random-conditioning` is
    explicitly set (so accidental random-descriptor sidecars can't be
    confused with workload-matched sidecars).
  * Stack-distance histogram added alongside positional IRD.
  * Drift metrics (ts_delta / obj_size W1) are computed per-stream and then
    averaged across streams.
  * `--real-manifest` persists the exact real files and record counts that
    fed the baseline, so the "real baseline" is reproducible even if the
    trace directory changes.

CLI
---
    python -m llgan.long_rollout_eval \\
        --checkpoint /home/darrell/checkpoints/alibaba_v162/final.pt \\
        --trace-dir /tiamat/zarathustra/traces/alibaba \\
        --fmt oracle_general \\
        --char-file /tiamat/zarathustra/trace_characterizations.jsonl \\
        --real-manifest /home/darrell/long_rollout_manifests/alibaba.json

Default rollout: 100,000 records, 4 streams, seed=42 (torch + numpy + random +
CUDA). All results are strictly reproducible across runs for fixed inputs.

Integration
-----------
Promotion gate for #28/#31/#32 checkpoints:
  1. frozen_sweep pass (short-window ★ acceptable).
  2. long_rollout_eval pass:
       a. HRC-MAE vs. real ≤ threshold (track per corpus),
       b. reuse-rate drift (first-decile vs. last-decile) ≤ 0.15 abs,
       c. half-to-half Wasserstein on ts_delta / obj_size ≤ real-trace baseline
          × 1.25.

Notes
-----
- Does NOT modify training; strictly evaluation-side.
- Does NOT replace eval.py; complements it for long-horizon claims.
- The "real baseline" is sampled deterministically from --trace-dir with the
  same seed so cross-run comparisons are apples-to-apples, and is pinned to
  a manifest so future runs compare against the same files/record-slices.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

# Checkpoints pickled by train.py reference bare module names ("dataset",
# "model", "train") because train.py runs from the llgan/ directory. Add
# llgan/ AND its parent to sys.path so torch.load finds both "dataset" and
# "llgan.dataset". Must happen before any torch.load call.
_LLGAN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_LLGAN_DIR.parent))
sys.path.insert(0, str(_LLGAN_DIR))


# ---------------------------------------------------------------------------
# Deterministic seeding
# ---------------------------------------------------------------------------

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # cuDNN autotune and non-deterministic kernels would violate the
        # "strictly reproducible across runs" promise in the docstring.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Conditioning sourcing — matches eval.py / generate.py rather than using
# synthetic torch.randn descriptors.
# ---------------------------------------------------------------------------

def _resolve_conditioning(cond_dim: int, n_streams: int, *,
                          char_file: str, source_traces: list,
                          random_conditioning: bool, device):
    """Return (cond_tensor_or_None, metadata_dict).

    Policy (Round 21 P1): a long-rollout sidecar is promotion infrastructure.
    It MUST NOT silently fall back to random descriptor vectors — that would
    measure a different contract from frozen eval / generate.py.

    - If `cond_dim == 0`: model is unconditional; return (None, {...source:
      'unconditional'}).
    - If `char_file` and `source_traces` are both provided: look up each
      source trace's characterization vector and stack into (n_streams,
      cond_dim). If n_streams > len(source_traces), cycle; if fewer streams
      are asked for than sources, use the first n_streams.
    - If `char_file` is provided but `source_traces` is empty: draw
      n_streams conditioning vectors uniformly at random from the
      characterization pool (deterministic under the caller's seed).
    - If neither is provided: raise unless `random_conditioning=True`.
    """
    meta = {"cond_dim": cond_dim}
    if cond_dim <= 0:
        meta["source"] = "unconditional"
        return None, meta

    if char_file:
        from dataset import load_file_characterizations
        lookup = load_file_characterizations(char_file, cond_dim)
        if not lookup:
            raise RuntimeError(f"char_file {char_file} loaded no vectors")
        if source_traces:
            selected = []
            keys_used = []
            for name in source_traces:
                key = Path(name).name
                vec = lookup.get(key)
                if vec is None:
                    for ext in (".zst", ".gz"):
                        if key.endswith(ext):
                            vec = lookup.get(key[: -len(ext)])
                            if vec is not None:
                                break
                if vec is None:
                    raise RuntimeError(
                        f"source-trace {key!r} not in {char_file}")
                selected.append(vec)
                keys_used.append(key)
            if len(selected) < n_streams:
                # cycle to fill
                reps = math.ceil(n_streams / len(selected))
                selected = (selected * reps)[:n_streams]
                keys_used = (keys_used * reps)[:n_streams]
            else:
                selected = selected[:n_streams]
                keys_used = keys_used[:n_streams]
            cond = torch.stack(selected).to(device)
            meta["source"] = "source_traces"
            meta["keys"] = keys_used
            return cond, meta
        # char_file without source_traces: sample from pool deterministically
        pool_keys = sorted(lookup.keys())
        # Deterministic draw via torch.randperm so the caller's seed
        # controls which vectors are chosen.
        idx = torch.randperm(len(pool_keys))[:n_streams].tolist()
        if len(idx) < n_streams:
            # pool smaller than n_streams — cycle
            idx = (idx * math.ceil(n_streams / max(len(idx), 1)))[:n_streams]
        keys_used = [pool_keys[i] for i in idx]
        cond = torch.stack([lookup[k] for k in keys_used]).to(device)
        meta["source"] = "char_file_random_sample"
        meta["keys"] = keys_used
        return cond, meta

    if random_conditioning:
        cond = torch.randn(n_streams, cond_dim, device=device) * 0.5
        meta["source"] = "random_torch_randn_0.5"
        return cond, meta

    raise RuntimeError(
        "conditional checkpoint (cond_dim={}) requires --char-file (with or "
        "without --source-traces) or explicit --random-conditioning. Refusing "
        "to silently use synthetic descriptors.".format(cond_dim))


# ---------------------------------------------------------------------------
# Long-rollout generator (mirrors generate.generate internals but returns the
# stitched record array directly without CSV round-tripping).
# ---------------------------------------------------------------------------

def _rollout(ckpt, n_records: int, n_streams: int, device, *,
             char_file: str = "", source_traces=None,
             random_conditioning: bool = False,
             retrieval_persist: bool = False):
    """Run a deterministic long-rollout and return (df, conditioning_meta).

    df: pandas DataFrame with a stream_id column prepended. Mirrors
    generate.generate but avoids CSV I/O so metrics can be computed directly.

    conditioning_meta: dict describing the actual conditioning source
    ('unconditional' / 'source_traces' / 'char_file_random_sample' /
    'random_torch_randn_0.5'), to be written into the result JSON so sidecar
    comparisons can audit the descriptor contract.
    """
    import pandas as pd
    from .model import Generator, Recovery

    cfg = ckpt["config"]
    prep = ckpt["prep"]

    latent_ae = "R" in ckpt
    latent_dim = getattr(cfg, "latent_dim", 0) if latent_ae else None
    cond_dim = getattr(cfg, "cond_dim", 0)

    G = Generator(cfg.noise_dim, prep.num_cols, cfg.hidden_size,
                  latent_dim=latent_dim, cond_dim=cond_dim,
                  film_cond=getattr(cfg, "film_cond", False),
                  gmm_components=getattr(cfg, "gmm_components", 0),
                  var_cond=getattr(cfg, "var_cond", False),
                  n_regimes=getattr(cfg, "n_regimes", 0),
                  num_lstm_layers=getattr(cfg, "num_lstm_layers", 1),
                  gp_prior=getattr(cfg, "gp_prior", False),
                  timestep=cfg.timestep,
                  retrieval_memory=getattr(cfg, "retrieval_memory", False),
                  retrieval_mem_size=getattr(cfg, "retrieval_mem_size", 32),
                  retrieval_key_dim=getattr(cfg, "retrieval_key_dim", 32),
                  retrieval_val_dim=getattr(cfg, "retrieval_val_dim", 32),
                  retrieval_decay=getattr(cfg, "retrieval_decay", 0.85),
                  retrieval_tau_write=getattr(cfg, "retrieval_tau_write", 0.5),
                  retrieval_n_warmup=getattr(cfg, "retrieval_n_warmup", 4),
                  ssm_backbone=getattr(cfg, "ssm_backbone", False),
                  ssm_state_dim=getattr(cfg, "ssm_state_dim", 16),
                  mtpp_timing=getattr(cfg, "mtpp_timing", False),
                  mtpp_sigma_min=getattr(cfg, "mtpp_sigma_min", 0.05),
                  ).to(device)
    g_weights = ckpt.get("G_ema", ckpt["G"])
    G.load_state_dict({k: v.to(device) for k, v in g_weights.items()})
    G.eval()

    R = None
    if latent_ae:
        r_keys = ckpt["R"].keys()
        binary_cols = None
        if any(k.startswith("fc_cont") for k in r_keys):
            binary_cols = [i for i, col in enumerate(prep.col_names)
                           if col.lower() in {"opcode", "type", "rw", "op"}
                           or col.endswith("_reuse")]
        R = Recovery(latent_dim, cfg.hidden_size, prep.num_cols,
                     binary_cols=binary_cols).to(device)
        R.load_state_dict(ckpt["R"])
        R.eval()

    opcode_col = -1
    for i, col in enumerate(prep.col_names):
        if col.lower() in {"opcode", "type", "rw", "op"}:
            opcode_col = i
            break

    timestep = cfg.timestep
    records_per_stream = math.ceil(n_records / n_streams)
    windows_per_stream = math.ceil(records_per_stream / timestep)

    stream_windows: list[list[np.ndarray]] = [[] for _ in range(n_streams)]

    cond_tensor, cond_meta = _resolve_conditioning(
        cond_dim, n_streams,
        char_file=char_file,
        source_traces=source_traces or [],
        random_conditioning=random_conditioning,
        device=device,
    )

    with torch.no_grad():
        if cond_tensor is not None:
            cond = cond_tensor
            if getattr(G, 'cond_encoder', None) is not None:
                cond, _ = G.cond_encoder(cond, training=False)
            if getattr(G, 'regime_sampler', None) is not None:
                cond = G.regime_sampler(cond)
            noise = G.sample_noise(n_streams, device, cond=cond)
            z_global = torch.cat([cond, noise], dim=1)
        else:
            z_global = torch.randn(n_streams, cfg.noise_dim, device=device)
        hidden = None
        # IDEA #28: optionally thread the retrieval memory bank state across
        # window boundaries at eval time. Mirrors generate.py's
        # --retrieval-persist-across-windows contract.
        retrieval_enabled = getattr(G, "retrieval", None) is not None
        persist_retrieval = retrieval_persist and retrieval_enabled
        retrieval_state = None

        for _ in range(windows_per_stream):
            z_local = torch.randn(n_streams, timestep, cfg.noise_dim, device=device)
            if persist_retrieval:
                latent, hidden, retrieval_state = G(
                    z_global, z_local, hidden=hidden,
                    return_hidden=True,
                    retrieval_state=retrieval_state,
                    return_retrieval_state=True,
                )
                retrieval_state = {
                    k: (v.detach() if torch.is_tensor(v) else v)
                    for k, v in retrieval_state.items()
                }
            else:
                latent, hidden = G(z_global, z_local, hidden=hidden, return_hidden=True)
            # SSM backbone returns (state, None); LSTM returns (h, c). Guard the
            # second element so either backbone carries cleanly across windows.
            h0 = hidden[0].detach() if hidden[0] is not None else None
            h1 = hidden[1].detach() if hidden[1] is not None else None
            hidden = (h0, h1)
            out = R(latent) if R is not None else latent
            if opcode_col >= 0:
                out[:, :, opcode_col] = (out[:, :, opcode_col] >= 0).float() * 2 - 1
            out_np = out.cpu().numpy()
            for s in range(n_streams):
                stream_windows[s].append(out_np[s])

    per_stream_dfs = []
    for s in range(n_streams):
        arr_s = np.concatenate(stream_windows[s], axis=0)[:records_per_stream]
        df_s = prep.inverse_transform(arr_s)
        df_s.insert(0, "stream_id", s)
        per_stream_dfs.append(df_s)

    df = pd.concat(per_stream_dfs, ignore_index=True).head(n_records)
    return df, cond_meta


# ---------------------------------------------------------------------------
# Real-trace sampler — deterministic, matches rollout scale.
# ---------------------------------------------------------------------------

def _sample_real_stream(trace_dir: str, fmt: str, n_records: int, n_streams: int,
                        seed: int, manifest_path: str = ""):
    """Sample n_streams independent real trace streams and return
    (df, manifest_dict).

    Deterministic under (seed, trace_dir, fmt, n_streams) OR under a supplied
    manifest. The manifest pins the exact files and record counts that fed
    the baseline so the "real" side of a sidecar is reproducible even if the
    trace directory changes, files are added/removed/renamed, or readers
    become non-deterministic in how much they return per file.

    Manifest schema (JSON):
        {
          "trace_dir": "...",
          "fmt": "...",
          "n_records": int,
          "n_streams": int,
          "seed": int,
          "streams": [
            [{"path": "...", "records_taken": int}, ...],  # stream 0
            [{"path": "...", "records_taken": int}, ...],  # stream 1
            ...
          ]
        }

    Behavior:
      - If `manifest_path` is non-empty and the file exists, load it and
        replay exactly (assert (fmt, n_records, n_streams) match; the caller
        is responsible for using a consistent seed).
      - Else sample from the shuffled file pool as before, and if
        `manifest_path` is non-empty, write the manifest at the end.
    """
    import pandas as pd
    from llgan.train import _collect_files
    from llgan.dataset import _READERS

    reader = _READERS.get(fmt)
    if reader is None:
        raise RuntimeError(f"Unknown trace format '{fmt}'")

    records_per_stream = math.ceil(n_records / n_streams)

    # Replay from manifest if one exists.
    if manifest_path and Path(manifest_path).exists():
        manifest = json.loads(Path(manifest_path).read_text())
        if manifest.get("fmt") != fmt:
            raise RuntimeError(
                f"manifest fmt={manifest.get('fmt')!r} != requested {fmt!r}")
        if manifest.get("n_records") != n_records:
            raise RuntimeError(
                f"manifest n_records={manifest.get('n_records')} != "
                f"requested {n_records}")
        if manifest.get("n_streams") != n_streams:
            raise RuntimeError(
                f"manifest n_streams={manifest.get('n_streams')} != "
                f"requested {n_streams}")
        per_stream_dfs = []
        for s, entries in enumerate(manifest["streams"]):
            acc = []
            for entry in entries:
                path = entry["path"]
                want = int(entry["records_taken"])
                if not Path(path).exists():
                    raise RuntimeError(
                        f"manifest references missing file {path}")
                df = reader(path, want)
                if df is None or len(df) < want:
                    raise RuntimeError(
                        f"manifest replay short-read: {path} "
                        f"wanted={want} got={0 if df is None else len(df)}")
                acc.append(df.head(want))
            if not acc:
                continue
            df_s = pd.concat(acc, ignore_index=True).head(records_per_stream)
            df_s.insert(0, "stream_id", s)
            per_stream_dfs.append(df_s)
        if not per_stream_dfs:
            raise RuntimeError(f"empty manifest at {manifest_path}")
        df_all = pd.concat(per_stream_dfs, ignore_index=True).head(n_records)
        return df_all, manifest

    all_files = sorted(_collect_files(trace_dir, fmt))
    if not all_files:
        raise RuntimeError(f"No files found in {trace_dir}")

    rng = random.Random(seed)
    pool = all_files[:]
    rng.shuffle(pool)

    per_stream_dfs: list = []
    per_stream_manifest: list = []
    errors: list[str] = []
    file_idx = 0
    for s in range(n_streams):
        acc, have = [], 0
        entries: list = []
        while have < records_per_stream and file_idx < len(pool):
            path = pool[file_idx]
            file_idx += 1
            try:
                df = reader(str(path), records_per_stream)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{Path(path).name}: {e}")
                continue
            if df is None or len(df) == 0:
                continue
            # Track exact records taken (capped at need) so replay is
            # deterministic even if future reader versions return more/less.
            need = records_per_stream - have
            take = min(need, len(df))
            acc.append(df.head(take))
            entries.append({"path": str(path), "records_taken": int(take)})
            have += take
        if not acc:
            continue
        df_s = pd.concat(acc, ignore_index=True).head(records_per_stream)
        df_s.insert(0, "stream_id", s)
        per_stream_dfs.append(df_s)
        per_stream_manifest.append(entries)

    if not per_stream_dfs:
        raise RuntimeError(
            f"No real records from {trace_dir}; first errors: {errors[:3]}")
    df_all = pd.concat(per_stream_dfs, ignore_index=True).head(n_records)

    manifest = {
        "trace_dir": trace_dir,
        "fmt": fmt,
        "n_records": n_records,
        "n_streams": n_streams,
        "seed": seed,
        "streams": per_stream_manifest,
    }
    if manifest_path:
        Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
        Path(manifest_path).write_text(json.dumps(manifest, indent=2))
    return df_all, manifest


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _per_stream_obj_ids(df) -> list:
    """Return a list of per-stream obj_id numpy arrays in stream_id order.
    NO cross-stream offset — metrics are computed per-stream and aggregated
    by the caller. Cross-stream id collisions (e.g. two streams emitting
    the same id) are then correctly NOT counted as reuse, because each
    stream is measured independently."""
    if 'obj_id' not in df.columns:
        raise RuntimeError("DataFrame missing 'obj_id' column — long-rollout eval needs it.")
    out = []
    for _s, g in df.groupby('stream_id', sort=True):
        out.append(g['obj_id'].astype(np.int64).values)
    return out


def _hrc_single(obj_ids: np.ndarray, cache_sizes: np.ndarray) -> np.ndarray:
    """LRU HRC on a single stream at the given cache sizes."""
    hrcs = np.zeros(len(cache_sizes), dtype=np.float64)
    if len(obj_ids) == 0:
        return hrcs
    for j, cs in enumerate(cache_sizes):
        cache: "OrderedDict[int, bool]" = OrderedDict()
        hits = 0
        for oid in obj_ids:
            key = int(oid)
            if key in cache:
                hits += 1
                cache.move_to_end(key)
            else:
                cache[key] = True
                if len(cache) > cs:
                    cache.popitem(last=False)
        hrcs[j] = hits / len(obj_ids)
    return hrcs


def _ird_positional(obj_ids: np.ndarray) -> np.ndarray:
    """POSITIONAL inter-reference distance: for each position i where the
    object was seen before, the number of accesses between it and its
    previous occurrence (i.e., i - last_i). NOT stack distance. Caller
    should read the output as 'positional IRD'."""
    last = {}
    dists = []
    for i, oid in enumerate(obj_ids):
        key = int(oid)
        if key in last:
            dists.append(i - last[key])
        last[key] = i
    if not dists:
        return np.zeros(0, dtype=np.int64)
    return np.asarray(dists, dtype=np.int64)


def _stack_distances(obj_ids: np.ndarray) -> np.ndarray:
    """STACK / REUSE distance: for each reuse of a key, the number of
    DISTINCT intervening keys since its previous occurrence. This is the
    quantity that governs LRU HRC: stack distance d means the reuse is a hit
    iff cache size > d. Cold misses (first-access) are excluded from the
    returned array — those correspond to compulsory misses at any cache
    size.

    Exact O(N log N) implementation using a Fenwick (BIT) tree over
    positions. We maintain a bitmap of positions currently holding the
    latest-access of some unique key; for each reuse at position i of key
    k with prior position prev = last[k], the stack distance equals the
    number of active bits strictly greater than prev (i.e., unique keys
    whose last access was after prev and before i). We then clear bit
    prev, set bit i, and set last[k] = i.
    """
    N = len(obj_ids)
    if N == 0:
        return np.zeros(0, dtype=np.int64)

    # 1-indexed Fenwick tree of length N.
    bit = [0] * (N + 2)

    def bit_update(idx: int, delta: int) -> None:
        idx += 1  # 1-indexed
        while idx <= N:
            bit[idx] += delta
            idx += idx & -idx

    def bit_prefix(idx: int) -> int:
        """Sum of bit[0..idx] inclusive."""
        idx += 1
        s = 0
        while idx > 0:
            s += bit[idx]
            idx -= idx & -idx
        return s

    last: dict = {}
    out: list = []
    active = 0
    for i, oid in enumerate(obj_ids):
        key = int(oid)
        if key in last:
            prev = last[key]
            # active positions strictly greater than prev:
            sd = active - bit_prefix(prev)
            out.append(sd)
            bit_update(prev, -1)
            active -= 1
        last[key] = i
        bit_update(i, 1)
        active += 1
    if not out:
        return np.zeros(0, dtype=np.int64)
    return np.asarray(out, dtype=np.int64)


def _reuse_rate_deciles_cumulative(obj_ids: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Cumulative reuse rate per decile: access is reuse if the obj_id has
    been seen anywhere earlier in the stream. This is the classic coupon-
    collector / warmup curve — useful as a sanity check that both fake and
    real exhibit the same shape, NOT as a drift indicator (which AD
    correctly flagged: the curve is monotone by construction)."""
    N = len(obj_ids)
    if N == 0:
        return np.zeros(n_bins, dtype=np.float64)
    seen: set = set()
    is_reuse = np.zeros(N, dtype=np.int8)
    for i, oid in enumerate(obj_ids):
        key = int(oid)
        if key in seen:
            is_reuse[i] = 1
        else:
            seen.add(key)
    edges = np.linspace(0, N, n_bins + 1, dtype=np.int64)
    return np.array([is_reuse[edges[k]:edges[k+1]].mean() if edges[k+1] > edges[k] else 0.0
                     for k in range(n_bins)], dtype=np.float64)


def _reuse_rate_deciles_local(obj_ids: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """LOCAL reuse rate per decile: within each decile, fraction of accesses
    whose obj_id also appears elsewhere *within the same decile*. Isolates
    'is the generator's local locality stable across the stream' from
    cold-start warmup — this is the drift indicator the Round 20 reviewer
    actually cares about."""
    N = len(obj_ids)
    if N == 0:
        return np.zeros(n_bins, dtype=np.float64)
    edges = np.linspace(0, N, n_bins + 1, dtype=np.int64)
    out = np.zeros(n_bins, dtype=np.float64)
    for k in range(n_bins):
        chunk = obj_ids[edges[k]:edges[k+1]]
        if len(chunk) == 0:
            continue
        _uniq, counts = np.unique(chunk, return_counts=True)
        reused_ids = _uniq[counts > 1]
        # fraction of accesses landing on a locally-reused id
        hits = np.isin(chunk, reused_ids).sum()
        out[k] = hits / len(chunk)
    return out


def _half_drift(values: np.ndarray) -> tuple:
    """Wasserstein-1 between the first and second half of a 1-D sample,
    plus a scale constant (median absolute value) for unit-free comparison.

    Returns (w1, scale) where w1/scale is dimensionless and roughly
    comparable across corpora with very different value magnitudes. Caller
    divides: fake_w1 / max(scale, eps) vs real_w1 / max(scale, eps)."""
    if len(values) < 2:
        return 0.0, 1.0
    try:
        from scipy.stats import wasserstein_distance
    except Exception:
        mid = len(values) // 2
        a = np.sort(values[:mid])
        b = np.sort(values[mid:mid + len(a)])
        w1 = float(np.abs(a - b).mean())
    else:
        mid = len(values) // 2
        w1 = float(wasserstein_distance(values[:mid], values[mid:]))
    scale = float(np.median(np.abs(values))) if len(values) else 1.0
    # Pure div-by-zero guard (1e-12). A larger unit-bearing floor (e.g. 1.0)
    # would silently denormalize small-magnitude corpora; with this guard,
    # genuinely degenerate data produces an honest blow-up ratio instead.
    return w1, max(scale, 1e-12)


def _pooled_hist(arr_list: list, n_bins: int) -> tuple:
    """Build a log-spaced histogram from the pooled concatenation of a list
    of int arrays; return (hist_normalized, bin_edges, median, p90). Safe
    on empty input."""
    pooled = np.concatenate(arr_list) if arr_list else np.zeros(0, dtype=np.int64)
    if len(pooled) == 0:
        return [], [], 0, 0
    max_val = int(pooled.max())
    bin_edges = np.unique(np.geomspace(1, max(max_val, 2), n_bins + 1).astype(np.int64))
    hist, _ = np.histogram(pooled, bins=bin_edges)
    hist = hist.astype(np.float64) / max(hist.sum(), 1)
    return (hist.tolist(), bin_edges.tolist(),
            int(np.median(pooled)), int(np.percentile(pooled, 90)))


def _metrics_for_stream(df, cache_sizes: np.ndarray, n_ird_bins: int = 32) -> dict:
    """Compute metrics PER-STREAM and aggregate across streams. HRC is
    averaged across streams, IRD/stack-distance are pooled across streams,
    reuse rates are averaged, footprint is averaged, and drift (ts_delta /
    obj_size half-to-half W1) is computed PER-STREAM then averaged.

    Round 21 fix: drift metrics used to concatenate all streams and then
    split the pooled array in half, which measured between-stream
    heterogeneity rather than first-half vs second-half temporal drift.
    Now each stream's drift is computed independently and averaged."""
    per_stream = _per_stream_obj_ids(df)
    if not per_stream:
        raise RuntimeError("empty DataFrame passed to _metrics_for_stream")

    footprints = []
    reuse_access_rates = []
    reuse_object_rates = []
    deciles_cumulative = []
    deciles_local = []
    ird_pooled: list = []
    stack_pooled: list = []
    hrcs = []

    for obj_ids in per_stream:
        if len(obj_ids) == 0:
            continue
        unique_ids, id_counts = np.unique(obj_ids, return_counts=True)
        footprints.append(int(unique_ids.size))
        reuse_object_rates.append(float((id_counts > 1).sum()) / max(unique_ids.size, 1))
        total_acc = int(id_counts.sum())
        total_first_accesses = int(unique_ids.size)
        reuse_access_rates.append((total_acc - total_first_accesses) / total_acc)

        deciles_cumulative.append(_reuse_rate_deciles_cumulative(obj_ids, n_bins=10))
        deciles_local.append(_reuse_rate_deciles_local(obj_ids, n_bins=10))
        ird_pooled.append(_ird_positional(obj_ids))
        stack_pooled.append(_stack_distances(obj_ids))
        hrcs.append(_hrc_single(obj_ids, cache_sizes))

    ird_hist, ird_bin_edges, ird_positional_median, ird_positional_p90 = \
        _pooled_hist(ird_pooled, n_ird_bins)
    stack_hist, stack_bin_edges, stack_median, stack_p90 = \
        _pooled_hist(stack_pooled, n_ird_bins)

    hrc_avg = np.mean(hrcs, axis=0) if hrcs else np.zeros(len(cache_sizes))

    # Per-stream drift (Round 21 fix). For each stream, compute the W1
    # between first-half and second-half of its ts_delta / obj_size values,
    # normalize by that stream's median-abs-value, then mean across streams.
    per_stream_drift_ts_raw: list = []
    per_stream_drift_ts_norm: list = []
    per_stream_drift_size_raw: list = []
    per_stream_drift_size_norm: list = []

    has_ts = 'ts' in df.columns
    has_size = 'obj_size' in df.columns
    for _s, g in df.groupby('stream_id', sort=True):
        if has_ts:
            d = np.diff(g['ts'].astype(np.float64).values)
            if len(d) >= 2:
                w1, scale = _half_drift(d)
                per_stream_drift_ts_raw.append(w1)
                per_stream_drift_ts_norm.append(w1 / scale)
        if has_size:
            sz = g['obj_size'].astype(np.float64).values
            if len(sz) >= 2:
                w1, scale = _half_drift(sz)
                per_stream_drift_size_raw.append(w1)
                per_stream_drift_size_norm.append(w1 / scale)

    def _mean_or_zero(xs):
        return float(np.mean(xs)) if xs else 0.0

    drift_ts_raw_mean = _mean_or_zero(per_stream_drift_ts_raw)
    drift_ts_norm_mean = _mean_or_zero(per_stream_drift_ts_norm)
    drift_size_raw_mean = _mean_or_zero(per_stream_drift_size_raw)
    drift_size_norm_mean = _mean_or_zero(per_stream_drift_size_norm)

    total_records = int(sum(len(x) for x in per_stream))
    return {
        "n_records": total_records,
        "n_streams": len(per_stream),
        "footprint_mean_per_stream": float(np.mean(footprints)) if footprints else 0.0,
        "footprints_per_stream": footprints,
        "reuse_access_rate": float(np.mean(reuse_access_rates)) if reuse_access_rates else 0.0,
        "reuse_object_rate": float(np.mean(reuse_object_rates)) if reuse_object_rates else 0.0,
        "reuse_decile_cumulative_first": float(np.mean([d[0] for d in deciles_cumulative])) if deciles_cumulative else 0.0,
        "reuse_decile_cumulative_last": float(np.mean([d[-1] for d in deciles_cumulative])) if deciles_cumulative else 0.0,
        "reuse_decile_local_first": float(np.mean([d[0] for d in deciles_local])) if deciles_local else 0.0,
        "reuse_decile_local_last": float(np.mean([d[-1] for d in deciles_local])) if deciles_local else 0.0,
        "reuse_decile_local_drift": float(np.mean([d[-1] - d[0] for d in deciles_local])) if deciles_local else 0.0,
        "reuse_deciles_cumulative": [float(x) for x in np.mean(deciles_cumulative, axis=0)] if deciles_cumulative else [],
        "reuse_deciles_local": [float(x) for x in np.mean(deciles_local, axis=0)] if deciles_local else [],
        "ird_positional_median": ird_positional_median,
        "ird_positional_p90": ird_positional_p90,
        "ird_positional_histogram": ird_hist,
        "ird_positional_bin_edges": ird_bin_edges,
        "stack_distance_median": stack_median,
        "stack_distance_p90": stack_p90,
        "stack_distance_histogram": stack_hist,
        "stack_distance_bin_edges": stack_bin_edges,
        "hrc": hrc_avg.tolist(),
        "cache_sizes": cache_sizes.tolist(),
        # Per-stream averaged drift metrics. "_per_stream_mean" is the
        # average across streams of each stream's half-to-half W1.
        "drift_ts_delta_w1_per_stream_mean": drift_ts_raw_mean,
        "drift_ts_delta_w1_normalized_per_stream_mean": drift_ts_norm_mean,
        "drift_obj_size_w1_per_stream_mean": drift_size_raw_mean,
        "drift_obj_size_w1_normalized_per_stream_mean": drift_size_norm_mean,
        "drift_ts_delta_w1_per_stream": [float(x) for x in per_stream_drift_ts_norm],
        "drift_obj_size_w1_per_stream": [float(x) for x in per_stream_drift_size_norm],
    }


def _gap(fake: dict, real: dict) -> dict:
    """Compute fake-vs-real gaps for a small set of headline metrics.
    Drift ratios use the scale-normalized Wasserstein-1 so they are
    dimensionless and comparable across corpora with different value
    magnitudes."""
    hrc_fake = np.asarray(fake.get("hrc", []))
    hrc_real = np.asarray(real.get("hrc", []))
    n = min(len(hrc_fake), len(hrc_real))
    hrc_mae = float(np.abs(hrc_fake[:n] - hrc_real[:n]).mean()) if n else None

    def _ratio(f, r):
        return (f / r) if (r is not None and r > 0) else None

    return {
        "hrc_mae": hrc_mae,
        "reuse_access_rate_delta": fake["reuse_access_rate"] - real["reuse_access_rate"],
        "reuse_object_rate_delta": fake["reuse_object_rate"] - real["reuse_object_rate"],
        "reuse_decile_local_drift_fake_minus_real":
            fake["reuse_decile_local_drift"] - real["reuse_decile_local_drift"],
        "drift_ts_delta_w1_ratio":
            _ratio(fake["drift_ts_delta_w1_normalized_per_stream_mean"],
                   real["drift_ts_delta_w1_normalized_per_stream_mean"]),
        "drift_obj_size_w1_ratio":
            _ratio(fake["drift_obj_size_w1_normalized_per_stream_mean"],
                   real["drift_obj_size_w1_normalized_per_stream_mean"]),
        "ird_positional_median_ratio":
            _ratio(fake["ird_positional_median"], real["ird_positional_median"]),
        "ird_positional_p90_ratio":
            _ratio(fake["ird_positional_p90"], real["ird_positional_p90"]),
        "stack_distance_median_ratio":
            _ratio(fake["stack_distance_median"], real["stack_distance_median"]),
        "stack_distance_p90_ratio":
            _ratio(fake["stack_distance_p90"], real["stack_distance_p90"]),
        "footprint_ratio":
            _ratio(fake["footprint_mean_per_stream"], real["footprint_mean_per_stream"]),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint.")
    p.add_argument("--trace-dir", required=True, help="Real trace directory.")
    p.add_argument("--fmt", required=True, help="Trace format (e.g. oracle_general).")
    p.add_argument("--n-records", type=int, default=100_000,
                   help="Number of records in the long rollout (default 100000).")
    p.add_argument("--n-streams", type=int, default=4,
                   help="Number of parallel streams (default 4).")
    p.add_argument("--seed", type=int, default=42,
                   help="Deterministic seed for torch/numpy/random (default 42).")
    p.add_argument("--cache-sizes", type=str, default="",
                   help="Comma-separated override cache sizes. Default = log-spaced "
                        "across 1..footprint, 20 points.")
    p.add_argument("--output", default="",
                   help="Override output JSON path. Default: <ckpt-parent>/"
                        "long_rollout_<ckpt-stem>.json.")
    # --- Round 21 P1 sidecar flags ---
    p.add_argument("--char-file", default="",
                   help="Path to trace_characterizations.jsonl. If provided, "
                        "conditioning is drawn from this pool (matches "
                        "eval.py / generate.py). Required for conditional "
                        "checkpoints unless --random-conditioning is set.")
    p.add_argument("--source-traces", default="",
                   help="Comma-separated list of source trace basenames; their "
                        "characterization vectors (from --char-file) feed per-"
                        "stream conditioning in order. If fewer names than "
                        "n_streams are given, the list is cycled. Requires "
                        "--char-file.")
    p.add_argument("--random-conditioning", action="store_true",
                   help="Explicitly opt into random N(0,0.5) conditioning for "
                        "conditional checkpoints. Without this flag (and "
                        "without --char-file), the sidecar refuses to run on "
                        "conditional checkpoints to avoid silent "
                        "train/eval/generate descriptor mismatch.")
    p.add_argument("--real-manifest", default="",
                   help="Path to a real-baseline manifest JSON. If the file "
                        "exists, the real baseline is replayed from it "
                        "verbatim. If it does not exist, a new manifest is "
                        "written at the end of the run. Empty = do not "
                        "read/write a manifest.")
    p.add_argument("--retrieval-persist-across-windows", action="store_true",
                   help="IDEA #28: thread the retrieval-memory bank (K/V/T/mask) "
                        "across window boundaries during the long rollout, "
                        "instead of re-initialising each window. No-op if the "
                        "checkpoint has no retrieval memory.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    _seed_everything(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[error] checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 2

    print(f"[long_rollout_eval] device={device}  seed={args.seed}")
    print(f"[long_rollout_eval] checkpoint={ckpt_path}")

    t0 = time.time()
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    source_traces = [s.strip() for s in args.source_traces.split(",") if s.strip()]
    if source_traces and not args.char_file:
        print("[error] --source-traces requires --char-file", file=sys.stderr)
        return 2

    print(f"[long_rollout_eval] generating {args.n_records:,} records "
          f"across {args.n_streams} stream(s) …")
    fake_df, cond_meta = _rollout(
        ckpt, args.n_records, args.n_streams, device,
        char_file=args.char_file,
        source_traces=source_traces,
        random_conditioning=args.random_conditioning,
        retrieval_persist=args.retrieval_persist_across_windows,
    )
    print(f"[long_rollout_eval] conditioning: {cond_meta.get('source')}")
    t_fake = time.time() - t0

    print(f"[long_rollout_eval] sampling real trace baseline …")
    if args.real_manifest and Path(args.real_manifest).exists():
        print(f"[long_rollout_eval] replaying real manifest: {args.real_manifest}")
    real_df, real_manifest = _sample_real_stream(
        args.trace_dir, args.fmt, args.n_records,
        args.n_streams, args.seed, manifest_path=args.real_manifest,
    )
    t_real = time.time() - t0 - t_fake

    # Determine cache sizes from the mean per-stream real footprint (so cache
    # scales make sense for an N-stream comparison rather than N× the mean).
    real_per_stream = _per_stream_obj_ids(real_df)
    real_footprint = int(np.mean([np.unique(s).size for s in real_per_stream])) \
        if real_per_stream else 0
    if args.cache_sizes:
        cache_sizes = np.array([int(s.strip()) for s in args.cache_sizes.split(",") if s.strip()],
                               dtype=np.int64)
    else:
        cache_sizes = np.unique(np.geomspace(
            max(1, real_footprint // 1000), max(real_footprint, 2), 20
        ).astype(np.int64))
    print(f"[long_rollout_eval] footprint (real)={real_footprint}  "
          f"cache_sizes=[{cache_sizes.min()}…{cache_sizes.max()}] "
          f"({len(cache_sizes)} points)")

    fake_m = _metrics_for_stream(fake_df, cache_sizes)
    real_m = _metrics_for_stream(real_df, cache_sizes)
    gap_m = _gap(fake_m, real_m)

    result = {
        "checkpoint": str(ckpt_path),
        "trace_dir": args.trace_dir,
        "fmt": args.fmt,
        "seed": args.seed,
        "n_records": args.n_records,
        "n_streams": args.n_streams,
        "rollout_time_sec": t_fake,
        "real_sample_time_sec": t_real,
        "conditioning": cond_meta,
        "real_manifest_path": args.real_manifest,
        "real_manifest": real_manifest,
        "fake": fake_m,
        "real": real_m,
        "gap": gap_m,
    }

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = ckpt_path.parent / f"long_rollout_{ckpt_path.stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    # Terminal summary.
    print()
    print("─" * 68)
    print(f"{'metric':<32}{'fake':>14}{'real':>14}{'gap':>8}")
    print("─" * 68)
    def _row(name, f_val, r_val, fmt="{:>14.4f}"):
        gap_txt = ""
        if isinstance(f_val, (int, float)) and isinstance(r_val, (int, float)) and r_val:
            gap_txt = f"{(f_val - r_val) / r_val * 100:+7.1f}%"
        f_txt = fmt.format(f_val) if isinstance(f_val, (int, float)) else f"{f_val:>14}"
        r_txt = fmt.format(r_val) if isinstance(r_val, (int, float)) else f"{r_val:>14}"
        print(f"{name:<32}{f_txt}{r_txt}  {gap_txt}")
    _row("reuse_access_rate", fake_m["reuse_access_rate"], real_m["reuse_access_rate"])
    _row("reuse_object_rate", fake_m["reuse_object_rate"], real_m["reuse_object_rate"])
    _row("reuse_decile_local_first", fake_m["reuse_decile_local_first"], real_m["reuse_decile_local_first"])
    _row("reuse_decile_local_last", fake_m["reuse_decile_local_last"], real_m["reuse_decile_local_last"])
    _row("reuse_decile_local_drift", fake_m["reuse_decile_local_drift"], real_m["reuse_decile_local_drift"])
    _row("ird_positional_median", fake_m["ird_positional_median"], real_m["ird_positional_median"], fmt="{:>14.0f}")
    _row("ird_positional_p90", fake_m["ird_positional_p90"], real_m["ird_positional_p90"], fmt="{:>14.0f}")
    _row("stack_distance_median", fake_m["stack_distance_median"], real_m["stack_distance_median"], fmt="{:>14.0f}")
    _row("stack_distance_p90", fake_m["stack_distance_p90"], real_m["stack_distance_p90"], fmt="{:>14.0f}")
    _row("drift_ts_delta_w1_norm",
         fake_m["drift_ts_delta_w1_normalized_per_stream_mean"],
         real_m["drift_ts_delta_w1_normalized_per_stream_mean"])
    _row("drift_obj_size_w1_norm",
         fake_m["drift_obj_size_w1_normalized_per_stream_mean"],
         real_m["drift_obj_size_w1_normalized_per_stream_mean"])
    _row("footprint_per_stream", fake_m["footprint_mean_per_stream"], real_m["footprint_mean_per_stream"], fmt="{:>14.0f}")
    print("─" * 68)
    print(f"HRC-MAE (mean across streams) : {gap_m['hrc_mae']:.4f}" if gap_m['hrc_mae'] else
          "HRC-MAE: unavailable")
    print(f"json     → {out_path}")
    print("─" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
