"""Batch launcher for TraceBootstrap shuffle multi-seed panels.

This wraps `altgan.launch_trace_bootstrap_multiseed` with pinned corpus presets
so LANL can publish/refresh the TraceBootstrap (shuffle) ledger rows on the
official `llgan.cachesim_eval` surface with one command.

Target (1M corpora): shuffle @ chunk_size=65536, 4 seeds {42,80,81,82}.
Tencent remains special (100k, chunk_size=8192) and is not handled here.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class CorpusPreset:
    corpus: str
    trace_dir: str
    fmt: str
    real_manifest: str
    real_ref: str
    n_records: int = 1_000_000
    n_streams: int = 4
    mode: str = "shuffle"
    chunk_size: int = 65_536
    cache_sizes: str = "32,128,512,2048,8192"
    policies: str = "lru,arc,fifo,sieve,slru,car"


_PRESETS: dict[str, CorpusPreset] = {
    # Trace dirs are only used when sampling (no manifest) or as a fallback when
    # manifests use relative paths; on /tiamat manifests typically pin absolute
    # file paths, so these are safe defaults.
    "twitter": CorpusPreset(
        corpus="twitter",
        trace_dir="/tiamat/zarathustra/traces/twitter_cluster",
        fmt="oracle_general",
        real_manifest="/tiamat/zarathustra/llgan-output/manifests/twitter_cluster_stackatlas.json",
        real_ref="/tiamat/zarathustra/llgan-output/refs/twitter_cluster_real.csv",
    ),
    "metakv": CorpusPreset(
        corpus="metakv",
        trace_dir="/tiamat/zarathustra/traces",
        fmt="oracle_general",
        real_manifest="/tiamat/zarathustra/llgan-output/manifests/metakv_stackatlas.json",
        real_ref="/tiamat/zarathustra/llgan-output/refs/metakv_real.csv",
    ),
    "metacdn": CorpusPreset(
        corpus="metacdn",
        trace_dir="/tiamat/zarathustra/traces",
        fmt="oracle_general",
        real_manifest="/tiamat/zarathustra/llgan-output/manifests/metacdn_stackatlas.json",
        real_ref="/tiamat/zarathustra/llgan-output/refs/metacdn_real.csv",
    ),
    "wiki": CorpusPreset(
        corpus="wiki",
        trace_dir="/tiamat/zarathustra/traces/s3-cache-datasets/cache_dataset_oracleGeneral/2019_wiki",
        fmt="oracle_general",
        real_manifest="/tiamat/zarathustra/llgan-output/manifests/wiki_stackatlas.json",
        real_ref="/tiamat/zarathustra/llgan-output/refs/wiki_real.csv",
    ),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--corpora",
        default="twitter,metakv,metacdn,wiki",
        help=f"Comma-separated subset of: {','.join(sorted(_PRESETS))}",
    )
    p.add_argument("--seeds", default="42,80,81,82")
    p.add_argument("--output-root", default="/tiamat/zarathustra/altgan-output")
    p.add_argument(
        "--markdown",
        action="store_true",
        help="Forward `--emit-markdown` to the underlying multi-seed runner.",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    corpora = [c.strip() for c in args.corpora.split(",") if c.strip()]
    unknown = [c for c in corpora if c not in _PRESETS]
    if unknown:
        raise SystemExit(f"Unknown corpora: {unknown}. Choices: {sorted(_PRESETS)}")

    for corpus in corpora:
        preset = _PRESETS[corpus]
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "altgan.launch_trace_bootstrap_multiseed",
            "--corpus",
            preset.corpus,
            "--trace-dir",
            preset.trace_dir,
            "--fmt",
            preset.fmt,
            "--real-manifest",
            preset.real_manifest,
            "--real-ref",
            preset.real_ref,
            "--seeds",
            args.seeds,
            "--n-records",
            str(preset.n_records),
            "--n-streams",
            str(preset.n_streams),
            "--mode",
            preset.mode,
            "--chunk-size",
            str(preset.chunk_size),
            "--cache-sizes",
            preset.cache_sizes,
            "--policies",
            preset.policies,
            "--output-root",
            args.output_root,
        ]
        if args.markdown:
            cmd.append("--emit-markdown")
        if args.dry_run:
            cmd.append("--dry-run")

        print(f"\n=== TRACEBOOTSTRAP SHUFFLE PACK: {corpus} ===", flush=True)
        subprocess.run(cmd, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
