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
from pathlib import Path


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


def _replace_between_markers(*, text: str, begin: str, end: str, replacement: str) -> str:
    """Replace the content between two unique marker lines (inclusive markers remain)."""
    begin_idx = text.find(begin)
    end_idx = text.find(end)
    if begin_idx == -1 or end_idx == -1:
        missing = []
        if begin_idx == -1:
            missing.append("begin")
        if end_idx == -1:
            missing.append("end")
        raise ValueError(f"Missing {', '.join(missing)} marker(s): {begin!r} / {end!r}")
    if end_idx < begin_idx:
        raise ValueError(f"End marker appears before begin marker: {begin!r} / {end!r}")

    after_begin = begin_idx + len(begin)
    new_text = (
        text[:after_begin]
        + "\n"
        + replacement.rstrip()
        + "\n"
        + text[end_idx:]
    )
    return new_text


def _default_repo_root() -> Path:
    # altgan/<this_file>.py -> repo root = parent of altgan/
    return Path(__file__).resolve().parents[1]


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
        "--emit-markdown-dir",
        default=None,
        help="If set, write one paste-ready Markdown snippet per corpus into this directory.",
    )
    p.add_argument(
        "--emit-summary-json-dir",
        default=None,
        help="If set, write one machine-readable JSON summary per corpus into this directory.",
    )
    p.add_argument(
        "--update-lanl-docs",
        action="store_true",
        help=(
            "After each corpus completes, update RESPONSE-LANL.md and altgan/RESULTS.md "
            "by replacing the corresponding TRACEBOOTSTRAP_SHUFFLE_* marker blocks "
            "with the emitted per-corpus Markdown snippet."
        ),
    )
    p.add_argument(
        "--response-lanl-md",
        default=None,
        help="Path to RESPONSE-LANL.md (default: auto-detect from repo root).",
    )
    p.add_argument(
        "--altgan-results-md",
        default=None,
        help="Path to altgan/RESULTS.md (default: auto-detect from repo root).",
    )
    p.add_argument(
        "--markdown",
        action="store_true",
        help="Forward `--emit-markdown` to the underlying multi-seed runner.",
    )
    p.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue other corpora if one fails; report failures at the end.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Forward `--skip-existing` to the underlying multi-seed runner.",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    corpora = [c.strip() for c in args.corpora.split(",") if c.strip()]
    unknown = [c for c in corpora if c not in _PRESETS]
    if unknown:
        raise SystemExit(f"Unknown corpora: {unknown}. Choices: {sorted(_PRESETS)}")

    output_root = Path(args.output_root)
    md_dir = Path(args.emit_markdown_dir) if args.emit_markdown_dir else None
    json_dir = Path(args.emit_summary_json_dir) if args.emit_summary_json_dir else None

    if args.update_lanl_docs:
        # Ensure we always emit paste-ready snippets where the doc updater can find them.
        if md_dir is None:
            md_dir = output_root / "paste_ready"
        if json_dir is None:
            json_dir = output_root / "paste_ready"
        args.markdown = True

    if not args.dry_run:
        if md_dir:
            md_dir.mkdir(parents=True, exist_ok=True)
        if json_dir:
            json_dir.mkdir(parents=True, exist_ok=True)

    response_path = None
    results_path = None
    if args.update_lanl_docs:
        repo_root = _default_repo_root()
        response_path = Path(args.response_lanl_md) if args.response_lanl_md else (repo_root / "RESPONSE-LANL.md")
        results_path = (
            Path(args.altgan_results_md) if args.altgan_results_md else (repo_root / "altgan" / "RESULTS.md")
        )

    failures: list[tuple[str, int]] = []
    for corpus in corpora:
        preset = _PRESETS[corpus]
        md_path = md_dir / f"tracebootstrap_shuffle_{preset.corpus}.md" if md_dir else None
        summary_path = json_dir / f"tracebootstrap_shuffle_{preset.corpus}.json" if json_dir else None
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
        if md_path:
            cmd.extend(["--emit-markdown-to", str(md_path)])
        if summary_path:
            cmd.extend(["--emit-summary-json-to", str(summary_path)])
        if args.skip_existing:
            cmd.append("--skip-existing")
        if args.dry_run:
            cmd.append("--dry-run")

        print(f"\n=== TRACEBOOTSTRAP SHUFFLE PACK: {corpus} ===", flush=True)
        try:
            subprocess.run(cmd, check=True)
            if args.update_lanl_docs and not args.dry_run:
                if md_path is None:
                    raise SystemExit("--update-lanl-docs requires an emit-markdown destination.")
                if not md_path.exists():
                    raise SystemExit(f"Expected Markdown snippet not found: {md_path}")
                snippet = md_path.read_text()
                marker = preset.corpus.upper()
                begin = f"<!-- BEGIN TRACEBOOTSTRAP_SHUFFLE_{marker} -->"
                end = f"<!-- END TRACEBOOTSTRAP_SHUFFLE_{marker} -->"

                for doc_path in (response_path, results_path):
                    if doc_path is None:
                        continue
                    doc_text = doc_path.read_text()
                    new_text = _replace_between_markers(
                        text=doc_text,
                        begin=begin,
                        end=end,
                        replacement=snippet,
                    )
                    if new_text != doc_text:
                        doc_path.write_text(new_text)
                        print(f"[docs] Updated: {doc_path}", flush=True)
        except subprocess.CalledProcessError as e:
            print(
                f"\n[error] TraceBootstrap shuffle pack failed for corpus '{corpus}' "
                f"(exit {e.returncode}).",
                file=sys.stderr,
                flush=True,
            )
            if not args.keep_going:
                return int(e.returncode)
            failures.append((corpus, int(e.returncode)))

    if failures:
        print("\n[error] TraceBootstrap shuffle pack failures:", file=sys.stderr, flush=True)
        for corpus, code in failures:
            print(f"- {corpus}: exit {code}", file=sys.stderr, flush=True)
        return failures[0][1] if failures[0][1] != 0 else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
