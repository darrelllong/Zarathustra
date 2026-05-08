"""Run cachesim on a footprint-scaled cache-size ladder.

This is LANL-owned methodology tooling. It leaves `llgan.cachesim_eval`
unchanged, computes the distinct-object footprint from the real reference CSV,
then invokes the official evaluator with powers-of-two cache sizes from 1 up
to `2^ceil(log2(N + 1))`.  For CSV traces, the default footprint key matches
`tools/cachesim`: `(stream_id, obj_id)` when a stream-like column is present,
otherwise `obj_id`.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import math
import subprocess
import sys
from pathlib import Path
from typing import Iterable, TextIO


DEFAULT_POLICIES = "lru,arc,fifo,sieve,slru,car"
DEFAULT_OBJ_ID_COLUMNS = ("obj_id", "object_id", "key", "block_id")
DEFAULT_STREAM_ID_COLUMNS = ("stream_id", "stream", "tenant")


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="")
    if path.suffix == ".zst":
        try:
            import zstandard as zstd  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Reading .zst references requires the optional `zstandard` package"
            ) from exc
        raw = path.open("rb")
        reader = zstd.ZstdDecompressor().stream_reader(raw)
        return io.TextIOWrapper(reader, newline="")
    return path.open("rt", newline="")


def _choose_obj_id_column(fieldnames: Iterable[str], requested: str | None) -> str:
    fields = list(fieldnames)
    if requested:
        if requested not in fields:
            raise ValueError(f"requested obj-id column {requested!r} not in CSV header {fields!r}")
        return requested
    for name in DEFAULT_OBJ_ID_COLUMNS:
        if name in fields:
            return name
    for name in fields:
        lowered = name.lower()
        if "obj" in lowered and "id" in lowered:
            return name
    raise ValueError(
        "could not infer object-id column; pass --obj-id-column. "
        f"CSV header was {fields!r}"
    )


def _choose_stream_id_column(
    fieldnames: Iterable[str],
    requested: str | None,
    key_mode: str,
) -> str | None:
    fields = list(fieldnames)
    if key_mode == "obj_id":
        return None
    if requested:
        if requested not in fields:
            raise ValueError(
                f"requested stream-id column {requested!r} not in CSV header {fields!r}"
            )
        return requested
    for name in DEFAULT_STREAM_ID_COLUMNS:
        if name in fields:
            return name
    if key_mode == "stream_obj":
        raise ValueError(
            "could not infer stream-id column for --key-mode stream_obj; "
            f"CSV header was {fields!r}"
        )
    return None


def real_footprint(
    path: Path,
    obj_id_column: str | None = None,
    stream_id_column: str | None = None,
    key_mode: str = "auto",
) -> tuple[int, str]:
    """Return exact distinct cache-key count and the column(s) used."""
    with _open_text(path) as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} does not look like a headered CSV")
        obj_column = _choose_obj_id_column(reader.fieldnames, obj_id_column)
        stream_column = _choose_stream_id_column(reader.fieldnames, stream_id_column, key_mode)
        seen: set[str | tuple[str, str]] = set()
        for row in reader:
            obj_value = row.get(obj_column)
            if obj_value is None or obj_value == "":
                continue
            if stream_column is None:
                seen.add(obj_value)
            else:
                stream_value = row.get(stream_column)
                seen.add((stream_value or "0", obj_value))
    key_desc = obj_column if stream_column is None else f"{stream_column},{obj_column}"
    return len(seen), key_desc


def cache_sizes_for_footprint(footprint: int, min_cache_size: int = 1) -> list[int]:
    """Powers of two from `min_cache_size` through `2^ceil(log2(N + 1))`."""
    if footprint < 0:
        raise ValueError("footprint must be non-negative")
    if min_cache_size < 1:
        raise ValueError("min cache size must be >= 1")
    max_size = 1 if footprint == 0 else 1 << math.ceil(math.log2(footprint + 1))
    sizes: list[int] = []
    size = 1
    while size <= max_size:
        if size >= min_cache_size:
            sizes.append(size)
        size *= 2
    return sizes


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run llgan.cachesim_eval with a real-footprint power-of-two ladder."
    )
    parser.add_argument("--fake", required=True, help="Synthetic CSV/zst trace")
    parser.add_argument("--real", required=True, help="Real reference CSV/zst trace")
    parser.add_argument(
        "--policies",
        default=DEFAULT_POLICIES,
        help=f"Comma-separated policies (default {DEFAULT_POLICIES})",
    )
    parser.add_argument(
        "--obj-id-column",
        default=None,
        help="Column used for exact real-trace footprint (default: infer from header)",
    )
    parser.add_argument(
        "--stream-id-column",
        default=None,
        help="Stream column for exact real-trace footprint (default: infer when present)",
    )
    parser.add_argument(
        "--key-mode",
        choices=("auto", "obj_id", "stream_obj"),
        default="auto",
        help=(
            "Footprint key mode: auto matches tools/cachesim CSV behavior, "
            "obj_id ignores streams, stream_obj requires a stream column."
        ),
    )
    parser.add_argument(
        "--min-cache-size",
        type=int,
        default=1,
        help="Smallest power-of-two cache size to include (default 1)",
    )
    parser.add_argument("--out", default="-", help="JSON report path for llgan.cachesim_eval")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the inferred ladder and evaluator command without running cachesim",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    real_path = Path(args.real)
    footprint, column = real_footprint(
        real_path,
        obj_id_column=args.obj_id_column,
        stream_id_column=args.stream_id_column,
        key_mode=args.key_mode,
    )
    sizes = cache_sizes_for_footprint(footprint, args.min_cache_size)
    size_arg = ",".join(str(size) for size in sizes)
    command = [
        sys.executable,
        "-m",
        "llgan.cachesim_eval",
        "--fake",
        args.fake,
        "--real",
        args.real,
        "--cache-sizes",
        size_arg,
        "--policies",
        args.policies,
        "--out",
        args.out,
    ]

    print(f"real footprint: {footprint} distinct objects (column {column})")
    print(f"cache sizes: {size_arg}")
    print("+ " + " ".join(command))
    if args.dry_run:
        return 0
    completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
