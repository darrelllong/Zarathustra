#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from parsers import (
    TraceIdentity,
    canonical_identity_for_path,
    feature_hints_for_profile,
    ml_use_case_for_profile,
    parse_trace,
)


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(sanitize_nonfinite(row), sort_keys=True, allow_nan=False) + "\n")


def sanitize_nonfinite(value):
    if isinstance(value, dict):
        return {key: sanitize_nonfinite(val) for key, val in value.items()}
    if isinstance(value, list):
        return [sanitize_nonfinite(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_nonfinite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def canonicalize_row(row: Dict[str, object]) -> TraceIdentity:
    return canonical_identity_for_path(str(row["path"]))


def needs_reparse(row: Dict[str, object], identity: TraceIdentity) -> bool:
    profile = row.get("profile") or {}
    parser = str(profile.get("parser", ""))
    old_fmt = str(row.get("format", ""))
    old_family = str(row.get("family", ""))
    if old_fmt != identity.format:
        return True
    if old_family != identity.family:
        return parser in {"generic_text", "parquet_stub", "lcs_stub", "error", "unsupported", "parquet_unavailable"}
    if parser in {"parquet_stub", "lcs_stub", "error", "unsupported", "parquet_unavailable", "lcs_error"}:
        return True
    if parser == "generic_text" and identity.format in {"oracle_general", "lcs"}:
        return True
    if identity.family in {"2007_wiki", "2016_wiki", "2017_systor", "2023_metaCDN", "2023_metaStorage", "2020_twr_cdn"}:
        return True
    return False


def build_summary(rows: List[Dict[str, object]]) -> Dict[str, object]:
    by_dataset = defaultdict(lambda: {"files": 0, "trace_files": 0, "bytes": 0})
    by_format = Counter()
    by_role = Counter()
    total_bytes = 0
    for row in rows:
        total_bytes += int(row["size_bytes"])
        by_format[str(row["format"])] += 1
        by_role[str(row["role"])] += 1
        ds = by_dataset[str(row["dataset"])]
        ds["files"] += 1
        ds["bytes"] += int(row["size_bytes"])
        if row.get("is_trace"):
            ds["trace_files"] += 1
    return {
        "root": "/tiamat/zarathustra/traces",
        "total_files": len(rows),
        "total_bytes": total_bytes,
        "by_role": dict(by_role),
        "by_format": dict(by_format),
        "by_dataset": dict(sorted(by_dataset.items())),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Normalize and selectively repair trace characterizations")
    ap.add_argument("--inventory", default="/tiamat/zarathustra/analysis/out/trace_inventory.jsonl")
    ap.add_argument("--characterizations", default="/tiamat/zarathustra/analysis/out/trace_characterizations.jsonl")
    ap.add_argument("--out-dir", default="/tiamat/zarathustra/r-output/normalized")
    ap.add_argument("--sample-records", type=int, default=4096)
    args = ap.parse_args()

    inventory_rows = load_jsonl(Path(args.inventory))
    char_rows = load_jsonl(Path(args.characterizations))
    char_by_rel = {str(row.get("rel_path", "")): row for row in char_rows}

    normalized_inventory: List[Dict[str, object]] = []
    normalized_chars: List[Dict[str, object]] = []
    reparsed = 0

    for inv_row in inventory_rows:
        identity = canonicalize_row(inv_row)
        inv_copy = dict(inv_row)
        inv_copy["dataset"] = identity.dataset
        inv_copy["family"] = identity.family
        inv_copy["format"] = identity.format
        normalized_inventory.append(inv_copy)

        if not inv_copy.get("is_trace"):
            continue

        existing = char_by_rel.get(str(inv_copy.get("rel_path", "")))
        if existing is None:
            parsed = parse_trace(identity.path, max_records=args.sample_records)
            repaired_row = dict(inv_copy)
            repaired_row["analysis_sample_records"] = args.sample_records
            repaired_row["profile"] = parsed.profile
            repaired_row["ml_use_case"] = ml_use_case_for_profile(parsed.identity, parsed.profile)
            repaired_row["feature_hints"] = feature_hints_for_profile(parsed.identity, parsed.profile)
            normalized_chars.append(repaired_row)
            reparsed += 1
            continue

        char_copy = dict(existing)
        char_copy["dataset"] = identity.dataset
        char_copy["family"] = identity.family
        char_copy["format"] = identity.format
        if needs_reparse(char_copy, identity):
            parsed = parse_trace(identity.path, max_records=args.sample_records)
            char_copy["profile"] = parsed.profile
            char_copy["analysis_sample_records"] = args.sample_records
            reparsed += 1
        char_copy["ml_use_case"] = ml_use_case_for_profile(identity, char_copy["profile"])
        char_copy["feature_hints"] = feature_hints_for_profile(identity, char_copy["profile"])
        normalized_chars.append(char_copy)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(out_dir / "trace_inventory.normalized.jsonl", normalized_inventory)
    dump_jsonl(out_dir / "trace_characterizations.normalized.jsonl", normalized_chars)

    with open(out_dir / "trace_inventory_summary.normalized.json", "w", encoding="utf-8") as fh:
        json.dump(sanitize_nonfinite(build_summary(normalized_inventory)), fh, indent=2, sort_keys=True, allow_nan=False)

    parser_counts = Counter(str(row.get("profile", {}).get("parser", "unknown")) for row in normalized_chars)
    summary = {
        "inventory_rows": len(normalized_inventory),
        "trace_rows": sum(1 for row in normalized_inventory if row.get("is_trace")),
        "characterized_rows": len(normalized_chars),
        "reparsed_rows": reparsed,
        "parser_counts": dict(parser_counts),
        "logical_families": sorted({(row["dataset"], row["family"]) for row in normalized_chars}),
    }
    with open(out_dir / "normalize_summary.json", "w", encoding="utf-8") as fh:
        json.dump(sanitize_nonfinite(summary), fh, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps(sanitize_nonfinite(summary), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
