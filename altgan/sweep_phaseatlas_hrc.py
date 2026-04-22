"""Run HRC-targeted PhaseAtlas calibration sweeps.

This wrapper keeps the object-process comparison disciplined: every cell calls
``altgan.evaluate_neural_atlas`` with the same real manifest, summarizes cache
and mark metrics, and ranks by HRC first. It is intended for quick remote
launches where interrupted sweeps must be resumable.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--trace-dir", required=True)
    p.add_argument("--fmt", required=True)
    p.add_argument("--char-file", required=True)
    p.add_argument("--real-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--prefix", default="phaseatlas_hrc")
    p.add_argument("--cond-dim", type=int, default=13)
    p.add_argument("--panels", default="4x100000,8x50000",
                   help="Comma-separated n_streams x n_records panels.")
    p.add_argument("--seeds", default="42")
    p.add_argument("--temperatures", default="1.0")
    p.add_argument("--transition-blends", default="0.0,0.25,0.5,0.75,1.0")
    p.add_argument("--phase-modes", default="natural,forced",
                   help="Comma-separated phase modes: natural, forced.")
    p.add_argument("--disable-neural-marks", action="store_true",
                   help="Force reservoir marks when the checkpoint has a mark head.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Reuse existing eval JSONs instead of rerunning completed cells.")
    p.add_argument("--summary-csv", default="")
    p.add_argument("--best-json", default="")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for panel in _split_panels(args.panels):
        n_streams, n_records = panel
        for seed in _split_int(args.seeds):
            for temp in _split_float(args.temperatures):
                for blend in _split_float(args.transition_blends):
                    for phase_mode in _split_str(args.phase_modes):
                        force_phase = _parse_phase_mode(phase_mode)
                        output = out_dir / _label(
                            args.prefix,
                            n_streams=n_streams,
                            n_records=n_records,
                            seed=seed,
                            temp=temp,
                            blend=blend,
                            force_phase=force_phase,
                        )
                        _run_eval(
                            args,
                            output,
                            n_streams=n_streams,
                            n_records=n_records,
                            seed=seed,
                            temp=temp,
                            blend=blend,
                            force_phase=force_phase,
                        )
                        rows.append(_summarize(output, n_streams, n_records, seed, temp, blend, force_phase))

    summary_path = Path(args.summary_csv) if args.summary_csv else out_dir / f"{args.prefix}_summary.csv"
    _write_summary(summary_path, rows)
    print(f"[altgan.sweep_phaseatlas_hrc] wrote {summary_path}", flush=True)
    best_path = Path(args.best_json) if args.best_json else out_dir / f"{args.prefix}_best.json"
    _write_best(best_path, rows)
    print(f"[altgan.sweep_phaseatlas_hrc] wrote {best_path}", flush=True)
    return 0


def _run_eval(
    args: argparse.Namespace,
    output: Path,
    *,
    n_streams: int,
    n_records: int,
    seed: int,
    temp: float,
    blend: float,
    force_phase: bool,
) -> None:
    if args.skip_existing and output.exists():
        print(f"[altgan.sweep_phaseatlas_hrc] reusing {output}", flush=True)
        return
    cmd = [
        sys.executable, "-u", "-m", "altgan.evaluate_neural_atlas",
        "--model", args.model,
        "--trace-dir", args.trace_dir,
        "--fmt", args.fmt,
        "--char-file", args.char_file,
        "--cond-dim", str(args.cond_dim),
        "--condition-from-real-manifest",
        "--transition-blend", str(blend),
        "--temperature", str(temp),
        "--n-records", str(n_records),
        "--n-streams", str(n_streams),
        "--seed", str(seed),
        "--real-manifest", args.real_manifest,
        "--output", str(output),
    ]
    if force_phase:
        cmd.append("--force-phase-schedule")
    if args.disable_neural_marks:
        cmd.append("--disable-neural-marks")
    print(f"[altgan.sweep_phaseatlas_hrc] running {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _summarize(
    path: Path,
    n_streams: int,
    n_records: int,
    seed: int,
    temp: float,
    blend: float,
    force_phase: bool,
) -> dict:
    data = json.loads(path.read_text())
    fake = data["fake"]
    real = data["real"]
    gap = data["gap"]
    mark = data.get("mark_quality", {})
    return {
        "path": str(path),
        "n_streams": n_streams,
        "n_records": n_records,
        "seed": seed,
        "temperature": temp,
        "transition_blend": blend,
        "force_phase_schedule": force_phase,
        "uses_neural_marks": data.get("uses_neural_marks"),
        "hrc_mae": gap["hrc_mae"],
        "fake_reuse": fake["reuse_access_rate"],
        "real_reuse": real["reuse_access_rate"],
        "fake_stack_median": fake["stack_distance_median"],
        "real_stack_median": real["stack_distance_median"],
        "fake_stack_p90": fake["stack_distance_p90"],
        "real_stack_p90": real["stack_distance_p90"],
        "mark_score": mark.get("mark_score", ""),
    }


def _write_summary(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_best(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    best_hrc = min(rows, key=lambda r: float(r["hrc_mae"]))
    payload = {
        "best_hrc": best_hrc,
        "n_rows": len(rows),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _split_panels(text: str) -> list[tuple[int, int]]:
    panels = []
    for item in _split_str(text):
        try:
            streams_s, records_s = item.lower().split("x", 1)
            panels.append((int(streams_s), int(records_s)))
        except ValueError as exc:
            raise ValueError(f"invalid panel {item!r}; expected NxM") from exc
    return panels


def _parse_phase_mode(text: str) -> bool:
    mode = text.strip().lower()
    if mode == "natural":
        return False
    if mode == "forced":
        return True
    raise ValueError(f"unknown phase mode {text!r}; expected natural or forced")


def _split_float(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _split_int(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _split_str(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _slug(value: object) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def _label(
    prefix: str,
    *,
    n_streams: int,
    n_records: int,
    seed: int,
    temp: float,
    blend: float,
    force_phase: bool,
) -> str:
    phase = "forced" if force_phase else "natural"
    return (
        f"{prefix}_{n_streams}x{n_records}_seed-{seed}"
        f"_temp-{_slug(temp)}_blend-{_slug(blend)}_phase-{phase}_eval.json"
    )


if __name__ == "__main__":
    raise SystemExit(main())
