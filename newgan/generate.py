"""
Minimal CLI to copy a trace without modification.

The original ``newgan/generate.py`` copied the trace file to a new location.
Because the traces are read‑only on ``/tiamat`` we now simply provide a
wrapper that reads the trace and writes it verbatim to an output path.

This keeps the interface stable for users who already call
``newgan/generate``.
"""

import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Copy a trace without changes")
    parser.add_argument("--trace", required=True, help="Input trace on /tiamat")
    parser.add_argument("--output", required=True, help="Where to write the copy")
    parser.add_argument("--cond", action="store_true", help="Print conditioning vector if available")
    args = parser.parse_args()

    # Load as CSV; for binary formats users can adjust the reader.
    df = pd.read_csv(args.trace)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Copied trace to {out_path}")

    if args.cond:
        # If we have the trace_characterizations.jsonl, show the vector
        try:
            from llgan.dataset import load_file_characterizations, profile_to_cond_vector
            import json
            from pathlib import Path

            jsonl = Path("/Users/darrell/Zarathustra/traces/characterization/trace_characterizations.jsonl")
            cond_lookup = load_file_characterizations(str(jsonl))
            vec = cond_lookup.get(out_path.name)
            if vec is not None:
                print("Conditioning vector:", vec.tolist())
            else:
                print("No conditioning vector found for this file")
        except Exception as e:
            print("Could not compute conditioning vector:", e)


if __name__ == "__main__":
    main()
