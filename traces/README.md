# Trace Analysis Artifacts

This directory mirrors the remote trace-analysis work from `vinge.local`.

Contents:
- `analysis/`: the Python and shell scripts used to inventory and characterize the trace corpus under `/tiamat/zarathustra/traces`
- `characterization/`: the generated outputs copied from `/tiamat/zarathustra/analysis/out`

Key outputs:
- `characterization/ML_PRIMING_SUMMARY.md`: human-readable summary
- `characterization/trace_rollup.json`: grouped rollup data
- `characterization/trace_inventory_summary.json`: high-level inventory counts
- `characterization/trace_inventory.jsonl`: per-file inventory
- `characterization/trace_characterizations.jsonl`: per-trace characterization records
