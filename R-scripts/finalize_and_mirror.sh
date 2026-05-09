#!/usr/bin/env bash
# After all extraction+grouping+verdicts are done, mirror final reports
# from /tiamat to /Volumes/Gigantor so the Mac copy is ready for hand-off.
#
# Run from gigantor (the Mac):  bash finalize_and_mirror.sh [APPEND_DIR]
set -euo pipefail
DEFAULT_APPEND="/tiamat/zarathustra/r-output/append_run_$(date +%Y%m%d)"
SRC=${1:-$DEFAULT_APPEND}
SRC_NAME=$(basename "$SRC")

DEST_GG_ZA="/Volumes/Gigantor/zarathustra/r-output/$SRC_NAME"
DEST_GG_TOP_REPORT="/Volumes/Gigantor/zarathustra/ZARATHUSTRA-EXTENDED-REPORT.md"
DEST_GG_BB="/Volumes/Gigantor/backblaze"

echo "Mirroring Zarathustra outputs $SRC -> $DEST_GG_ZA"
mkdir -p "$DEST_GG_ZA"
rsync -av --delete \
  vinge.local:"$SRC/" \
  "$DEST_GG_ZA/"

if [[ -f "$DEST_GG_ZA/ZARATHUSTRA-EXTENDED-REPORT.md" ]]; then
  cp -f "$DEST_GG_ZA/ZARATHUSTRA-EXTENDED-REPORT.md" "$DEST_GG_TOP_REPORT"
  echo "  top-level report -> $DEST_GG_TOP_REPORT"
fi

echo "Mirroring Backblaze outputs from /tiamat -> /Volumes/Gigantor/backblaze"
rsync -av \
  vinge.local:/tiamat/backblaze/REPORT.md \
  vinge.local:/tiamat/backblaze/analysis/moment_statistics_extended.csv \
  vinge.local:/tiamat/backblaze/analysis/spectral_failure_counts.csv \
  vinge.local:/tiamat/backblaze/analysis/periodicity_failure_counts.csv \
  vinge.local:/tiamat/backblaze/analysis/randomness_failure_counts.csv \
  vinge.local:/tiamat/backblaze/analysis/acf_daily_failures.csv \
  vinge.local:/tiamat/backblaze/analysis/model_clusters.csv \
  vinge.local:/tiamat/backblaze/analysis/model_cluster_diagnostics.csv \
  vinge.local:/tiamat/backblaze/analysis/model_cluster_gap_statistic.csv \
  "$DEST_GG_BB/" 2>/dev/null || true

# Make a small companion index in /Volumes/Gigantor/zarathustra
cat > /Volumes/Gigantor/zarathustra/HANDOFF-INDEX.md <<EOF
# Hand-off Index

Generated: $(date)

## Zarathustra extended trace analysis
- \`ZARATHUSTRA-EXTENDED-REPORT.md\` — top-level report
- \`r-output/$SRC_NAME/\` — full append-run directory
  - \`new_features.csv\` — per-trace extended features
  - \`per_trace/*.json\` — per-trace evidence
  - \`predictability_leaderboard.csv\` — predictability verdicts
  - \`predictability/*.json\` — per-trace verdict + evidence
  - \`trace_clusters.csv\`, \`family_cohesion.csv\` — grouping
  - \`sample_manifest.csv\` — which traces were sampled

## Backblaze drive-failure profile (extended)
- \`backblaze/REPORT.md\` — full report (original + appended sections)
- \`backblaze/analysis/moment_statistics_extended.csv\` — m7..m10 + cumulants
- \`backblaze/analysis/spectral_failure_counts.csv\` — Lomb-Scargle + spec.pgram peaks
- \`backblaze/analysis/periodicity_failure_counts.csv\` — Ljung-Box + STL
- \`backblaze/analysis/randomness_failure_counts.csv\` — full randomness battery
- \`backblaze/analysis/acf_daily_failures.csv\` — autocorrelation up to lag 730
- \`backblaze/analysis/model_clusters.csv\` — model-level k-means/hclust assignments
EOF
echo "wrote /Volumes/Gigantor/zarathustra/HANDOFF-INDEX.md"
