#!/usr/bin/env bash
# CPU: label necrosis from this run's own TCGA reports and train a draft slide-level attention model.
#$ -N necrosis_report_mil
#$ -S /bin/bash
#$ -pe sharedmem 8
#$ -l h_rss=8G
#$ -l h_rt=24:00:00
#$ -j y
set -euo pipefail
REPO="${REPO:-/exports/igmm/eddie/Glioblastoma-WGS/scripts/entip}"
. "$REPO/training/runs/triage/eddie/config.sh"
activate_env

S="$BASE/runs/triage-diverse"
OUT="$BASE/runs/highgrade/necrosis-report1"
REPORTS="$OUT/reports/private"
[ -f "$REPORTS/text-ready" ] || { echo "the report text is not ready in $REPORTS" >&2; exit 1; }

# A fresh run: no earlier patient list is used, and 20% of its own labelled patients stay unseen
mkdir -p "$OUT/private"
chmod -R go-rwx "$OUT/private"
python "$REPO/training/runs/highgrade_necrosis/report_mil.py" --reports "$REPORTS" --manifest "$S/trial.csv" \
  --bags "$S/bags-h-optimus-0" --test-fraction 0.2 --out "$OUT"
chmod -R go-rwx "$OUT/private"
echo "done: $OUT/summary.json"
