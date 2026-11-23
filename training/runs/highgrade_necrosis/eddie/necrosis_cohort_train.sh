#!/usr/bin/env bash
# CPU: confirm the chosen slides were embedded, then train the draft necrosis attention model.
#$ -N necrosis_cohort_train
#$ -S /bin/bash
#$ -pe sharedmem 8
#$ -l h_rss=8G
#$ -l h_rt=48:00:00
#$ -j y
set -euo pipefail
REPO="${REPO:-/exports/igmm/eddie/Glioblastoma-WGS/scripts/entip}"
. "$REPO/training/runs/triage/eddie/config.sh"
activate_env
H="$BASE/runs/highgrade"
W="$H/necrosis-report2"
TOOLS="$REPO/training/runs/highgrade_necrosis"
python "$TOOLS/necrosis_cohort.py" check --trial "$W/trial.csv" --bags "$W/bags-h-optimus-0"
mkdir -p "$W/model/private"
chmod -R go-rwx "$W/model/private"
python "$TOOLS/report_mil.py" --reports "$H/necrosis-report1/reports/private" --manifest "$W/trial.csv" \
  --bags "$W/bags-h-optimus-0" --test-fraction 0.2 --out "$W/model"
chmod -R go-rwx "$W/model/private"
echo "done: $W/model/summary.json"
