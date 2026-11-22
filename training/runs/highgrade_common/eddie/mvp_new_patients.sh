#!/usr/bin/env bash
# CPU: freeze the retrained MVP head, score new patients and build their blinded labelling page.
#$ -N mvp_new_patients
#$ -S /bin/bash
#$ -pe sharedmem 2
#$ -l h_rss=16G
#$ -l h_rt=04:00:00
#$ -j y
#$ -o /exports/igmm/eddie/Glioblastoma-WGS/entip/runs/highgrade/tile-heads-r2/logs
set -euo pipefail
REPO="${REPO:-/exports/igmm/eddie/Glioblastoma-WGS/scripts/entip}"
. "$REPO/training/runs/triage/eddie/config.sh"
activate_env

H="$BASE/runs/highgrade"
S="$BASE/runs/triage-diverse"
HOLD=""
if [ -f "$H/held_out.csv" ]; then HOLD="--holdout $H/held_out.csv"; fi
# shellcheck disable=SC2086  # HOLD is empty or two words; these paths have no spaces
python "$REPO/training/runs/highgrade_mvp/new_patients.py" --head "$H/tile-heads-r2-recheck1/mvp/head.joblib" \
  --manifest "$S/trial.csv" --bags "$S/bags-h-optimus-0" --slides "$S/slides" \
  --exclude "$H/review-round2/tile-review-key-round2.csv" $HOLD --out "$H/mvp-new1"
echo "done: $H/mvp-new1/labels.html"
