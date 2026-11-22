#!/usr/bin/env bash
# CPU: search the diverse set's embeddings for tiles like the 87 confirmed MVP tiles and build a blinded review page.
#$ -N mvp_retrieve
#$ -S /bin/bash
#$ -pe sharedmem 4
#$ -l h_rss=8G
#$ -l h_rt=08:00:00
#$ -j y
#$ -o /exports/igmm/eddie/Glioblastoma-WGS/entip/runs/highgrade/tile-heads-r2/logs
set -euo pipefail
REPO="${REPO:-/exports/igmm/eddie/Glioblastoma-WGS/scripts/entip}"
. "$REPO/training/runs/triage/eddie/config.sh"
activate_env

H="$BASE/runs/highgrade"
S="$BASE/runs/triage-diverse"
OLD_HOLD=""
if [ -f "$H/held_out.csv" ]; then OLD_HOLD="--holdout $H/held_out.csv"; fi
# shellcheck disable=SC2086  # empty or two words; these paths have no spaces
python "$REPO/training/runs/highgrade_mvp/retrieve.py" \
  --labelled "$H/mvp-recheck1/key.csv" "$H/mvp-recheck1/labels-AH-recheck1.csv" \
  --labelled "$H/mvp-new1/key.csv" "$H/mvp-new1/labels-AH-new1.csv" --expect-references 87 \
  --reviewed "$H/tile-heads-r2/tile-locations.csv" --manifest "$S/trial.csv" --bags "$S/bags-h-optimus-0" \
  --slides "$S/slides" --holdout "$H/mvp-final-test/patients.csv" $OLD_HOLD --out "$H/mvp-retrieve1"
echo "done: $H/mvp-retrieve1/labels.html"
