#!/usr/bin/env bash
# CPU: build the blinded high-power MVP recheck page from the MVP head's out-of-fold scores.
#$ -N mvp_recheck
#$ -S /bin/bash
#$ -pe sharedmem 1
#$ -l h_rss=16G
#$ -l h_rt=02:00:00
#$ -j y
#$ -o /exports/igmm/eddie/Glioblastoma-WGS/entip/runs/highgrade/tile-heads-r2/logs
set -euo pipefail
REPO="${REPO:-/exports/igmm/eddie/Glioblastoma-WGS/scripts/entip}"
. "$REPO/training/runs/triage/eddie/config.sh"
activate_env

H="$BASE/runs/highgrade"
python "$REPO/training/runs/highgrade_mvp/recheck.py" --oof "$H/tile-heads-r2/mvp/oof.csv" \
  --locations "$H/tile-heads-r2/tile-locations.csv" --slides "$BASE/runs/triage-diverse/slides" \
  --out "$H/mvp-recheck1"
echo "done: $H/mvp-recheck1/labels.html"
