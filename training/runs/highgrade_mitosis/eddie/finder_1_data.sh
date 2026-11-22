#!/usr/bin/env bash
# Staging queue (has internet): download and checksum the public mitosis data, then fix the train/tune/unseen split.
#$ -N mitosis_finder_data
#$ -q staging
#$ -l h_rt=48:00:00
#$ -l h_rss=16G
#$ -j y
#$ -o /exports/igmm/eddie/Glioblastoma-WGS/entip/runs/highgrade/mitosis-finder1/logs
set -euo pipefail
REPO="${REPO:-/exports/igmm/eddie/Glioblastoma-WGS/scripts/entip}"
. "$REPO/training/runs/triage/eddie/config.sh"
activate_env

DATA="$BASE/data/mitosis-public"
RUN="$BASE/runs/highgrade/mitosis-finder1"
mkdir -p "$DATA" "$RUN"
python "$REPO/training/runs/highgrade_mitosis/finder_data.py" fetch --data "$DATA"
if [ ! -f "$RUN/index.json" ]; then
  python "$REPO/training/runs/highgrade_mitosis/finder_data.py" prepare --data "$DATA" --out "$RUN"
fi
echo "done: $RUN/summary.json"
