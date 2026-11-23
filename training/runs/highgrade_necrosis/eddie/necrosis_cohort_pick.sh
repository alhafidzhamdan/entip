#!/usr/bin/env bash
# Staging queue (has internet): list TCGA FFPE slides and choose those with a clear necrosis report. Downloads no slides.
#$ -N necrosis_cohort_pick
#$ -q staging
#$ -S /bin/bash
#$ -l h_rss=16G
#$ -l h_rt=06:00:00
#$ -j y
set -euo pipefail
REPO="${REPO:-/exports/igmm/eddie/Glioblastoma-WGS/scripts/entip}"
. "$REPO/training/runs/triage/eddie/config.sh"
activate_env
export ENTIP_DUCKDB_THREADS=1
H="$BASE/runs/highgrade"
W="$H/necrosis-report2"
mkdir -p "$W"
cd "$W"
[ -s manifest.csv ] || entip-train triage manifest --prep ffpe --out manifest.csv > manifest.json
grep -qs IDH_CODEL_SUBTYPE idh.tsv || python "$REPO/training/runs/triage/fetch_idh.py" idh.tsv
python "$REPO/training/runs/highgrade_necrosis/necrosis_cohort.py" pick --manifest manifest.csv --idh idh.tsv \
  --labels "$H/necrosis-report1/private/report-labels.csv" --out "$W"
echo "done: $W/cohort.json"
