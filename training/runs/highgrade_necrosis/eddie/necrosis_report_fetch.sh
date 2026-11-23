#!/usr/bin/env bash
# Staging queue (has internet): download this run's own copy of the open-access TCGA glioma reports.
#$ -N necrosis_report_fetch
#$ -q staging
#$ -S /bin/bash
#$ -l h_rss=8G
#$ -l h_rt=06:00:00
#$ -j y
set -euo pipefail
REPO="${REPO:-/exports/igmm/eddie/Glioblastoma-WGS/scripts/entip}"
. "$REPO/training/runs/triage/eddie/config.sh"
activate_env
python "$REPO/training/runs/highgrade_mvp/report_labels.py" fetch --out "$BASE/runs/highgrade/necrosis-report1/reports"
