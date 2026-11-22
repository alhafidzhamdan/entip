#!/usr/bin/env bash
# Staging queue (has internet): download open-access TCGA glioma reports.
#$ -N mvp_reports_fetch
#$ -q staging
#$ -S /bin/bash
#$ -l h_rss=8G
#$ -l h_rt=06:00:00
#$ -j y
set -euo pipefail
REPO="${REPO:-/exports/igmm/eddie/Glioblastoma-WGS/scripts/entip}"
. "$REPO/training/runs/triage/eddie/config.sh"
activate_env
H="$BASE/runs/highgrade"
python "$REPO/training/runs/highgrade_mvp/report_labels.py" fetch --out "$H/mvp-reports"
