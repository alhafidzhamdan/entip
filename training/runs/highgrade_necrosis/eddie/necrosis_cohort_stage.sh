#!/usr/bin/env bash
# Staging queue (has internet): download the chosen slides and make sure the H-optimus-0 weights are cached.
#$ -N necrosis_cohort_stage
#$ -q staging
#$ -S /bin/bash
#$ -l h_rss=16G
#$ -l h_rt=48:00:00
#$ -j y
set -euo pipefail
REPO="${REPO:-/exports/igmm/eddie/Glioblastoma-WGS/scripts/entip}"
. "$REPO/training/runs/triage/eddie/config.sh"
activate_env
cd "$BASE/runs/highgrade/necrosis-report2"
python -c "import timm; timm.create_model('hf-hub:bioptimus/H-optimus-0', pretrained=True, init_values=1e-5, dynamic_img_size=False, num_classes=0)"
python "$REPO/training/runs/triage/eddie/stage_slides.py" trial.csv slides/
