#!/usr/bin/env bash
# Job 1 of 3, staging queue (has internet): IDC manifest -> pick 10 slides -> download them -> fetch H-optimus-0.
#$ -N triage_test_stage
#$ -q staging
#$ -l h_rt=02:00:00
#$ -l h_rss=32G
#$ -j y
set -euo pipefail
. "$TRIAL/eddie/config.sh"
activate_env
export ENTIP_DUCKDB_THREADS=1        # one slot here; more duckdb threads need more memory
mkdir -p "$WORK"
cd "$WORK"

[ -s manifest.csv ] || entip-train triage manifest --prep ffpe paxgene_paraffin --out manifest.csv > manifest.json
python "$TRIAL/pick_slides.py" manifest.csv trial.csv --per-label glioma=5 normal_brain=5 | tee trial.txt
python "$TRIAL/eddie/stage_slides.py" trial.csv slides/

if [ "$ENCODER" = "h-optimus-0" ]; then         # compute nodes have no internet: cache the weights now
  python -c "import timm; timm.create_model('hf-hub:bioptimus/H-optimus-0', pretrained=True, init_values=1e-5, dynamic_img_size=False, num_classes=0); print('H-optimus-0 cached in', __import__('os').environ['HF_HOME'])"
fi
