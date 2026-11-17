#!/usr/bin/env bash
# Job 1 of 3, staging queue (has internet): IDC manifest -> TCGA IDH status -> pick the slides in the run plan ->
# download them -> fetch H-optimus-0.
#$ -N triage_stage
#$ -q staging
#$ -l h_rt=48:00:00
# Memory: 16 GB is the staging queue maximum per slot; the IDC query needs about 4 GB on one thread.
#$ -l h_rss=16G
#$ -j y
set -euo pipefail
. "$TRIAL/eddie/config.sh"
activate_env
export ENTIP_DUCKDB_THREADS=1        # one slot here; more duckdb threads need more memory
mkdir -p "$WORK"
cd "$WORK"

plan_list() { python -c "import json,sys; print(' '.join(json.load(open(sys.argv[1])).get(sys.argv[2], [])))" "$PLAN" "$1"; }
cp "$PLAN" plan.json                                  # keep the plan this run used next to its results
# shellcheck disable=SC2046
[ -s manifest.csv ] || entip-train triage manifest $(plan_list manifest_args) --out manifest.csv > manifest.json
grep -qs IDH_CODEL_SUBTYPE idh.tsv || python "$TRIAL/fetch_idh.py" idh.tsv    # refetch a table from before 1p/19q was added
python "$TRIAL/pick_slides.py" manifest.csv trial.csv --plan "$PLAN" --idh idh.tsv | tee trial.txt
python "$TRIAL/eddie/stage_slides.py" trial.csv slides/

if [ "$ENCODER" = "h-optimus-0" ]; then         # compute nodes have no internet: cache the weights now
  python -c "import timm; timm.create_model('hf-hub:bioptimus/H-optimus-0', pretrained=True, init_values=1e-5, dynamic_img_size=False, num_classes=0); print('H-optimus-0 cached in', __import__('os').environ['HF_HOME'])"
fi
