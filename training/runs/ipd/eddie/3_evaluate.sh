#!/usr/bin/env bash
# CPU: score the trained triage model on the IPD-Brain patients, as in use. Nothing is trained or refitted.
#$ -N ipd_evaluate
#$ -pe sharedmem 4
#$ -l h_rss=8G
#$ -l h_rt=12:00:00
#$ -j y
set -euo pipefail
. "$ITRIAL/eddie/config.sh"
activate_env
cd "$TWORK"
entip-train triage evaluate --manifest manifest.csv --bags "bags-$ENCODER" --model "$MODEL" \
  --out "evaluation-$MODEL_RUN.json" | tee "evaluation-$MODEL_RUN.txt"
