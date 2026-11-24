#!/usr/bin/env bash
# GPU: score the second finder on the reviewed glioma fields, add the development patients' review to training,
# fine-tune a third finder from the second, freeze it, and score it on the held-out patients only.
# If the job runs out of time, submit it again: finished steps are skipped and training resumes.
#$ -N mitosis_finder3_glioma
#$ -S /bin/bash
#$ -q gpu
#$ -l gpu=1
#$ -l h200=true
#$ -pe sharedmem 4
#$ -l h_rss=24G
#$ -l h_rt=24:00:00
#$ -j y
#$ -o /exports/igmm/eddie/Glioblastoma-WGS/entip/runs/highgrade/mitosis-finder3/logs
set -euo pipefail
REPO="${REPO:-/exports/igmm/eddie/Glioblastoma-WGS/scripts/entip}"
. "$REPO/training/runs/triage/eddie/config.sh"
activate_env
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
nvidia-smi -L || true

H="$BASE/runs/highgrade"
S="$BASE/runs/triage-diverse"
ROUND="$H/mitosis-glioma-round1"
SECOND="$H/mitosis-finder2"
RUN="$H/mitosis-finder3"
CODE="$REPO/training/runs/highgrade_mitosis"
for f in "$ROUND/private/mitosis-marks-AH.csv" "$ROUND/private/mitosis-call-labels-AH.csv" "$SECOND/frozen.json"; do
  [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }
done

if [ ! -f "$ROUND/score-mitosis-finder2.json" ]; then
  python "$CODE/glioma_retrain.py" score --round "$ROUND" --finder "$SECOND" --slides "$S/slides" --groups held_out development
fi
if [ ! -f "$RUN/index.json" ]; then
  python "$CODE/glioma_retrain.py" data --round "$ROUND" --source "$SECOND" --slides "$S/slides" \
    --reference "$H/mitosis-ref1/private/patients.csv" --out "$RUN"
fi
if [ ! -f "$RUN/frozen.json" ]; then
  python "$CODE/finder.py" train --run "$RUN" --init "$SECOND" --glioma-share 0.25 --epochs 20 --lr 3e-4 \
    --eval-every 2 --workers "${NSLOTS:-4}"
  python "$CODE/finder.py" freeze --run "$RUN"
fi
if [ ! -f "$ROUND/score-mitosis-finder3.json" ]; then
  python "$CODE/glioma_retrain.py" score --round "$ROUND" --finder "$RUN" --slides "$S/slides" --groups held_out
fi
chmod -R go-rwx "$RUN/glioma"
echo "done: $ROUND/score-mitosis-finder2.json and $ROUND/score-mitosis-finder3.json"
