#!/usr/bin/env bash
# Re-run only step 2 (triage training) of a finished run, on its existing bags: no download, no GPU.
#   RUN=diverse bash training/runs/triage/eddie/retrain.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../../../.." && pwd)}"
. "$HERE/config.sh"
[ -d "$WORK/bags-$ENCODER" ] || { echo "no bags in $WORK/bags-$ENCODER (RUN=$RUN); run submit.sh first" >&2; exit 1; }
mkdir -p "$WORK/logs"
pass="BASE=$BASE,REPO=$REPO,RUN=$RUN,PLAN=$PLAN,WORK=$WORK,ENV_DIR=$ENV_DIR,ENCODER=$ENCODER,TRIAL=$TRIAL,HF_HOME=$HF_HOME"
t=$(qsub -terse -v "$pass" -o "$WORK/logs" -N "triage_${RUN}_train" "$HERE/3_train.sh")
echo "submitted training of run $RUN: $t"
echo "when it finishes: python3 $TRIAL/mistakes.py $WORK/report-$ENCODER.json"
