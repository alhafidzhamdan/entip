#!/usr/bin/env bash
# Submit the three jobs on Eddie, each waiting for the one before:  bash training/runs/triage-trial-10/eddie/submit.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../../../.." && pwd)}"
. "$HERE/config.sh"
mkdir -p "$WORK/logs"
pass="REPO=$REPO,WORK=$WORK,ENV_DIR=$ENV_DIR,ENCODER=$ENCODER,TRIAL=$TRIAL"
q() { qsub -terse -v "$pass" -o "$WORK/logs" "$@"; }

s=$(q "$HERE/1_stage.sh")
e=$(q -hold_jid "$s" "$HERE/2_embed.sh")
t=$(q -hold_jid "$e" "$HERE/3_train.sh")
echo "submitted: stage $s -> embed $e -> train $t"
echo "watch: qstat;  logs: $WORK/logs;  results: $WORK/report-$ENCODER.json"
