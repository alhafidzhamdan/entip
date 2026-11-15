#!/usr/bin/env bash
# Job 2 of 3, GPU: step 1 (pre-training) - embed every staged slide with the frozen H-optimus-0, offline.
#$ -N triage_test_embed
#$ -l h_rt=04:00:00
# The GPU request (queue, parallel environment or gpu resource, memory) comes from GPU_REQUEST in config.sh,
# passed by submit.sh, so it can be changed without editing this file.
#$ -j y
set -euo pipefail
. "$TRIAL/eddie/config.sh"
activate_env
export HF_HUB_OFFLINE=1                          # weights come from $HF_HOME, cached by job 1
cd "$WORK"
nvidia-smi -L || true

todo=()
for d in slides/*/; do
  uid="$(basename "$d")"
  [ -s "bags-$ENCODER/$uid.npz" ] || todo+=("$d")
done
echo "${#todo[@]} slides to embed with $ENCODER"
if [ "${#todo[@]}" -gt 0 ]; then               # one process, so the model loads once
  entip-train embed extract "${todo[@]}" --out "bags-$ENCODER" --encoder "$ENCODER" | tee -a "embed-$ENCODER.jsonl"
fi
