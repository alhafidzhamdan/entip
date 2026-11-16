#!/usr/bin/env bash
# Job 2 of 3, GPU: step 1 (pre-training) - embed every staged slide with the frozen H-optimus-0, offline.
# Submitted as an array of EMBED_SHARDS tasks (config.sh), each on its own GPU: task i embeds every
# EMBED_SHARDS-th slide, starting at the i-th, and skips slides whose bag already exists, so a task that runs out
# of time can simply be submitted again. Each task writes its own index-<i>.jsonl.
#$ -N triage_embed
#$ -l h_rt=48:00:00
# The GPU request (queue, parallel environment or gpu resource, memory) comes from GPU_REQUEST in config.sh,
# passed by submit.sh, so it can be changed without editing this file.
#$ -j y
set -euo pipefail
. "$TRIAL/eddie/config.sh"
activate_env
export HF_HUB_OFFLINE=1                          # weights come from $HF_HOME, cached by job 1
cd "$WORK"
nvidia-smi -L || true

task="${SGE_TASK_ID:-undefined}"; last="${SGE_TASK_LAST:-undefined}"
if [ "$task" = "undefined" ] || [ "$last" = "undefined" ]; then task=1; last=1; fi
index="index.jsonl"; [ "$last" -gt 1 ] && index="index-$task.jsonl"

todo=()
i=0
for d in slides/*/; do                            # glob order is sorted, so every task sees the same list
  i=$((i + 1))
  [ $(( (i - 1) % last )) -eq $((task - 1)) ] || continue
  uid="$(basename "$d")"
  [ -s "bags-$ENCODER/$uid.npz" ] || todo+=("$d")
done
echo "task $task of $last: ${#todo[@]} slides to embed with $ENCODER"
if [ "${#todo[@]}" -gt 0 ]; then               # one process per task, so the model loads once
  entip-train embed extract "${todo[@]}" --out "bags-$ENCODER" --encoder "$ENCODER" --index "$index" \
    | tee -a "embed-$ENCODER-$task.jsonl"
fi
