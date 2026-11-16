#!/usr/bin/env bash
# GPU: embed the IPD-Brain slides with the same frozen encoder as the model under test. Array task i of N takes
# every N-th slide; slides whose bag exists are skipped, so a task can be resubmitted.
#$ -N ipd_embed
#$ -l h_rt=48:00:00
#$ -j y
set -euo pipefail
. "$ITRIAL/eddie/config.sh"
activate_env
export HF_HUB_OFFLINE=1
cd "$TWORK"
nvidia-smi -L || true
task="${SGE_TASK_ID:-undefined}"; last="${SGE_TASK_LAST:-undefined}"
if [ "$task" = "undefined" ] || [ "$last" = "undefined" ]; then task=1; last=1; fi
index="index.jsonl"; [ "$last" -gt 1 ] && index="index-$task.jsonl"
mpp=(); [ -n "$IPD_MPP" ] && mpp=(--mpp "$IPD_MPP")
todo=()
i=0
for f in slides/*; do
  i=$((i + 1))
  [ $(( (i - 1) % last )) -eq $((task - 1)) ] || continue
  sid="$(basename "$f")"; sid="${sid%.*}"
  [ -s "bags-$ENCODER/$sid.npz" ] || todo+=("$f")
done
echo "task $task of $last: ${#todo[@]} slides to embed with $ENCODER ${mpp[*]:-}"
if [ "${#todo[@]}" -gt 0 ]; then
  entip-train embed extract "${todo[@]}" --out "bags-$ENCODER" --encoder "$ENCODER" --index "$index" ${mpp[@]+"${mpp[@]}"} \
    | tee -a "embed-$ENCODER-$task.jsonl"
fi
