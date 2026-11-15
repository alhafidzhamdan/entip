#!/usr/bin/env bash
# Job 2 of 3, GPU: train the student (offline; ImageNet start weights were cached by submit.sh). Resumes from
# last.pt if an earlier run was cut short. The GPU request comes from GPU_REQUEST, passed by submit.sh.
#$ -N distill_train
#$ -l h_rt=48:00:00
#$ -j y
set -euo pipefail
. "$DTRIAL/eddie/config.sh"
activate_env
export HF_HUB_OFFLINE=1
nvidia-smi -L || true
entip-glioma distill train --tiles "$DWORK/tiles" --out "$DWORK/student" --arch "$STUDENT_ARCH" \
  --epochs "$EPOCHS" --resume
