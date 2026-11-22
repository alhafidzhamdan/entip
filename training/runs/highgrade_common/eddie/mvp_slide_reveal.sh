#!/usr/bin/env bash
# CPU: after the locked comparison, score where the frozen MVP head points and build the reveal overlay.
#$ -N mvp_slide_reveal
#$ -S /bin/bash
#$ -pe sharedmem 8
#$ -l h_rss=8G
#$ -l h_rt=6:00:00
#$ -j y
#$ -o /exports/igmm/eddie/Glioblastoma-WGS/entip/runs/highgrade/tile-heads-r2/logs
set -euo pipefail
REPO="${REPO:-/exports/igmm/eddie/Glioblastoma-WGS/scripts/entip}"
. "$REPO/training/runs/triage/eddie/config.sh"
activate_env
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

H="$BASE/runs/highgrade"
S="$BASE/runs/triage-diverse"
python "$REPO/training/runs/highgrade_mvp/slide_reveal.py" --pilot "$H/mvp-pilot1" \
  --calls "$H/mvp-pilot1/slide-calls-AH.csv" --bags "$S/bags-h-optimus-0" --workers "${NSLOTS:-1}"
chmod -R go-rwx "$H/mvp-pilot1/reveal"
echo "done: $H/mvp-pilot1/reveal/reveal.html"
