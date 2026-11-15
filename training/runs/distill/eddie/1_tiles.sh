#!/usr/bin/env bash
# Job 1 of 3, CPU: cut each teacher bag's tiles from its slide into the tile cache.
#$ -N distill_tiles
#$ -pe sharedmem 4
#$ -l h_rss=8G
#$ -l h_rt=24:00:00
#$ -j y
set -euo pipefail
. "$DTRIAL/eddie/config.sh"
activate_env
mkdir -p "$DWORK"
entip-glioma distill tiles --bags "$SRC/bags-h-optimus-0" --slides "$SRC/slides" --manifest "$SRC/trial.csv" \
  --out "$DWORK/tiles" --max-tiles "$MAX_TILES" | tail -1
