#!/usr/bin/env bash
# Run once on an Eddie login node (it has internet): makes the conda env in the group space and installs the packages.
#   bash training/runs/triage-test-run/eddie/setup.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../../../.." && pwd)}"
. "$HERE/config.sh"

set +u          # module and conda activate/deactivate scripts read unset variables
. /etc/profile.d/modules.sh
module load anaconda
mkdir -p "$(dirname "$ENV_DIR")" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$HF_HOME"
[ -d "$ENV_DIR" ] || conda create -y -p "$ENV_DIR" python=3.11
source activate "$ENV_DIR"
set -u
pip install -e "$REPO/platform/live-training" -e "$REPO/training/glioma[idc,slides,foundation]"
python -c "import entip_glioma, idc_index, openslide, torch, timm; print('env ok, torch', torch.__version__)"
echo "Env ready at $ENV_DIR. Next: bash $HERE/submit.sh"
