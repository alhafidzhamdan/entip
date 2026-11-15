#!/usr/bin/env bash
# Run once on an Eddie login node (it has internet): makes the conda env in the group space and installs the packages.
#   bash training/runs/triage/eddie/setup.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../../../.." && pwd)}"
. "$HERE/config.sh"

set +u          # module and conda activate/deactivate scripts read unset variables
. /etc/profile.d/modules.sh
module load anaconda
mkdir -p "$(dirname "$ENV_DIR")"
cat <<EOF
Installing into:
  env          $ENV_DIR
  conda cache  $CONDA_PKGS_DIRS
  pip temp     $TMPDIR (pip cache off)
  home (~)     $HOME  (all other tool caches and data)
  H-optimus-0  $HF_HOME
Needs about 8 GB and tens of thousands of files (PyTorch + CUDA).
EOF
[ -x "$ENV_DIR/bin/python" ] || conda create -y -p "$ENV_DIR" -c conda-forge --override-channels python=3.11
set -u
export PYTHONNOUSERSITE=1   # never fall back to ~/.local (home quota)
unset PYTHONPATH PYTHONHOME  # in case the anaconda module set them
PY="$ENV_DIR/bin/python"     # the env's own python and pip, whatever conda has activated
"$PY" -m pip install -e "$REPO/platform/live-training" -e "$REPO/training/glioma[idc,slides,foundation]"
"$PY" -c "import entip_glioma, idc_index, openslide, torch, timm; print('env ok, torch', torch.__version__)"
echo "Env ready at $ENV_DIR. Next: bash $HERE/submit.sh"
