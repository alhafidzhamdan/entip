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
mkdir -p "$(dirname "$ENV_DIR")" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$HF_HOME" "$XDG_CACHE_HOME" "$TMPDIR"
cat <<EOF
Installing into:
  env          $ENV_DIR
  conda cache  $CONDA_PKGS_DIRS
  pip temp     $TMPDIR (pip cache off)
  other caches $XDG_CACHE_HOME
  H-optimus-0  $HF_HOME
Needs about 8 GB and tens of thousands of files (PyTorch + CUDA). Quota for $USER on igmmfs01:
EOF
mmlsquota -u "$USER" igmmfs01 2>/dev/null || echo "  (mmlsquota not available here; check your IGMM quota before installing)"
[ -x "$ENV_DIR/bin/python" ] || conda create -y -p "$ENV_DIR" python=3.11
set -u
export PYTHONNOUSERSITE=1   # never fall back to ~/.local (home quota)
PY="$ENV_DIR/bin/python"     # the env's own python and pip, whatever conda has activated
"$PY" -m pip install -e "$REPO/platform/live-training" -e "$REPO/training/glioma[idc,slides,foundation]"
"$PY" -c "import entip_glioma, idc_index, openslide, torch, timm; print('env ok, torch', torch.__version__)"
echo "Env ready at $ENV_DIR. Next: bash $HERE/submit.sh"
