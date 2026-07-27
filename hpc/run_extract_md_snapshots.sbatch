#!/bin/bash
#SBATCH --account=PCCF0025
#SBATCH --job-name=extract-md-snap
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=leej101@ccf.org

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${PY_CONDA_ENV:-helmgfn-ai}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

TRAJ="${TRAJ:?TRAJ is required}"
TEMPLATE="${TEMPLATE:?TEMPLATE is required}"
OUTPUT="${OUTPUT:?OUTPUT is required}"
N_FRAMES="${N_FRAMES:-100}"
STRIP_TO_TEMPLATE="${STRIP_TO_TEMPLATE:-1}"

CMD=(
    python
    scripts/md/extract_amber_nc_snapshots.py
    --traj "${TRAJ}"
    --template "${TEMPLATE}"
    --output "${OUTPUT}"
    --n-frames "${N_FRAMES}"
)

if [[ "${STRIP_TO_TEMPLATE}" == "1" ]]; then
    CMD+=(--strip-to-template)
fi

echo "[info] repo_root=${REPO_ROOT}"
echo "[info] traj=${TRAJ}"
echo "[info] template=${TEMPLATE}"
echo "[info] output=${OUTPUT}"
echo "[info] n_frames=${N_FRAMES}"
echo "[run] ${CMD[*]}"
"${CMD[@]}"

