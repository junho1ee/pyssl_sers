#!/bin/bash
#SBATCH --account=PCCF0025
#SBATCH --job-name=pyssl-preproc-serial
#SBATCH --partition=gpu-exp
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
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

module purge
module load pytorch/2.8.0

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

TASK="${TASK:-class30}"
AUGTYPE="${AUGTYPE:-phys}"
PRE="${PRE:-no_pre}"
DATA_VARIANTS="${DATA_VARIANTS:-full minimal}"
FOLDS="${FOLDS:-0 1 2 3 4 5 6 7 8 9}"

echo "[info] host=$(hostname)"
echo "[info] repo_root=${REPO_ROOT}"
echo "[info] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[info] python=$(command -v python)"
python - <<'PY'
import lightning
import omegaconf
import torch

print(f"[info] torch={torch.__version__}")
print(f"[info] lightning={lightning.__version__}")
print(f"[info] omegaconf={omegaconf.__version__}")
print(f"[info] cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[info] gpu={torch.cuda.get_device_name(0)}")
PY

for DATA_VARIANT in ${DATA_VARIANTS}; do
    for FOLD in ${FOLDS}; do
        CMD=(
            python
            lightning_finetune_pred.py
            --pre "${PRE}"
            --task "${TASK}"
            --augtype "${AUGTYPE}"
            --fold "${FOLD}"
            --data_variant "${DATA_VARIANT}"
            --no-use_pretrained
            --no-use_augmentation
        )

        echo "[run] $(date --iso-8601=seconds) ${CMD[*]}"
        "${CMD[@]}"
    done
done
