#!/bin/bash
#SBATCH --account=PCCF0025
#SBATCH --job-name=pyssl-sup-ft-main
#SBATCH --partition=gpu-exp
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-19%20
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err
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

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
AUGTYPE="${AUGTYPE:-phys}"
TASK="${TASK:-class30}"
DATA_VARIANT="${DATA_VARIANT:-full}"

if (( TASK_ID < 10 )); then
    FOLD="${TASK_ID}"
    PRETRAINED_VERSION="${PRETRAINED_VERSION_NO_AUG:-version_1}"
    USE_DOWNSTREAM_AUG=0
else
    FOLD="$((TASK_ID - 10))"
    PRETRAINED_VERSION="${PRETRAINED_VERSION_AUG:-version_2}"
    USE_DOWNSTREAM_AUG=1
fi

CMD=(
    python
    lightning_finetune_pred.py
    --pre supervised
    --task "${TASK}"
    --augtype "${AUGTYPE}"
    --fold "${FOLD}"
    --data_variant "${DATA_VARIANT}"
    --pretrained_version "${PRETRAINED_VERSION}"
    --use_pretrained
)

if [[ "${USE_DOWNSTREAM_AUG}" == "1" ]]; then
    CMD+=(--use_augmentation)
else
    CMD+=(--no-use_augmentation)
fi

echo "[info] host=$(hostname)"
echo "[info] repo_root=${REPO_ROOT}"
echo "[info] task_id=${TASK_ID}"
echo "[info] fold=${FOLD}"
echo "[info] augtype=${AUGTYPE}"
echo "[info] data_variant=${DATA_VARIANT}"
echo "[info] pretrained_version=${PRETRAINED_VERSION}"
echo "[info] use_downstream_aug=${USE_DOWNSTREAM_AUG}"
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

echo "[run] ${CMD[*]}"
"${CMD[@]}"
