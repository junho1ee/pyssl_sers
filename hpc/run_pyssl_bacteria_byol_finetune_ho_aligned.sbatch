#!/bin/bash
#SBATCH --account=PCCF0025
#SBATCH --job-name=pyssl-byol-ft-ho
#SBATCH --partition=gpu-exp
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-4%5
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

FOLD="${SLURM_ARRAY_TASK_ID:-0}"
AUGTYPE="${AUGTYPE:-phys}"
TASK="${TASK:-class30}"
DATA_VARIANT="${DATA_VARIANT:-full}"
RUN_TAG="${RUN_TAG:-ho_split_adam_downstream_aug}"
PRETRAINED_VERSION="${PRETRAINED_VERSION:-version_1}"
SPLIT_MODE="${SPLIT_MODE:-random_holdout}"
N_SPLITS="${N_SPLITS:-5}"
VALID_SIZE="${VALID_SIZE:-0.1}"
OPTIMIZER="${OPTIMIZER:-adam}"
ADAM_BETA1="${ADAM_BETA1:-0.5}"
ADAM_BETA2="${ADAM_BETA2:-0.999}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
N_EPOCHS="${N_EPOCHS:-200}"
PATIENCE="${PATIENCE:-50}"
USE_DOWNSTREAM_AUG="${USE_DOWNSTREAM_AUG:-1}"

CMD=(
    python
    lightning_finetune_pred.py
    --pre byol
    --task "${TASK}"
    --augtype "${AUGTYPE}"
    --fold "${FOLD}"
    --data_variant "${DATA_VARIANT}"
    --pretrained_version "${PRETRAINED_VERSION}"
    --use_pretrained
    --split_mode "${SPLIT_MODE}"
    --n_splits "${N_SPLITS}"
    --valid_size "${VALID_SIZE}"
    --optimizer "${OPTIMIZER}"
    --adam_beta1 "${ADAM_BETA1}"
    --adam_beta2 "${ADAM_BETA2}"
    --lr "${LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --batch_size "${BATCH_SIZE}"
    --n_epochs "${N_EPOCHS}"
    --patience "${PATIENCE}"
    --run_tag "${RUN_TAG}"
)

if [[ "${USE_DOWNSTREAM_AUG}" == "1" ]]; then
    CMD+=(--use_augmentation)
else
    CMD+=(--no-use_augmentation)
fi

echo "[info] host=$(hostname)"
echo "[info] repo_root=${REPO_ROOT}"
echo "[info] fold=${FOLD}"
echo "[info] augtype=${AUGTYPE}"
echo "[info] task=${TASK}"
echo "[info] data_variant=${DATA_VARIANT}"
echo "[info] run_tag=${RUN_TAG}"
echo "[info] pretrained_version=${PRETRAINED_VERSION}"
echo "[info] use_downstream_aug=${USE_DOWNSTREAM_AUG}"
echo "[info] split_mode=${SPLIT_MODE} n_splits=${N_SPLITS} valid_size=${VALID_SIZE}"
echo "[info] optimizer=${OPTIMIZER} lr=${LR} betas=(${ADAM_BETA1},${ADAM_BETA2}) weight_decay=${WEIGHT_DECAY}"
echo "[info] batch_size=${BATCH_SIZE} n_epochs=${N_EPOCHS} patience=${PATIENCE}"
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
