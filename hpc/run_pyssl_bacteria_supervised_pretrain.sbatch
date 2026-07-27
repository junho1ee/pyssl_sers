#!/bin/bash
#SBATCH --account=PCCF0025
#SBATCH --job-name=pyssl-sup-pre
#SBATCH --partition=gpu-exp
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
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

set +e
module purge
set -e
module load pytorch/2.8.0

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

AUGTYPE="${AUGTYPE:-phys}"
DATA_VARIANT="${DATA_VARIANT:-full}"
USE_PRETRAIN_AUG="${USE_PRETRAIN_AUG:-0}"
LOGGER_VERSION="${LOGGER_VERSION:-version_1}"
export LOGGER_VERSION
OPTIMIZER="${OPTIMIZER:-adamw}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.999}"
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
BATCH_SIZE="${BATCH_SIZE:-512}"
N_EPOCHS="${N_EPOCHS:-200}"
PATIENCE="${PATIENCE:-10}"
VALID_SIZE="${VALID_SIZE:-0.1}"

CMD=(
    python
    lightning_pretrain_supervised.py
    --pre supervised
    --augtype "${AUGTYPE}"
    --data_variant "${DATA_VARIANT}"
    --optimizer "${OPTIMIZER}"
    --adam_beta1 "${ADAM_BETA1}"
    --adam_beta2 "${ADAM_BETA2}"
    --lr "${LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --batch_size "${BATCH_SIZE}"
    --n_epochs "${N_EPOCHS}"
    --patience "${PATIENCE}"
    --valid_size "${VALID_SIZE}"
)

if [[ "${USE_PRETRAIN_AUG}" == "1" ]]; then
    CMD+=(--use_augmentation)
else
    CMD+=(--no-use_augmentation)
fi

echo "[info] host=$(hostname)"
echo "[info] repo_root=${REPO_ROOT}"
echo "[info] augtype=${AUGTYPE}"
echo "[info] data_variant=${DATA_VARIANT}"
echo "[info] use_pretrain_aug=${USE_PRETRAIN_AUG}"
echo "[info] logger_version=${LOGGER_VERSION}"
echo "[info] optimizer=${OPTIMIZER} lr=${LR} betas=(${ADAM_BETA1},${ADAM_BETA2}) weight_decay=${WEIGHT_DECAY}"
echo "[info] batch_size=${BATCH_SIZE} n_epochs=${N_EPOCHS} patience=${PATIENCE} valid_size=${VALID_SIZE}"
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
