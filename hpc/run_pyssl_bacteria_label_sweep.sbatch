#!/bin/bash
#SBATCH --account=PCCF0025
#SBATCH --job-name=pyssl-labelsweep
#SBATCH --partition=gpu-exp
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --array=0-59%15
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-/fs/ess/PCCF0025/leej101/pyssl_sers}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"
module purge
module load pytorch/2.8.0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

TID="${SLURM_ARRAY_TASK_ID:-0}"
COND=$(( TID / 20 ))
REM=$(( TID % 20 ))
LEVEL_IDX=$(( REM / 5 ))
FOLD=$(( REM % 5 ))
LEVELS=(10 20 50 100)
NLAB="${LEVELS[$LEVEL_IDX]}"

COMMON=(
    --task class30 --augtype phys --fold "${FOLD}" --data_variant full
    --split_mode random_holdout --n_splits 5 --valid_size 0.1
    --optimizer adam --adam_beta1 0.5 --adam_beta2 0.999
    --lr 1e-3 --weight_decay 0
    --batch_size 64 --n_epochs 200 --patience 50
    --use_augmentation
    --n_labels_per_class "${NLAB}"
    --run_tag "labelsweep_n${NLAB}"
)

case "${COND}" in
  0) CMD=(python lightning_finetune_pred.py --pre supervised --use_pretrained
          --reuse_pretrained_classifier
          --pretrained_version version_ho_adam_es10_aug
          --pretrained_ckpt_name last.ckpt "${COMMON[@]}") ;;
  1) CMD=(python lightning_finetune_pred.py --pre byol --use_pretrained
          --pretrained_version version_1 "${COMMON[@]}") ;;
  2) CMD=(python lightning_finetune_pred.py --pre no_pre --no-use_pretrained "${COMMON[@]}") ;;
esac

echo "[info] tid=${TID} cond=${COND} n_labels=${NLAB} fold=${FOLD}"
echo "[run] ${CMD[*]}"
"${CMD[@]}"
