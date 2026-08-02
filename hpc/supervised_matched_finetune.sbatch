#!/bin/bash
#SBATCH --account=PCCF0025
#SBATCH --job-name=sup-matched-ft
#SBATCH --partition=gpu-exp
#SBATCH --nodes=1 --gres=gpu:v100-32g:1 --cpus-per-task=4 --mem=32G --time=01:00:00
#SBATCH --array=0-19%20
#SBATCH --output=logs/slurm/%x-%A_%a.out --error=logs/slurm/%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"; mkdir -p logs/slurm
module purge; module load pytorch/2.8.0
export PYTHONUNBUFFERED=1

# Supervised reference pretraining under the Table-3 downstream protocol
# (stratified 10-fold, no downstream augmentation), so it can be compared inside
# Table 3 instead of across the five-split protocol of Table 2.
#  0-9 : classifier head reused, matching the "supervised pretraining" row of Table 2
# 10-19: classifier head reinitialized, matching what every self-supervised row gets
TID="${SLURM_ARRAY_TASK_ID:-0}"
FOLD=$(( TID % 10 ))

COMMON=(--pre supervised --task class30 --augtype phys --fold "${FOLD}"
        --data_variant full
        --pretrained_version version_ho_adam_es10_aug --pretrained_ckpt_name last.ckpt
        --use_pretrained --no-use_augmentation)

if (( TID < 10 )); then
  echo "[info] supervised reuse-head fold=${FOLD} (Table-3 protocol)"
  python lightning_finetune_pred.py "${COMMON[@]}" \
    --reuse_pretrained_classifier --run_tag matched_table3
else
  echo "[info] supervised fresh-head fold=${FOLD} (Table-3 protocol)"
  python lightning_finetune_pred.py "${COMMON[@]}" \
    --no-reuse_pretrained_classifier --run_tag matched_table3_freshhead
fi
