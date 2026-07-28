#!/bin/bash
#SBATCH --account=PCCF0025
#SBATCH --job-name=byol-ctrl
#SBATCH --partition=gpu-exp
#SBATCH --nodes=1 --gres=gpu:v100-32g:1 --cpus-per-task=4 --mem=32G --time=00:40:00
#SBATCH --array=0-9%10
#SBATCH --output=logs/slurm/%x-%A_%a.out --error=logs/slurm/%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"; module purge; module load pytorch/2.8.0
export PYTHONUNBUFFERED=1
TID="${SLURM_ARRAY_TASK_ID:-0}"
# 0-4: BYOL@1024 under the ep1300-probe protocol (adam/1e-3/aug/5-fold)
# 5-9: BYOL@2048 under the Table-3 protocol (control for the published 78.6)
if (( TID < 5 )); then
  python lightning_finetune_pred.py --pre byol --task class30 --augtype phys --fold "$TID" \
    --data_variant full --pretrained_version version_bs1024 --pretrained_ckpt_name last.ckpt \
    --use_pretrained --split_mode random_holdout --n_splits 5 --valid_size 0.1 \
    --optimizer adam --adam_beta1 0.5 --adam_beta2 0.999 --lr 1e-3 --weight_decay 0 \
    --batch_size 64 --n_epochs 200 --patience 50 --use_augmentation --run_tag ctrl_bs1024_probeproto
else
  python lightning_finetune_pred.py --pre byol --task class30 --augtype phys --fold "$((TID-5))" \
    --data_variant full --pretrained_version version_1 --pretrained_ckpt_name last.ckpt \
    --use_pretrained --no-use_augmentation --run_tag ctrl_bs2048_table3proto
fi
