#!/bin/bash
#SBATCH --account=PCCF0025
#SBATCH --job-name=byol-epochprobe
#SBATCH --partition=gpu-exp
#SBATCH --nodes=1 --gres=gpu:v100-32g:1 --cpus-per-task=4 --mem=32G --time=00:40:00
#SBATCH --array=0-29%10
#SBATCH --output=logs/slurm/%x-%A_%a.out --error=logs/slurm/%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"; mkdir -p logs/slurm
module purge; module load pytorch/2.8.0
export PYTHONUNBUFFERED=1

# Downstream accuracy of BYOL@1024 as a function of the pretraining epoch.
# Replaces the two-fold `ckptprobe` measurement whose probe.ckpt no longer exists:
# every saved pretraining checkpoint is fine-tuned over the same five folds under
# the fixed probe protocol (adam/1e-3/aug, 200 epochs, patience 50), so the late-
# training degradation claim rests on a matched five-fold curve.
EPOCHS=(499 999 1499 1999 2499 2999)
TID="${SLURM_ARRAY_TASK_ID:-0}"
EP="${EPOCHS[$(( TID / 5 ))]}"
FOLD=$(( TID % 5 ))

CKPT_DIR=results/bacteria-id/pretraining/phys/byol/lightning_logs/version_bs1024/checkpoints
CKPT=$(basename "$(ls "${CKPT_DIR}"/epoch=${EP}-loss=*.ckpt | head -1)")
echo "[info] epoch=${EP} fold=${FOLD} ckpt=${CKPT}"

python lightning_finetune_pred.py --pre byol --task class30 --augtype phys --fold "${FOLD}" \
  --data_variant full --pretrained_version version_bs1024 --pretrained_ckpt_name "${CKPT}" \
  --use_pretrained --split_mode random_holdout --n_splits 5 --valid_size 0.1 \
  --optimizer adam --adam_beta1 0.5 --adam_beta2 0.999 --lr 1e-3 --weight_decay 0 \
  --batch_size 64 --n_epochs 200 --patience 50 --use_augmentation \
  --run_tag "epochprobe_e${EP}"
