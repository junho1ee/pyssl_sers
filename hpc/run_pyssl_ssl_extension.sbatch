#!/bin/bash
#SBATCH --account=PCCF0025
#SBATCH --job-name=ssl-ext
#SBATCH --partition=gpu-exp
#SBATCH --nodes=1 --gres=gpu:v100-32g:1 --cpus-per-task=4 --mem=32G --time=01:00:00
#SBATCH --array=0-59%20
#SBATCH --output=logs/slurm/%x-%A_%a.out --error=logs/slurm/%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"; mkdir -p logs/slurm
module purge; module load pytorch/2.8.0
export PYTHONUNBUFFERED=1
TID="${SLURM_ARRAY_TASK_ID:-0}"
PRES=(mocov3 simclrv2)
COMMON=(--augtype phys --data_variant full --pretrained_version version_bs1024
        --pretrained_ckpt_name last.ckpt --use_pretrained
        --split_mode random_holdout --n_splits 5 --valid_size 0.1
        --optimizer adam --adam_beta1 0.5 --adam_beta2 0.999 --lr 1e-3 --weight_decay 0
        --batch_size 64 --n_epochs 200 --patience 50 --use_augmentation)

if (( TID < 20 )); then                    # Table 2: 2 methods x 2 tasks x 5 folds
  P="${PRES[$(( TID / 10 ))]}"; R=$(( TID % 10 ))
  TASK=$([ $R -lt 5 ] && echo class30 || echo class2); FOLD=$(( R % 5 ))
  echo "[T2] pre=$P task=$TASK fold=$FOLD"
  python lightning_finetune_pred.py --pre "$P" --task "$TASK" --fold "$FOLD" \
      "${COMMON[@]}" --run_tag ho_split_adam_downstream_aug
else                                       # Fig 4: 2 methods x 4 levels x 5 folds
  T=$(( TID - 20 )); P="${PRES[$(( T / 20 ))]}"; R=$(( T % 20 ))
  LEVELS=(10 20 50 100); NLAB="${LEVELS[$(( R / 5 ))]}"; FOLD=$(( R % 5 ))
  echo "[F4] pre=$P n_labels=$NLAB fold=$FOLD"
  python lightning_finetune_pred.py --pre "$P" --task class30 --fold "$FOLD" \
      "${COMMON[@]}" --n_labels_per_class "$NLAB" --run_tag "labelsweep_n${NLAB}"
fi
