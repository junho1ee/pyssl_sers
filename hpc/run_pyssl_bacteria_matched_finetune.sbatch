#!/bin/bash
#SBATCH --account=PCCF0025
#SBATCH --job-name=ssl-matched-ft
#SBATCH --partition=gpu-exp
#SBATCH --nodes=1 --gres=gpu:v100-32g:1 --cpus-per-task=4 --mem=32G --time=01:00:00
#SBATCH --array=0-59%20
#SBATCH --output=logs/slurm/%x-%A_%a.out --error=logs/slurm/%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"; mkdir -p logs/slurm
module purge; module load pytorch/2.8.0
export PYTHONUNBUFFERED=1
TID="${SLURM_ARRAY_TASK_ID:-0}"
PRES=(byol byol mocov3 mocov3 simclrv2 simclrv2)
AUGS=(phys crop phys crop phys crop)
CFG=$(( TID / 10 )); FOLD=$(( TID % 10 ))
PRE="${PRES[$CFG]}"; AUGTYPE="${AUGS[$CFG]}"
echo "[info] pre=${PRE} aug=${AUGTYPE} fold=${FOLD} (Table-3 protocol, bs1024 ckpt)"
python lightning_finetune_pred.py --pre "${PRE}" --task class30 --augtype "${AUGTYPE}" \
    --fold "${FOLD}" --data_variant full \
    --pretrained_version version_bs1024 --pretrained_ckpt_name last.ckpt \
    --use_pretrained --no-use_augmentation --run_tag matched_bs1024
