#!/bin/bash
#SBATCH --account=PCCF0025
#SBATCH --job-name=covid-transfer
#SBATCH --partition=gpu-exp
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --array=0-11%12
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs/slurm
module purge; module load pytorch/2.8.0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

TID="${SLURM_ARRAY_TASK_ID:-0}"
TASKS=(covid_vs_suspected covid_vs_healthy suspected_vs_healthy)
PRES=(no_pre_noaug no_pre_aug supervised byol)
TASK="${TASKS[$((TID / 4))]}"
PRE="${PRES[$((TID % 4))]}"

echo "[info] tid=${TID} task=${TASK} pre=${PRE}"
python covid_transfer_eval.py --task "${TASK}" --pre "${PRE}" \
    --repeats 50 --protocols spectrum subject --out results/covid
