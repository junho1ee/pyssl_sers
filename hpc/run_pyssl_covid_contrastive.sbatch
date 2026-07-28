#!/bin/bash
#SBATCH --account=PCCF0025
#SBATCH --job-name=covid-contrastive
#SBATCH --partition=gpu-exp
#SBATCH --nodes=1 --gres=gpu:v100-32g:1 --cpus-per-task=4 --mem=32G --time=02:30:00
#SBATCH --array=0-5%6
#SBATCH --output=logs/slurm/%x-%A_%a.out --error=logs/slurm/%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"; mkdir -p logs/slurm
module purge; module load pytorch/2.8.0
export PYTHONUNBUFFERED=1
TID="${SLURM_ARRAY_TASK_ID:-0}"
TASKS=(covid_vs_suspected covid_vs_healthy suspected_vs_healthy)
PRES=(mocov3 simclrv2)
TASK="${TASKS[$(( TID / 2 ))]}"; PRE="${PRES[$(( TID % 2 ))]}"
echo "[info] task=${TASK} pre=${PRE}"
python covid_transfer_eval.py --task "${TASK}" --pre "${PRE}" \
    --repeats 50 --protocols spectrum subject --out results/covid
