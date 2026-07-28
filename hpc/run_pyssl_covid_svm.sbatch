#!/bin/bash
#SBATCH --account=PCCF0025
#SBATCH --job-name=covid-svm
#SBATCH --partition=cpu
#SBATCH --nodes=1 --cpus-per-task=8 --mem=16G --time=01:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/slurm/%x-%A_%a.out --error=logs/slurm/%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"; mkdir -p logs/slurm
module purge; module load pytorch/2.8.0
export PYTHONUNBUFFERED=1
TASKS=(covid_vs_suspected covid_vs_healthy suspected_vs_healthy)
T="${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "[info] task=$T pre=svm"
python covid_transfer_eval.py --task "$T" --pre svm --repeats 50 --protocols spectrum subject --out results/covid
