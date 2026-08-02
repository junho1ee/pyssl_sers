#!/bin/bash
#SBATCH --account=PCCF0025
#SBATCH --job-name=covid-smoke
#SBATCH --partition=gpu-exp
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs/slurm
module purge; module load pytorch/2.8.0
export PYTHONUNBUFFERED=1
python covid_transfer_eval.py --task covid_vs_healthy --pre byol --repeats 3 --out results/covid_smoke
