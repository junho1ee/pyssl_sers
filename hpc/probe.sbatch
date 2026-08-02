#!/bin/bash
#SBATCH --account=PCCF0025
#SBATCH --job-name=ssl-probe
#SBATCH --partition=gpu-exp
#SBATCH --nodes=1 --gpus-per-node=1 --cpus-per-task=4 --mem=64G --time=00:30:00
#SBATCH --output=logs/slurm/%x-%j.out --error=logs/slurm/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"; module purge; module load pytorch/2.8.0
export PYTHONUNBUFFERED=1
python scripts_probe.py
