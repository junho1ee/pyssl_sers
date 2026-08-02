#!/bin/bash
#SBATCH --account=PCCF0025
#SBATCH --job-name=pyssl-ssl-pre
#SBATCH --partition=gpu-exp
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-3%4
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs/slurm
module purge; module load pytorch/2.8.0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

TID="${SLURM_ARRAY_TASK_ID:-0}"
PRES=(mocov3 mocov3 simclrv2 simclrv2)
AUGS=(phys crop phys crop)
PRE="${PRES[$TID]}"
AUGTYPE="${AUGS[$TID]}"
N_EPOCHS="${N_EPOCHS:-3000}"
export LOGGER_VERSION="${LOGGER_VERSION:-version_1}"

echo "[info] pre=${PRE} augtype=${AUGTYPE} n_epochs=${N_EPOCHS} version=${LOGGER_VERSION}"
python lightning_pretrain_ssl.py --pre "${PRE}" --augtype "${AUGTYPE}" --n_epochs "${N_EPOCHS}"
