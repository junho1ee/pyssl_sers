#!/bin/bash
#SBATCH --account=PCCF0025
#SBATCH --job-name=ssl-matched
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-32g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --array=0-5%6
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs/slurm
module purge; module load pytorch/2.8.0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

TID="${SLURM_ARRAY_TASK_ID:-0}"
PRES=(byol byol mocov3 mocov3 simclrv2 simclrv2)
AUGS=(phys crop phys crop phys crop)
PRE="${PRES[$TID]}"; AUGTYPE="${AUGS[$TID]}"
export LOGGER_VERSION="version_bs1024"

echo "[info] pre=${PRE} aug=${AUGTYPE} batch=1024 epochs=3000 version=${LOGGER_VERSION}"
python lightning_pretrain_ssl.py --pre "${PRE}" --augtype "${AUGTYPE}" \
    --batch_size 1024 --n_epochs 3000
