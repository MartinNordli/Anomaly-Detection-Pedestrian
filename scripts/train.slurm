#!/bin/bash
#SBATCH --job-name=rpca-train
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd /cluster/home/martnal/rpca
mkdir -p logs

source scripts/env.sh

python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

jupyter nbconvert --to notebook --execute notebooks/main.ipynb \
    --output main.executed.ipynb --ExecutePreprocessor.timeout=-1
