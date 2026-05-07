#!/bin/bash
#SBATCH --job-name=rpca-jupyter
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

PORT=$(shuf -i 8500-8999 -n 1)
NODE=$(hostname -s)

cat <<EOF

==========================================================================
 Jupyter Lab is starting on $NODE:$PORT
 From your laptop run:
     ssh -L 8888:$NODE:$PORT $USER@idun-login.hpc.ntnu.no
 Then open the http://127.0.0.1:8888/?token=... URL printed below.
==========================================================================

EOF

python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

exec jupyter lab --no-browser --ip=0.0.0.0 --port=$PORT --notebook-dir=.
