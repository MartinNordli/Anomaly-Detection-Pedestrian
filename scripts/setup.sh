#!/bin/bash
# One-time setup: creates a self-contained conda env at ./.venv and installs
# all Python deps via pip. After this runs, ./.venv/bin/python and ./.venv/bin/pip
# work standalone — no module load needed for day-to-day use.
set -euo pipefail

cd "$(dirname "$0")/.."

module purge
module load Miniconda3/24.7.1-0

if [[ ! -x .venv/bin/python ]]; then
    conda create -p ./.venv python=3.11 -y
fi

./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu121 \
    'torch==2.5.1'
./.venv/bin/pip install -r requirements.txt

./.venv/bin/python -m ipykernel install --user --name rpca --display-name "Python (rpca)"

./.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"
echo "Done."
