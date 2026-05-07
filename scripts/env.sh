#!/bin/bash
# Source from the project root: `source scripts/env.sh`
# Puts the project's conda env on PATH. No cluster module load required —
# the env is self-contained.

_proj_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="${_proj_root}/.venv/bin:${PATH}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${_proj_root}${PYTHONPATH:+:${PYTHONPATH}}"
unset _proj_root
