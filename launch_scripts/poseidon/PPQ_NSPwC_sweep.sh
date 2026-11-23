#!/bin/bash
#SBATCH --job-name=PPQ_NS_PwC_grid
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --array=0-23%6
#SBATCH --time=04:00:00
#SBATCH --output=logs/PPQ-NS-PwC-grid-%A_%a.out

set -eo pipefail
source "$HOME/miniconda3/bin/activate" quant
cd "${PROJECTDIR:?}/ritchie_workspace/poseidon-quantized"
exec python -u scripts/isambard_nspwc_ppq_sweep.py coarse "${SLURM_ARRAY_TASK_ID:?}"
