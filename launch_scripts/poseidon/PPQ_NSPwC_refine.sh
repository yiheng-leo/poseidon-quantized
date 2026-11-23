#!/bin/bash
#SBATCH --job-name=PPQ_NS_PwC_refine
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --array=0-3%4
#SBATCH --time=12:00:00
#SBATCH --output=logs/PPQ-NS-PwC-refine-%A_%a.out

set -eo pipefail
source "$HOME/miniconda3/bin/activate" quant
cd "${PROJECTDIR:?}/ritchie_workspace/poseidon-quantized"
targets=(2 3 4 8)
exec python -u scripts/isambard_nspwc_ppq_sweep.py refine "${targets[${SLURM_ARRAY_TASK_ID:?}]}"
