#!/bin/bash
#SBATCH --job-name=SBPQ_NS_PwC_b64
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/SBPQ-NS-PwC-b64-%j.out

set -eo pipefail
source "$HOME/miniconda3/bin/activate" quant
cd "${PROJECTDIR:?}/ritchie_workspace/poseidon-quantized"
dataset=fluids.incompressible.PiecewiseConstants
exec python -u -m SBPQ.sbpq \
  --config configs/sbpq/default.json \
  --model_path "models/poseidon/$dataset" --dataset "$dataset" \
  --data_path poseidon/camlab-ethz/down_streams \
  --traj_num 16 --random_calibration n --calibration_seed 42 \
  --final_time 10 --time_back 0 --batch_size 64 --num_workers 0 --seed 42 --DEV cuda \
  --target_bits 8 --init_bits 8 --activation_bits 32 --mc_samples 10 \
  --total_steps 103 --likelihood_scale 0.03125 --sobolev_order 2 --save_every 100 \
  --save_path results/NS-PwC/sbpq-traj16-ft10-b64-steps103
