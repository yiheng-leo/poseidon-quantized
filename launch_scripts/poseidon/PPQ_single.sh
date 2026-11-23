#!/bin/bash
#SBATCH --job-name=PPQ_NS_PwC
#SBATCH --nodes=1
#SBATCH --gpus=1            # also allocates 72 CPU cores and 115GB memory
#SBATCH --time=24:00:00
#SBATCH --output=logs/PPQ-NS-PwC.out

source /home/u6abp/bwang.u6abp/miniconda3/bin/activate
conda activate quant
cd $PROJECTDIR/ritchie_workspace/poseidon-quantized
python -u -m PPQ.ppq \
  --config configs/ppq/layerwise.json \
  --variant layerwise \
  --model_path models/poseidon/fluids.incompressible.PiecewiseConstants \
  --dataset fluids.incompressible.PiecewiseConstants \
  --data_path poseidon/camlab-ethz/down_streams \
  --traj_num 16 \
  --random_calibration n --calibration_seed 42 \
  --final_time 10 --time_back 0 \
  --batch_size 32 --num_workers 0 \
  --activation_bits 32 \
  --seed 1 --DEV cuda \
  --target_bits 8 --init_bits 8 \
  --mc_samples 10 --epochs 5 \
  --learning_rate 9.1e-4 \
  --eta 1e-6 --percentile 1e-4 \
  --gamma 1e-6 \
  --save_every 1000 \
  --save_path results/NS-PwC/ppq-traj16-ft10