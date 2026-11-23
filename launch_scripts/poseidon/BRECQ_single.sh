#!/bin/bash
#SBATCH --job-name=BRECQ_NS_PwC
#SBATCH --nodes=1
#SBATCH --gpus=1            # also allocates 72 CPU cores and 115GB memory
#SBATCH --time=24:00:00
#SBATCH --output=logs/BRECQ-NS-PwC.out

source /home/u6abp/bwang.u6abp/miniconda3/bin/activate
conda activate quant
cd $PROJECTDIR/ritchie_workspace/poseidon-quantized
python -u -m BRECQ.brecq \
  --config configs/brecq/default.json \
  --model_path models/poseidon/fluids.incompressible.PiecewiseConstants \
  --dataset fluids.incompressible.PiecewiseConstants \
  --data_path poseidon/camlab-ethz/down_streams \
  --traj_num 16 --random_calibration n --calibration_seed 42 \
  --rec_batchsize 128 --reconstruction_iterations 5000 \
  --final_time 10 --time_back 0 \
  --batch_size 32 --seed 1 --DEV cuda \
  --save_path results/NS-PwC/brecq-traj16-ft10