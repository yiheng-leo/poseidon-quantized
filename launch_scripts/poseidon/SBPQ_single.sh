#!/bin/bash
#SBATCH --job-name=SBPQ_NS_PwC
#SBATCH --nodes=1
#SBATCH --gpus=1            # also allocates 72 CPU cores and 115GB memory
#SBATCH --time=24:00:00
#SBATCH --output=logs/SBPQ-NS-PwC.out

source /home/u6abp/bwang.u6abp/miniconda3/bin/activate
conda activate quant
cd $PROJECTDIR/ritchie_workspace/poseidon-quantized
python -u -m SBPQ.sbpq \
  --config configs/sbpq/default.json \
  --model_path models/poseidon/fluids.incompressible.PiecewiseConstants \
  --dataset fluids.incompressible.PiecewiseConstants  \
  --data_path poseidon/camlab-ethz/down_streams \
  --traj_num 16 \
  --random_calibration n \
  --calibration_seed 42 \
  --final_time 10 --time_back 0 \
  --batch_size 2 --DEV cuda \
  --sobolev_order 2 \
  --save_path quantization_artifacts/NS-PwC/sbpq-traj16-ft10-default