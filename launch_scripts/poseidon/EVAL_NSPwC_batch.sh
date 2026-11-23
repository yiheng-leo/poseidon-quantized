#!/bin/bash
#SBATCH --job-name=EVAL_NS_PwC_batch
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=logs/EVAL-NS-PwC-batch-%j.out

set -eo pipefail
source "$HOME/miniconda3/bin/activate" quant
cd "${PROJECTDIR:?}/ritchie_workspace/poseidon-quantized"
dataset=fluids.incompressible.PiecewiseConstants
root=results/NS-PwC
sweep_report="$root/ppq-layerwise-traj16-ft10-b32-sweep/sweep-summary.json"
mkdir -p "$(dirname "$sweep_report")"
python scripts/isambard_nspwc_ppq_sweep.py report > "$sweep_report.${SLURM_JOB_ID}.tmp"
mv "$sweep_report.${SLURM_JOB_ID}.tmp" "$sweep_report"
selected=$(python scripts/isambard_nspwc_ppq_sweep.py report --selected-only)
models=()
for name in brecq-traj16-ft10 ppq-traj16-ft10 sbpq-traj16-ft10 sbpq-traj16-ft10-default \
            sbpq-traj16-ft10-steps200 sbpq-traj16-ft10-b64-steps103; do
  if [[ -s "$root/$name/compressed_model.pt" && -s "$root/$name/metadata.json" ]]; then
    models+=("$root/$name")
  fi
done
while IFS= read -r path; do
  if [[ -n "$path" ]]; then
    models+=("$path")
  fi
done <<< "$selected"

exec python -u -m evaluation.evaluate \
  --models "${models[@]}" \
  --checkpoint "models/poseidon/$dataset" --dataset "$dataset" \
  --data_path poseidon/camlab-ethz/down_streams \
  --protocol safe-svd --split test --eval_obj ongt --final_time 20 \
  --DEV cuda --batch_size 256 --physics_batch_size 10 --num_workers 0 \
  --output_path "$root/result.json"
