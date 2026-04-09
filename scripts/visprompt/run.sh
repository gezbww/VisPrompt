#!/usr/bin/env bash
dataset="dataset"
n_shot=32
seed=102
#export CUDA_VISIBLE_DEVICES=0
for nt in sym asym; do
  for rate in 0.125 0.25 0.375 0.5 0.625 0.75; do
    echo "======  noise_type=$nt  noise_rate=$rate  ======"
    bash scripts/visprompt/main.sh "$dataset" "$n_shot" "$rate" "$nt" "$seed"
  done
done