#!/bin/bash

# train load 3 models
for i in {1..30}
do
    echo "Running iteration $i..."
    python main.py --gpu 0 --gpu_frac 0.20 --n_trials 30000 --mode train \
    --N 400 --P_inh 0.20 --som_N 0 --apply_dale True --gain 1.5 --task letters --task_load 3 \
    --act sigmoid --loss_fn l2 --decay_taus 4 25 --output_dir ../
    echo "Iteration $i complete."
done

# train load 2 models
for i in {1..30}
do
    echo "Running iteration $i..."
    python main.py --gpu 0 --gpu_frac 0.20 --n_trials 30000 --mode train \
    --N 400 --P_inh 0.20 --som_N 0 --apply_dale True --gain 1.5 --task letters --task_load 2 \
    --act sigmoid --loss_fn l2 --decay_taus 4 25 --output_dir ../
    echo "Iteration $i complete."
done

