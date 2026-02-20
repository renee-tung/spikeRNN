#!/bin/bash

# train models
for i in {1..4}
do
    echo "Running iteration $i..."
    python main.py --gpu 0 --gpu_frac 0.20 --n_trials 40000 --mode train \
    --N 400 --P_inh 0.20 --som_N 0 --apply_dale True --gain 1.5 --task sternberg \
    --act sigmoid --loss_fn l2 --decay_taus 4 25 --jitter_onset 5 --jitter_delay 3 \
    --output_dir ~/Documents/renee/
    echo "Iteration $i complete."
done

# python main.py --gpu 0 --gpu_frac 0.20 --n_trials 40000 --mode train \
# --N 400 --P_inh 0.20 --som_N 0 --apply_dale True --gain 1.5 --task sternberg \
# --act sigmoid --loss_fn l2 --decay_taus 4 25 --jitter_onset 5 --jitter_delay 3 \
# --output_dir ~/Documents/renee/