#!/bin/bash

# Loop to repeat the command X times
for i in {1..7}
do
    echo "Running iteration $i..."
    python main.py --gpu 0 --gpu_frac 0.20 --n_trials 20000 --mode train \
    --N 200 --P_inh 0.20 --som_N 0 --apply_dale True --gain 1.5 --task letters \
    --act sigmoid --loss_fn l2 --decay_taus 4 25 --output_dir ../
    echo "Iteration $i complete."
done

