#!/bin/bash

# train models
for i in {1..3}
do
    echo "Running iteration $i..."
    python main_interleaved.py --gpu 0 --gpu_frac 0.20 --n_trials 40000 --mode train \
    --N 1000 --P_inh 0.20 --som_N 0 --apply_dale True --gain 1.5 --task sternberg \
    --load_proportion 0.5 --delay_dur 50 \
    --act sigmoid --loss_fn l2 --decay_taus 4 150 --jitter_onset 5 --jitter_delay 3 \
    --output_dir ~/Documents/renee/
    echo "Iteration $i complete."
done


# python main_interleaved.py --gpu 0 --gpu_frac 0.20 --n_trials 40000 --mode train \
# --N 1000 --P_inh 0.20 --som_N 0 --apply_dale True --gain 1.5 --task sternberg \
# --load_proportion 0.5 --delay_dur 50 \
# --act sigmoid --loss_fn l2 --decay_taus 4 150 --jitter_onset 5 --jitter_delay 3 \
# --output_dir ~/Documents/renee/