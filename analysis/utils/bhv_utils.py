'''
UTIL FUNCTIONS FOR PLOTTING JUST BEHAVIOR
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import stats
import pandas as pd

import load_data as ld


def plot_trial_performance(model_name, condn_phrase, condn_num, ax=None):
    """
    Plot the trial performance for a given model and condition
    """
    # Load the data
    trial_labels, trial_perfs = ld.load_bhv_data(model_name, condn_phrase, condn_num)

    # Get the unique trial types and their indices
    unique_labels = np.unique(trial_labels, axis=0)
    stims, colors = get_trialtype_colors()

    trial_performances = []
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    for i in range(unique_labels.shape[0]):
        trial_type = unique_labels[i]
        trial_indices = np.where(np.all(trial_labels == trial_type, axis=1))[0]
        trial_performance = trial_perfs[trial_indices]
        trial_performances.append(trial_performance)
        # print(trial_performance.std())
        # plt.bar(i, trial_performance.mean(), yerr=trial_performance.std(), capsize=5)
        ax.bar(i, trial_performance.mean(), color=colors[i])
        # print performance on top of bar
        ax.text(i, trial_performance.mean() + 0.02, f'{trial_performance.mean():.3f}', ha='center', va='bottom')


    ax.set_xticks(range(len(unique_labels)), [str(label) for label in unique_labels])
    ax.set_xlim(-0.5, len(unique_labels) - 0.5)
    ax.set_ylim(0, 1.1)
    # plt.axhline(y=0.5, color='r', linestyle='--')
    # plt.axhline(y=0.75, color='g', linestyle='--')
    ax.set_title(model_name[-6:])
    ax.set_xlabel('Trial Type')
    ax.set_ylabel('Performance')
    # plt.grid()
        
    return ax







def get_trialtype_colors():
    stims = np.array([[-1,-1], [-1,1], [1,-1], [1,1]])
    colors = ['#6E439A','#2B1644', '#236975','#49BEA3']
    return stims, colors






