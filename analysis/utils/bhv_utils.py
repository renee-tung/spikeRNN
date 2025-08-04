'''
UTIL FUNCTIONS FOR PLOTTING JUST BEHAVIOR
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import stats
import pandas as pd

import load_data as ld


def plot_trial_performance(model_name, condn_phrase, condn_num, ax=None, title=None):
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
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f'{model_name[-6:]}, {condn_phrase} {condn_num}')
    ax.set_xlabel('Trial Type')
    ax.set_ylabel('Performance')
    # plt.grid()
        
    return ax


def plot_paired_trial_performance(model_name, condn_phrase1, condn_num1, condn_phrase2, condn_num2, 
                                  ax=None, title=None, text=True):
    """
    Plot the paired trial performance for two conditions
    """
    # Load the data
    trial_labels1, trial_perfs1 = ld.load_bhv_data(model_name, condn_phrase1, condn_num1)
    trial_labels2, trial_perfs2 = ld.load_bhv_data(model_name, condn_phrase2, condn_num2)

    # Get the unique trial types and their indices
    unique_labels = np.unique(trial_labels1, axis=0)
    stims, colors = get_trialtype_colors()

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    for i in range(unique_labels.shape[0]):
        trial_type = unique_labels[i]
        trial_indices1 = np.where(np.all(trial_labels1 == trial_type, axis=1))[0]
        trial_indices2 = np.where(np.all(trial_labels2 == trial_type, axis=1))[0]

        if len(trial_indices1) > 0 and len(trial_indices2) > 0:
            trial_performance1 = trial_perfs1[trial_indices1]
            trial_performance2 = trial_perfs2[trial_indices2]

            # Plot paired performance
            ax.plot([i - 0.15, i + 0.15], [trial_performance1.mean(), trial_performance2.mean()], color=colors[i], marker='o')
            # ax.errorbar(i - 0.15, trial_performance1.mean(), yerr=trial_performance1.std(), fmt='o', color=colors[i], capsize=5)
            # ax.errorbar(i + 0.15, trial_performance2.mean(), yerr=trial_performance2.std(), fmt='o', color=colors[i], capsize=5)

            if text:
                # Print performance on top of points
                ax.text(i - 0.15, trial_performance1.mean() + 0.02, f'{trial_performance1.mean():.2f}', ha='center', va='bottom')
                ax.text(i + 0.15, trial_performance2.mean() + 0.02, f'{trial_performance2.mean():.2f}', ha='center', va='bottom')

    ax.set_xticks(range(len(unique_labels)), [str(label) for label in unique_labels], fontsize=12)
    ax.set_xlim(-0.5, len(unique_labels) - 0.5)
    ax.set_ylim(0, 1.1)
    if title is not None:
        ax.set_title(title, fontsize=16)
    else:
        ax.set_title(f'{model_name[-6:]}, {condn_phrase1} {condn_num1} vs {condn_phrase2} {condn_num2}')
    ax.set_xlabel('Trial Type', fontsize=14)
    ax.set_ylabel('Performance', fontsize=14)
    ax.set_yticks(np.arange(0, 1.1, 0.2), np.round(np.arange(0, 1.1, 0.2),1), fontsize=12)

    return ax




def get_trialtype_colors():
    stims = np.array([[-1,-1], [-1,1], [1,-1], [1,1]])
    colors = ['#6E439A','#2B1644', '#236975','#49BEA3']
    return stims, colors






