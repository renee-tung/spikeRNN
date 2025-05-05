'''
UTIL FUNCTIONS FOR SINGLE NEURON ANALYSIS
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import stats
import pandas as pd

import load_data as ld
from bootstrap_method import *



''' 
NEURON TUNING CALCULATIONS
'''

def calc_stim1_tuning(model_name, condn_phrase, condn_num):
    """
    for this model + condition, get the stim1 tuning preference for all neurons
    """

    _, _, rates_data = ld.load_neural_data(model_name, condn_phrase, condn_num,
                                                  load_LFP=False, load_spikes=False, load_rates=True)
    # behavioral data
    trial_labels, trial_perfs = ld.load_bhv_data(model_name, condn_phrase, condn_num)

    # timing data
    times_ms, _,_ = ld.get_times_dict('ds', condn_phrase, condn_num)


    all_stim1_fr = np.mean(rates_data[int(times_ms['stim1_on']):int(times_ms['stim1_off']), :,:], axis=0)

    trial_types, trial_idxs = np.unique(trial_labels[:,0], return_inverse=True) # only stim1
    n_trial_types = len(trial_types)

    tuning = np.zeros(all_stim1_fr.shape[0]) # tuning for each neuron
    for n_neuron in range(all_stim1_fr.shape[0]):
        stim1_rates = np.zeros((n_trial_types, int(rates_data.shape[2]/n_trial_types)))
        for i, trial_type in enumerate(trial_types):
            trials_idx = (trial_idxs == i)
            stim1_rates[i,:] = all_stim1_fr[n_neuron, trials_idx]
        _, p = stats.mannwhitneyu(stim1_rates[0,:], stim1_rates[1,:])
        if p < 0.05:
            tuning[n_neuron] = trial_types[np.argmax([stim1_rates[0,:].mean(), stim1_rates[1,:].mean()])]
        else:
            tuning[n_neuron] = np.nan

    return tuning




'''
RASTER PLOT FUNCTIONS
'''

def plot_trial_raster(trial_df, condn_phrase, condn_num, sort=None, title=None, ax=None):
    """
    Plot a raster plot of all cells in the trial in the given DataFrame.
    
    Parameters:
    - trial_df: DataFrame containing trial data with 'trial' and 'spike' columns.
    - sort: Optional; if provided, the trials will be sorted based on this column.
    """
    trial_id = trial_df.iloc[0]['trial_id']
    model_name = trial_df.iloc[0]['model_name']
    # delay = trial_df.iloc[0]['delay']

    # behavioral data
    trial_labels, trial_perfs = ld.load_bhv_data(model_name, condn_phrase=condn_phrase, condn_num=condn_num)

    # timing data
    times_ms, times_real, fs_dict = ld.get_times_dict('ds', condn_phrase, condn_num) #ds is fs=1000, ms

    if sort is not None:
        trial_df = trial_df.iloc[sort]
    else:
        trial_df = trial_df.sort_values(by='cell_type') # automatically sort by E/I

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    
    # plot raster for eacn neuron
    for n_cell in range(len(trial_df)):
        cell_spk_times = trial_df.iloc[n_cell]['spk_times']/fs_dict['spk']*1000 # convert to ms
        color = 'red' if trial_df.iloc[n_cell]['cell_type'] == 'exc' else 'blue'
        ax.plot(cell_spk_times, np.ones_like(cell_spk_times) * n_cell, '|', markersize=5, color=color)

    # shade stimulus periods
    stim_colors = get_stim_plotting_colors(trial_labels[trial_id,:])
    ax.axvspan(times_ms['stim1_on'], times_ms['stim1_off'], color=stim_colors[0], alpha=0.3)
    ax.axvspan(times_ms['stim2_on'], times_ms['stim2_off'], color=stim_colors[1], alpha=0.3)

    ax.set_title(f"Trial {trial_id}, {trial_labels[trial_id,:]}, perf: {trial_perfs[trial_id]}")
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron')
    ax.set_ylim(-1, len(trial_df));



def plot_neuron_raster(cell_df, condn_phrase, condn_num, title=None):
    """
    Plot a raster plot of all trials for a given neuron in the DataFrame.
    
    Parameters:
    - cell_df: DataFrame containing neuron data with 'trial' and 'spike' columns.
    - sort: Optional; if provided, the trials will be sorted based on this column.
    """
    cell_id = cell_df.iloc[0]['cell_id']
    trial_ids = cell_df['trial_id']
    model_name = cell_df.iloc[0]['model_name']
    # delay = cell_df.iloc[0]['delay']

    # behavioral data
    trial_labels, trial_perfs = ld.load_bhv_data(model_name, condn_phrase=condn_phrase, condn_num=condn_num)
    trial_labels = trial_labels[trial_ids,:]
    trial_perfs = trial_perfs[trial_ids]

    # timing data
    times_ms, times_real, fs_dict = ld.get_times_dict('ds', condn_phrase, condn_num) #ds is fs=1000, ms
    
    # plot raster for each trial
    fig, ax = plt.subplots(figsize=(8,4))
    for n_trial in range(len(cell_df)):
        trial_spk_times = cell_df.iloc[n_trial]['spk_times']/fs_dict['spk']*1000 # convert to ms
        if cell_df.iloc[n_trial]['cell_type'] == 'exc':
            plt.plot(trial_spk_times, np.ones_like(trial_spk_times) * n_trial, '|', markersize=5, color='red')
        else:
            plt.plot(trial_spk_times, np.ones_like(trial_spk_times) * n_trial, '|', markersize=5, color='blue')

    # shade trials by stim identity
    unique_labels, indices = np.unique(trial_labels, axis=0, return_index=True)
    indices = np.concatenate((indices, [len(cell_df)]))

    for i in range(len(unique_labels)):
        stim_colors = get_stim_plotting_colors(unique_labels[i,:])
        height = indices[i+1] - indices[i]
        rect1 = patches.Rectangle((times_ms['stim1_on'], indices[i]), 
                                times_ms['stim1_off'] - times_ms['stim1_on'], height,
                                color=stim_colors[0], alpha=0.3)
        rect2 = patches.Rectangle((times_ms['stim2_on'], indices[i]), 
                                times_ms['stim2_off'] - times_ms['stim2_on'], height,
                                color=stim_colors[1], alpha=0.3)
        ax.add_patch(rect1)
        ax.add_patch(rect2)
    
    if title is not None:
        plt.title(f'Neuron {cell_id}, {title}')
    else:
        plt.title(f"Neuron {cell_id}")
    plt.xlabel('Time (ms)')
    plt.ylabel('Trial')
    plt.ylim(-1, len(cell_df))
    plt.show()


'''
FIRING RATE FUNCTIONS
'''

def plot_neuron_rates(model_name, cell_id, condn_phrase, condn_num, cut_off = 0, title=None):
    """
    Plot the firing rates of a neuron across trials.
    
    Parameters:
    - model_name: Name of the model.
    - cell_id: ID of the neuron.
    - condn_phrase: Condition phrase for loading data.
    - condn_num: Condition number for loading data.
    """
    # Load the firing rates
    _, _, rates_data = ld.load_neural_data(model_name, condn_phrase, condn_num,
                                                  load_LFP=False, load_spikes=False, load_rates=True)
    rates_data_cell = rates_data[:, cell_id, :]
    del rates_data

    exc_ind, inh_ind = ld.get_celltype_label(model_name)
    cell_type = 'exc' if cell_id in exc_ind else 'inh'

    # behavioral data
    trial_labels, trial_perfs = ld.load_bhv_data(model_name, condn_phrase, condn_num)

    # get timing info
    times_ms, times_real, fs_dict = ld.get_times_dict('ds', condn_phrase, condn_num) #ds is fs=1000, ms

    _, colors = get_trialtype_colors()
    plt.figure(figsize=(8,4))

    # plot firing rate for each trial type
    trial_types, trial_idxs = np.unique(trial_labels, axis=0, return_inverse=True)
    for i, trial_type in enumerate(trial_types):
        trials_idx = (trial_idxs == i)
        trials_rate = rates_data_cell[cut_off:, trials_idx]
        # trials_rate = rates_data[cut_off:, cell_id, trials_idx]

        # first just plot mean and sem
        mean_rate = np.mean(trials_rate, axis=1)
        sem_rate = stats.sem(trials_rate, axis=1)

        plt.plot(np.arange(cut_off,times_ms['T']), mean_rate, color=colors[i], label=f'{trial_type}')
        plt.fill_between(np.arange(cut_off,times_ms['T']), mean_rate-sem_rate, mean_rate+sem_rate, color=colors[i], alpha=0.2)

    # shade stim times
    plt.axvspan(times_ms['stim1_on'], times_ms['stim1_off'], color='gray', alpha=0.2)
    plt.axvspan(times_ms['stim2_on'], times_ms['stim2_off'], color='gray', alpha=0.2)

    # axes
    plt.xlabel('Time (ms)')
    plt.ylabel('Firing rate (Hz)')
    if title is not None:
        plt.title(f'{title}')
    else:
        plt.title(f'Firing rate for cell {cell_id} ({cell_type}) across trials')
    plt.legend()
    plt.xlim(0, times_ms['T'])
    # plt.ylim(0, 50)
    plt.tight_layout()
    plt.show()



'''
UTILITY FUNCTIONS
'''

def get_stim_plotting_colors(stim):
    stim1_color = 'b' if stim[0] == 1 else 'r'
    stim2_color = 'b' if stim[1] == 1 else 'r'
    return [stim1_color, stim2_color]

def get_trialtype_colors():
    stims = np.array([[-1,-1], [-1,1], [1,-1], [1,1]])
    colors = ['#6E439A','#2B1644', '#236975','#49BEA3']
    return stims, colors