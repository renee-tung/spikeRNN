'''
UTIL FUNCTIONS FOR SINGLE NEURON ANALYSIS
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import stats
from scipy.io import loadmat
import pandas as pd
from sklearn.metrics import pairwise_distances, silhouette_score
from umap import UMAP
from sklearn.cluster import KMeans
import pdb

import load_data as ld
from bootstrap_method import *


'''
PROCESSING SPIKE TIMES
'''



def count_spikes(spk_times, start, stop):
    """
    Count the number of spikes in a given time window.
    """
    spk_times = np.asarray(spk_times)
    return np.sum((spk_times >= start) & (spk_times <= stop))


def calc_lowfr_neurons(spk_df, times_spk, threshold=2):
    '''
    Returns the indices of neurons that have no trials above a certain firing rate threshold
    '''
    spk_df['total_spkcount'] = spk_df['spk_times'].apply(lambda spikes: count_spikes(spikes, times_spk['stim1_on'], times_spk['T']))
    n_neurons = len(np.unique(spk_df['cell_id']))
    n_above_threshold = np.zeros(n_neurons)
    avg_fr_neuron = np.zeros(n_neurons)
    for i_neuron in range(n_neurons):
        neuron_df = spk_df[spk_df['cell_id'] == i_neuron]
        avg_fr_neuron[i_neuron] = np.mean(neuron_df['total_spkcount']) / ((times_spk['T'] - times_spk['stim1_on']) / times_spk['fs'])
    #     n_above_threshold[i_neuron] = np.sum(neuron_df['total_spkcount'] >= (times_spk['T']-times_spk['stim1_on'])/times_spk['fs'] * threshold)

    return np.where(avg_fr_neuron < threshold)[0] # these neurons had no trials above desired fr threshold
    # return np.where(n_above_threshold == 0)[0] # these neurons had no trials above desired fr threshold


''' 
NEURON TUNING CALCULATIONS
'''


def calc_stim1_tuning(model_name, condn_phrase, condn_num, rates_data=None,
                      all_models_dir='/home/nuttidalab/Documents/renee/all_DMS_models/'):
    """
    for this model + condition, get the stim1 tuning preference for all neurons
    """
    if rates_data is None:
        _, _, rates_data = ld.load_neural_data(model_name, condn_phrase, condn_num,
                                                    load_LFP=False, load_spikes=False, load_rates=True,
                                                    all_models_dir=all_models_dir)
    # behavioral data
    trial_labels, trial_perfs = ld.load_bhv_data(model_name, condn_phrase, condn_num,
                                                all_models_dir=all_models_dir)

    # timing data
    times_ms, _,_ = ld.get_times_dict('ds', condn_phrase, condn_num, all_models_dir=all_models_dir) #ds is fs=1000, ms

    # mean across time during stim1 period
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
        if p < 0.01:
            tuning[n_neuron] = trial_types[np.argmax([stim1_rates[0,:].mean(), stim1_rates[1,:].mean()])]
        else:
            tuning[n_neuron] = np.nan

    return tuning

def calc_stim1_tuning_spikes(model_name, condn_phrase, condn_num, spk_df = None, 
                             all_models_dir='/home/nuttidalab/Documents/renee/all_DMS_models/'):
    """
    for this model + condition, get the stim1 tuning preference for all neurons
    """
    
    # see if this is already saved
    tuning_filepath = f'{all_models_dir}/{model_name}/tuning_{condn_phrase}_{condn_num}.mat'
    if os.path.exists(tuning_filepath):
        tuning = loadmat(tuning_filepath)['tuning'][0]
        return tuning
    
    
    if spk_df is None:
        _, spk_df, _ = ld.load_neural_data(model_name, condn_phrase, condn_num,
                                                    load_LFP=False, load_spikes=True, load_rates=False,
                                                    all_models_dir=all_models_dir)
        # results = ld.load_neural_data(model_name, condn_phrase, condn_num, remove_lowfr=True,
        #                                         load_LFP=False, load_spikes=True, load_rates=False)
        # spk_df = results['spk_df']
        # cell_ids = results['idxs_old']
        
        
    # behavioral data
    trial_labels, trial_perfs = ld.load_bhv_data(model_name, condn_phrase, condn_num,
                                                 all_models_dir=all_models_dir)

    # timing data
    times_spk, _,_ = ld.get_times_dict('spk', condn_phrase, condn_num, all_models_dir=all_models_dir)

    # get spike count within stim1 period added as a col to df
    spk_df['stim1_spkcount'] = spk_df['spk_times'].apply(lambda spikes: count_spikes(spikes, times_spk['stim1_on'], times_spk['stim1_off']))

    trial_types, trial_idxs = np.unique(trial_labels[:,0], return_inverse=True) # only stim1
    n_trial_types = len(trial_types)
    n_trials = len(trial_labels)

    n_cells = len(spk_df['cell_id'].unique())
    tuning = np.zeros(n_cells) # tuning for each neuron
    for i_neuron, neuron_id in enumerate(spk_df['cell_id'].unique()):
        neuron_df = spk_df[spk_df['cell_id'] == neuron_id]
        stim1_counts = np.zeros((n_trial_types, int(n_trials/n_trial_types)))
        for i, trial_type in enumerate(trial_types):
            trials_idx = (trial_idxs == i)
            stim1_counts[i,:] = neuron_df['stim1_spkcount'].iloc[trials_idx].values

        _, p = stats.mannwhitneyu(stim1_counts[0,:], stim1_counts[1,:])
        if p < 0.05:
            tuning[i_neuron] = trial_types[np.argmax([stim1_counts[0,:].mean(), stim1_counts[1,:].mean()])]
        else:
            tuning[i_neuron] = np.nan

    return tuning

def calc_stim1_tuning_spikes_corr(model_name, condn_phrase, condn_num, spk_df = None):
    """
    for this model + condition, get the stim1 tuning preference for all neurons
    """
    # if spk_df is None:
    #     _, spk_df, _ = ld.load_neural_data(model_name, condn_phrase, condn_num,
    #                                                 load_LFP=False, load_spikes=True, load_rates=False)
    # # behavioral data
    # trial_labels, trial_perfs = ld.load_bhv_data(model_name, condn_phrase, condn_num)

    # # timing data
    # times_spk, _,_ = ld.get_times_dict('spk', condn_phrase, condn_num)

    # # get spike count within stim1 period added as a col to df
    # spk_df['stim1_spkcount'] = spk_df['spk_times'].apply(lambda spikes: count_spikes(spikes, times_spk['stim1_on'], times_spk['stim1_off']))

    # trial_types, trial_idxs = np.unique(trial_labels[:,0], return_inverse=True) # only stim1
    # n_trial_types = len(trial_types)
    # n_trials = len(trial_labels)

    # n_cells = len(spk_df['cell_id'].unique())
    # tuning = np.zeros(n_cells) # tuning for each neuron
    # for i_neuron in range(n_cells):
    #     neuron_df = spk_df[spk_df['cell_id'] == i_neuron]
    #     stim1_counts = []
    #     for i, trial_type in enumerate(trial_types):
    #         trials_idx = np.where(trial_idxs == i)[0]
    #         trials_idx = trials_idx[np.where(trial_perfs[trials_idx] == 1)[0]] # only do trials with correct performance
    #         stim1_counts.append(neuron_df['stim1_spkcount'].iloc[trials_idx].values)

    #     _, p = stats.mannwhitneyu(stim1_counts[0], stim1_counts[1])
    #     if p < 0.05:
    #         tuning[i_neuron] = trial_types[np.argmax([stim1_counts[0].mean(), stim1_counts[1].mean()])]
    #     else:
    #         tuning[i_neuron] = np.nan

    # return tuning
    return "use calc_stim1_tuning_spikes instead, this function is deprecated"


def plot_stim1_tuning(tuning, cell_idxs=None, exc_ind = None, ax=None, title=None):
    """
    Plot the stim1 tuning for a given model and condition
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    if cell_idxs is None:
        cell_idxs = np.arange(len(tuning))

    x_labels = ['-1','+1','none']
    x = np.arange(len(x_labels))
    n_tuned = np.zeros(len(x_labels))
    n_exc = np.zeros(len(x_labels))
    tuning_options = [-1, 1, np.nan]
    for i, tuning_option in enumerate(tuning_options):
        n_tuned[i] = np.sum(tuning[cell_idxs] == tuning_option)
        if np.isnan(tuning_option):
            n_tuned[i] = np.sum(np.isnan(tuning[cell_idxs]))
        if exc_ind is not None:
            n_exc[i] = np.sum(tuning[cell_idxs[exc_ind]] == tuning_option)
            if np.isnan(tuning_option):
                n_exc[i] = np.sum(np.isnan(tuning[cell_idxs[exc_ind]]))
    
    if exc_ind is None:
        ax.bar(x, n_tuned, color='black', alpha=0.5)
    else:
        ax.bar(x, n_exc, color='red', alpha=0.5)
        ax.bar(x, n_tuned-n_exc, bottom=n_exc, color='blue', alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel('Number of neurons')
    ax.set_ylim([0, 100])
    ax.set_xlabel('Tuning')
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('Tuning of neurons to stim1')

    




'''
RASTER PLOT FUNCTIONS
'''

def plot_trial_raster(trial_df, condn_phrase, condn_num, sort=None, title=None, ax=None, hlines=None):
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

    if hlines is not None:
        for hline in hlines:
            ax.axhline(y=hline, color='k', linestyle='--')

    if title is not None:
        ax.set_title(f'Trial {trial_id}, {trial_labels[trial_id,:]}, perf: {trial_perfs[trial_id]}, {title}')
    else:
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

def plot_neuron_rates(model_name, cell_id, condn_phrase, condn_num, rates_data = None, cut_off = 50, baseline=False,
                      ax=None, title=None, all_models_dir='/home/nuttidalab/Documents/renee/all_DMS_models/'):
    """
    Plot the firing rates of a neuron across trials.
    
    Parameters:
    - model_name: Name of the model.
    - cell_id: ID of the neuron.
    - condn_phrase: Condition phrase for loading data.
    - condn_num: Condition number for loading data.
    """
    # Load the firing rates
    if rates_data is None:
        _, _, rates_data = ld.load_neural_data(model_name, condn_phrase, condn_num,
                                                    load_LFP=False, load_spikes=False, load_rates=True,
                                                    all_models_dir=all_models_dir)

    exc_ind, inh_ind = ld.get_celltype_label(model_name, all_models_dir=all_models_dir)
    cell_type = 'exc' if cell_id in exc_ind else 'inh'

    # behavioral data
    trial_labels, trial_perfs = ld.load_bhv_data(model_name, condn_phrase, condn_num,
                                                    all_models_dir=all_models_dir)

    # get timing info
    times_ms, times_real, fs_dict = ld.get_times_dict('ds', condn_phrase, condn_num, model_name=model_name,
                                                       all_models_dir=all_models_dir) #ds is fs=1000, ms

    if baseline:
        baseline_idx = get_fixation_baseline_times(times_ms)
        rates_data = baseline_norm_rate(rates_data, baseline_idx=baseline_idx)
        ylabel = 'Normalized Firing rate'

    _, colors = get_trialtype_colors()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    # plot firing rate for each trial type
    trial_types, trial_idxs = np.unique(trial_labels, axis=0, return_inverse=True)
    for i, trial_type in enumerate(trial_types):
        trials_idx = (trial_idxs == i)
        trials_rate = rates_data[cut_off:, cell_id, trials_idx]
        # trials_rate = rates_data[cut_off:, cell_id, trials_idx]

        # first just plot mean and sem
        mean_rate = np.mean(trials_rate, axis=1)
        sem_rate = stats.sem(trials_rate, axis=1)

        ax.plot(np.arange(cut_off,times_ms['T']), mean_rate, color=colors[i], label=f'{trial_type}')
        ax.fill_between(np.arange(cut_off,times_ms['T']), mean_rate-sem_rate, mean_rate+sem_rate, color=colors[i], alpha=0.2)

    # shade stim times
    ax.axvspan(times_ms['stim1_on'], times_ms['stim1_off'], color='gray', alpha=0.2)
    ax.axvspan(times_ms['stim2_on'], times_ms['stim2_off'], color='gray', alpha=0.2)

    # axes
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel(ylabel if baseline else 'Firing rate (Hz)')
    if title is not None:
        ax.set_title(f'{title}')
    else:
        ax.set_title(f'Avg cell {cell_id} fr ({cell_type})')
    ax.legend()
    ax.set_xlim(0, times_ms['T'])


def plot_neuron_rates_by_acc(model_name, cell_idx, condn_phrase, condn_num, rates_data = None, cut_off = 50, ax=None, title=None):
    """
    Plot the firing rates of a neuron across trials.
    
    Parameters:
    - model_name: Name of the model.
    - cell_id: ID of the neuron.
    - condn_phrase: Condition phrase for loading data.
    - condn_num: Condition number for loading data.
    """
    # Load the firing rates
    if rates_data is None:
        _, _, rates_data = ld.load_neural_data(model_name, condn_phrase, condn_num,
                                                    load_LFP=False, load_spikes=False, load_rates=True)

    exc_ind, inh_ind = ld.get_celltype_label(model_name)
    cell_type = 'exc' if cell_idx in exc_ind else 'inh'

    # behavioral data
    trial_labels, trial_perfs = ld.load_bhv_data(model_name, condn_phrase, condn_num)

    # get timing info
    times_ms, times_real, fs_dict = ld.get_times_dict('ds', condn_phrase, condn_num) #ds is fs=1000, ms

    _, colors = get_trialtype_colors()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    # plot firing rate for each trial type
    trial_types, trial_idxs = np.unique(trial_labels, axis=0, return_inverse=True)
    
    for i, trial_type in enumerate(trial_types):
        trials_idx = (trial_idxs == i)
        trials_rate = rates_data[cut_off:, cell_idx, trials_idx]
        # trials_rate = rates_data[cut_off:, cell_id, trials_idx]

        # get correct and incorrect trials
        correct_trials = np.where(trial_perfs[trials_idx] == 1)[0]
        incorrect_trials = np.where(trial_perfs[trials_idx] == 0)[0]

        mean_corr = np.mean(trials_rate[:, correct_trials], axis=1)
        sem_corr = stats.sem(trials_rate[:, correct_trials], axis=1)
        mean_inc = np.mean(trials_rate[:, incorrect_trials], axis=1)
        sem_inc = stats.sem(trials_rate[:, incorrect_trials], axis=1)
        # plot firing rate for correct trials
        ax.plot(np.arange(cut_off,times_ms['T']), mean_corr, color=colors[i], label=f'{trial_type} correct')
        ax.fill_between(np.arange(cut_off,times_ms['T']), mean_corr-sem_corr, mean_corr+sem_corr, color=colors[i], alpha=0.2)
        # plot firing rate for incorrect trials
        ax.plot(np.arange(cut_off,times_ms['T']), mean_inc, color=colors[i], linestyle='--', label=f'{trial_type} incorrect')
        ax.fill_between(np.arange(cut_off,times_ms['T']), mean_inc-sem_inc, mean_inc+sem_inc, color=colors[i], alpha=0.2)

    # shade stim times
    ax.axvspan(times_ms['stim1_on'], times_ms['stim1_off'], color='gray', alpha=0.2)
    ax.axvspan(times_ms['stim2_on'], times_ms['stim2_off'], color='gray', alpha=0.2)

    # axes
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Firing rate (Hz)')
    if title is not None:
        ax.set_title(f'{title}')
    else:
        ax.set_title(f'Cell {cell_idx} fr ({cell_type})')
    # ax.legend()
    ax.set_xlim(0, times_ms['T'])


def plot_neuron_rates_by_acc_bootstrap(model_name, cell_idx, condn_phrase, condn_num, rates_data = None, 
                                       nboot=1000, CI_int=(2.5, 97.5), random_seed=42,
                                       cut_off = 50, ax=None, title=None, plot=1):
    """
    Plot the firing rates of a neuron across trials.
    
    Parameters:
    - model_name: Name of the model.
    - cell_id: ID of the neuron.
    - condn_phrase: Condition phrase for loading data.
    - condn_num: Condition number for loading data.
    """
    # Load the firing rates
    if rates_data is None:
        _, _, rates_data = ld.load_neural_data(model_name, condn_phrase, condn_num,
                                                    load_LFP=False, load_spikes=False, load_rates=True)

    exc_ind, inh_ind = ld.get_celltype_label(model_name)
    cell_type = 'exc' if cell_idx in exc_ind else 'inh'

    # behavioral data
    trial_labels, trial_perfs = ld.load_bhv_data(model_name, condn_phrase, condn_num)

    # get timing info
    times_ms, times_real, fs_dict = ld.get_times_dict('ds', condn_phrase, condn_num) #ds is fs=1000, ms

    _, colors = get_trialtype_colors()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    # plot firing rate for each trial type
    trial_types, trial_idxs = np.unique(trial_labels, axis=0, return_inverse=True)
    
    for i, trial_type in enumerate(trial_types):
        trials_idx = (trial_idxs == i)
        trials_rate = rates_data[cut_off:, cell_idx, trials_idx]
        # trials_rate = rates_data[cut_off:, cell_id, trials_idx]

        # get correct and incorrect trials
        correct_trials = np.where(trial_perfs[trials_idx] == 1)[0]
        incorrect_trials = np.where(trial_perfs[trials_idx] == 0)[0]
        
        # bootstrap method
        t1_avg, t1_CI, t2_avg, t2_CI, diff_avg, diff_CI, p_diff = fnc_time_bootstrap_optimized_retX(trials_rate[:, correct_trials].T, 
                                                                                       trials_rate[:, incorrect_trials].T, 
                                                                                       nboot, CI_int, random_seed=random_seed)
        # # now plot
        # ax.plot(np.arange(cut_off,times_ms['T']), t1_avg, color=colors[i], label=f'{trial_type} correct')
        # ax.fill_between(np.arange(cut_off,times_ms['T']), t1_CI[:,0], t1_CI[:,1], color=colors[i], alpha=0.2)
        # ax.plot(np.arange(cut_off,times_ms['T']), t2_avg, color=colors[i], linestyle='--', label=f'{trial_type} incorrect')
        # ax.fill_between(np.arange(cut_off,times_ms['T']), t2_CI[:,0], t2_CI[:,1], color=colors[i], alpha=0.2)

        # now plot difference
        ax.plot(np.arange(cut_off,times_ms['T']), diff_avg, color=colors[i], label=f'{trial_type} diff')
        ax.fill_between(np.arange(cut_off,times_ms['T']), diff_CI[:,0], diff_CI[:,1], color=colors[i], alpha=0.2)
        # print p-value
        print(f'Cell {cell_idx}, trial type {trial_type}, p-value: {p_diff.mean()}')


    # shade stim times
    ax.axvspan(times_ms['stim1_on'], times_ms['stim1_off'], color='gray', alpha=0.2)
    ax.axvspan(times_ms['stim2_on'], times_ms['stim2_off'], color='gray', alpha=0.2)
    # axes
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Firing rate (Hz)')
    if title is not None:
        ax.set_title(f'{title}')
    else:
        ax.set_title(f'Cell {cell_idx} fr ({cell_type})')
    # ax.legend()
    ax.set_xlim(0, times_ms['T'])
    
    # return t1_CI, t2_CI, t1_avg, t2_avg, p_diff


def plot_trialtype_meanrates(model_name, condn_phrase, condn_num, rates_data = None, sort=None,
                     cut_off = 50, hlines = [], normalize=False):
    """
    Plot the mean firing rates of all neurons across all trial types.
    
    Parameters:
    - model_name: Name of the model.
    - cell_id: ID of the neuron.
    - condn_phrase: Condition phrase for loading data.
    - condn_num: Condition number for loading data.
    """
    # Load the firing rates
    if rates_data is None:
        _, _, rates_data = ld.load_neural_data(model_name, condn_phrase, condn_num,
                                                    load_LFP=False, load_spikes=False, load_rates=True)
    n_cells = rates_data.shape[1]

    # behavioral data
    trial_labels, trial_perfs = ld.load_bhv_data(model_name, condn_phrase, condn_num)

    # get timing info
    times_ms, times_real, fs_dict = ld.get_times_dict('ds', condn_phrase, condn_num) #ds is fs=1000, ms

    # plot firing rate for each trial type
    trial_types, trial_idxs = np.unique(trial_labels, axis=0, return_inverse=True)
    cell_meanfrs = np.zeros((n_cells, len(trial_types), int(times_ms['T'] - cut_off)))

    for i, trial_type in enumerate(trial_types):
        trials_idx = (trial_idxs == i)
        trials_rate = rates_data[cut_off:, :, trials_idx]  # shape: [time, cell, trial]
        cell_meanfrs[:, i, :] = np.mean(trials_rate, axis=2).T  # mean over trials, transpose to match shape

    if sort is None:
        sort = np.arange(n_cells)

    if normalize:
        print(cell_meanfrs.shape)
        cell_meanfrs = stats.zscore(cell_meanfrs, axis=2)
        vmin=-2; vmax=2; cmap='bwr'
        # baseline = np.mean(cell_meanfrs[:,:,cut_off:int(times_ms['stim1_on'])], axis=2)
        # cell_meanfrs = (cell_meanfrs - baseline[:,:,np.newaxis])
        # vmin=-5; vmax=5; cmap='bwr'
    else:
        vmin=0; vmax=40; cmap='Greys'
           
    
    fig, axs = plt.subplots(2,2, figsize=(16, 8))
    axs = axs.flatten()
    for i, trial_type in enumerate(trial_types):
        axs[i].imshow(cell_meanfrs[sort, i, :], aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        axs[i].set_title(f'Trial type {trial_type}')
        axs[i].set_xlabel('Time (ms)')
        axs[i].set_ylabel('Cell ID')
        axs[i].set_xticks(np.arange(cut_off, times_ms['T'], 200))
        axs[i].set_xticklabels(np.arange(cut_off, times_ms['T'], 200))
        axs[i].set_yticks(np.arange(0, n_cells, 10))
        axs[i].set_yticklabels(np.arange(0, n_cells, 10))
        axs[i].set_xlim(0, times_ms['T'] - cut_off)
        axs[i].set_ylim(0, n_cells)

        if len(hlines) > 0:
            for hline in hlines:
                axs[i].axhline(y=hline, color='k', linestyle='--')

        # shade stimulus periods
        stim_colors = get_stim_plotting_colors(trial_type)
        axs[i].axvspan(times_ms['stim1_on']-cut_off, times_ms['stim1_off']-cut_off, color=stim_colors[0], alpha=0.3)
        axs[i].axvspan(times_ms['stim2_on']-cut_off, times_ms['stim2_off']-cut_off, color=stim_colors[1], alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_trialtype_meanrates_by_acc(model_name, condn_phrase, condn_num, rates_data = None, sort=None,
                     cut_off = 50, ax=None, title=None):
    """
    Plot the mean firing rates of all neurons across all trial types.
    
    Parameters:
    - model_name: Name of the model.
    - cell_id: ID of the neuron.
    - condn_phrase: Condition phrase for loading data.
    - condn_num: Condition number for loading data.
    """
    # Load the firing rates
    if rates_data is None:
        _, _, rates_data = ld.load_neural_data(model_name, condn_phrase, condn_num,
                                                    load_LFP=False, load_spikes=False, load_rates=True)
    n_cells = rates_data.shape[1]

    # behavioral data
    trial_labels, trial_perfs = ld.load_bhv_data(model_name, condn_phrase, condn_num)

    # get timing info
    times_ms, times_real, fs_dict = ld.get_times_dict('ds', condn_phrase, condn_num) #ds is fs=1000, ms

    # plot firing rate for each trial type
    trial_types, trial_idxs = np.unique(trial_labels, axis=0, return_inverse=True)
    cell_meanfrs_corr = np.zeros((n_cells, len(trial_types), int(times_ms['T'] - cut_off)))
    cell_meanfrs_inc = np.zeros((n_cells, len(trial_types), int(times_ms['T'] - cut_off)))

    for i, trial_type in enumerate(trial_types):
        # get correct and incorrect trials
        trials_idx_corr = np.where((trial_idxs == i) & (trial_perfs == 1))[0]
        trials_idx_inc = np.where((trial_idxs == i) & (trial_perfs == 0))[0]
        
        cell_meanfrs_corr[:,i,:] = np.mean(rates_data[cut_off:, :, trials_idx_corr], axis=2).T  # mean over trials, transpose to match shape
        cell_meanfrs_inc[:,i,:] = np.mean(rates_data[cut_off:, :, trials_idx_inc], axis=2).T  # mean over trials, transpose to match shape

    if sort is None:
        sort = np.arange(n_cells)
    
    fig, axs = plt.subplots(3,4, figsize=(16, 12))
    for i, trial_type in enumerate(trial_types):
        axs[0,i].imshow(cell_meanfrs_corr[sort, i, :], aspect='auto', cmap='Grays', vmin=0, vmax=40)
        axs[1,i].imshow(cell_meanfrs_inc[sort, i, :], aspect='auto', cmap='Grays', vmin=0, vmax=40)
        axs[2,i].imshow(cell_meanfrs_corr[sort, i, :] - cell_meanfrs_inc[sort, i, :], aspect='auto', cmap='bwr', vmin=-8, vmax=8)
        axs[0,i].set_title(f'Trial type {trial_type} correct')
        axs[1,i].set_title(f'Trial type {trial_type} incorrect')
        axs[2,i].set_title(f'Trial type {trial_type} diff (corr - inc)')
        for j in range(3):
            # shade stimulus periods
            stim_colors = get_stim_plotting_colors(trial_type)
            axs[j,i].axvspan(times_ms['stim1_on']-cut_off, times_ms['stim1_off']-cut_off, color=stim_colors[0], alpha=0.3)
            axs[j,i].axvspan(times_ms['stim2_on']-cut_off, times_ms['stim2_off']-cut_off, color=stim_colors[1], alpha=0.3)
        axs[j,i].set_xlabel('Time (ms)')
        axs[j,i].set_ylabel('Cell ID')
        axs[j,i].set_xticks(np.arange(cut_off, times_ms['T'], 500))
        axs[j,i].set_xticklabels(np.arange(cut_off, times_ms['T'], 500))
        axs[j,i].set_yticks(np.arange(0, n_cells, 10))
        # axs[i,j].set_yticklabels(np.arange(0, n_cells, 10))
        axs[j,i].set_xlim(0, times_ms['T'] - cut_off)
        axs[j,i].set_ylim(0, n_cells)
    plt.tight_layout()
    plt.show()



'''
FUNCTIONAL SUBPOPULATION FUNCTIONS
'''

def calc_subpop(model_name, condn_phrase, condn_num, method= 'rate_dist', rates_data=None, plot=False):
    """
    for this model + condition, get the subpopulation preference for all neurons
    method = 'rate_dist' or 'tuning'

    """
    if rates_data is None:
        _, _, rates_data = ld.load_neural_data(model_name, condn_phrase, condn_num,
                                                    load_LFP=False, load_spikes=False, load_rates=True)
    # behavioral data
    trial_labels, trial_perfs = ld.load_bhv_data(model_name, condn_phrase, condn_num)

    # timing data
    times_ms, _,_ = ld.get_times_dict('ds', condn_phrase, condn_num)


    if method == 'rate_dist':
        # restructure rate data 
        # (matrix of just mean fr for each neuron for each trial type: time x neurons x 4 trial types), excluding fixation period
        stim1_on = int(times_ms['stim1_on'])
        t = int(rates_data.shape[0] - stim1_on)
        baseline_idx = [int(times_ms['stim1_on']/2), int(times_ms['stim1_on'])] 
        rates_data = baseline_norm_rate(rates_data, baseline_idx=baseline_idx) # baseline-norm each trial
        rates_mean = np.zeros((t, rates_data.shape[1], 4))
        trialtype_idxs = np.zeros((t*4))
        trial_types, trial_idxs = np.unique(trial_labels, axis=0, return_inverse=True)
        for i, trial_type in enumerate(trial_types):
            trials_idx = (trial_idxs == i)
            trialtype_idxs[i*t:(i+1)*t] = i
            # get mean rate
            rates_mean[:,:, i] = np.mean(rates_data[stim1_on:, :, trials_idx], axis=2)
        rates_mean = rates_mean.transpose(2,0,1).reshape(-1, rates_mean.shape[1]) # concatenate 4 trial types one after another in time
        N = rates_mean.shape[1]

        # z-score the rates, then get distances
        # rate_features = get_rate_features(rates_mean.T, normalize=True) #zscore
        # rate_dist = pairwise_distances(rate_features, metric='euclidean')
        rate_dist = pairwise_distances(rates_mean, metric='euclidean')
        rate_dist /= np.max(rate_dist)

        # UMAP on the rate distances
        embedding = embed_umap(rate_dist, dim=20)

        # KMeans on the embedding
        k, labels = get_kmeans_clusters(embedding, n_clusters=None, plot=plot)
        
    elif method == 'tuning':
        labels = calc_stim1_tuning(model_name, condn_phrase, condn_num, rates_data=rates_data)

    else:
        raise ValueError("Method not defined, must be 'rate_dist' or 'tuning'")

    return labels


def baseline_norm_rate(r, baseline_idx=None):
    '''
    normalize firing rates to z-scores
    r: firing rates, shape (n_neurons, T), or (T, n_neurons, n_trials)
    baseline: if None, use the mean of the first 100 ms as baseline
    '''
    n_dims = len(r.shape)
    if n_dims == 2:
        # r is (n_neurons, T)
        if baseline_idx is None:
            baseline_mean = r[:, :100].mean(axis=1, keepdims=True)
            baseline_std = r[:, :100].std(axis=1, keepdims=True)
        else:
            baseline_mean = r[:, baseline_idx[0]:baseline_idx[1]].mean(axis=1, keepdims=True)
            baseline_std = r[:, baseline_idx[0]:baseline_idx[1]].std(axis=1, keepdims=True)
        # do z-score normalization
        r = (r - baseline_mean) / (baseline_std + 1e-10)

    elif n_dims == 3:
        # r is (T, n_neurons, n_trials)
        if baseline_idx is None:
            baseline_mean = r[:100,:,:].mean(axis=0, keepdims=True)
            baseline_std = r[:100,:,:].std(axis=0, keepdims=True)
        else:
            baseline_mean = r[baseline_idx[0]:baseline_idx[1],:,:].mean(axis=0, keepdims=True)
            baseline_std = r[baseline_idx[0]:baseline_idx[1],:,:].std(axis=0, keepdims=True)
        # do z-score normalization
        r = (r - baseline_mean) / (baseline_std + 1e-10)

    return r 



def get_rate_features(r, normalize=True):
    '''
    normalize firing rates to z-scores
    r: firing rates, shape (n_neurons, T)
    normalize: if True, normalize to z-scores
    '''
    if normalize:
        r = (r - r.mean(axis=1, keepdims=True)) / (r.std(axis=1, keepdims=True) + 1e-8)
    return r  # shape (200, T)


def embed_umap(dist_matrix, dim=2):
    umap = UMAP(n_components=dim, random_state=42)
    embedding = umap.fit_transform(dist_matrix)
    return embedding


def get_kmeans_clusters(embedding, n_clusters=None, cluster_range=range(2,11), plot=False):
    """
    Get the KMeans clusters for the given embedding.
    
    Parameters:
    - embedding: The UMAP embedding of the data.
    - n_clusters: The number of clusters to use. If None, it will be determined using the silhouette score.
    
    Returns:
    - n_clusters: The best number of clusters determined by silhouette score (or provided).
    - labels: The cluster labels for each point in the embedding.
    """
    if n_clusters is None:
        silhouette_scores = []
        for n in cluster_range:
            kmeans = KMeans(n_clusters=n, random_state=42)
            labels = kmeans.fit_predict(embedding)
            silhouette_scores.append(silhouette_score(embedding, labels))
        n_clusters = np.argmax(silhouette_scores) + cluster_range.start

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embedding)

    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].plot(cluster_range, silhouette_scores)
        axs[0].axvline(x=n_clusters, color='r', linestyle='--')
        axs[0].set_title("Silhouette Score vs. Number of Clusters")
        axs[0].set_xlabel("Number of Clusters")
        axs[0].set_ylabel("Silhouette Score")
        axs[0].grid(True)
        
        scatter = axs[1].scatter(embedding[:, 0], embedding[:, 1], c=labels, s=30)
        axs[1].set_title(f"KMeans Clustering")
        axs[1].set_xlabel("Dim 1")
        axs[1].set_ylabel("Dim 2")
        axs[1].grid(True)
        plt.colorbar(scatter)

    return n_clusters, labels


def plot_rates_by_cluster(model_name, condn_phrase, condn_num, labels, rates_data=None):

    if rates_data is None:
        _, _, rates_data = ld.load_neural_data(model_name, condn_phrase, condn_num,
                                                    load_LFP=False, load_spikes=False, load_rates=True)
    sort = np.argsort(labels)
    _, hlines = np.unique(np.sort(labels), return_index=True)
    plot_trialtype_meanrates(model_name, condn_phrase, condn_num, rates_data = rates_data, cut_off=50, sort=sort, normalize=True,
                                hlines = hlines)


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

def get_fixation_baseline_times(times_dict):
    """
    Get the baseline period for the given times dictionary.
    """
    baseline = [int(times_dict['stim1_on']/2), int(times_dict['stim1_on'])]
    return baseline