import numpy as np 
import sys
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import scipy
import os
import importlib
import pandas as pd

sys.path.append('/home/nuttidalab/Documents/renee/spikeRNN/analysis/utils')
import load_data as ld 
import single_neuron_utils as sn 
import bhv_utils as bhv
import lfp_utils as lu

from scipy.stats import circmean, circstd

def get_neuron_sfc(neuron_df, lfp_phase, fs_spk, fs_lfp, times_fs, threshold=2):
    """
    Compute spike-field coupling for a single neuron. Focus only on delay period.
    neuron_df: DataFrame (spike_df) for a single neuron)
    lfp_phase: array of shape (n_trials, n_times) with LFP phase values
    threshold: threshold in Hertz for number of spikes during delay period to include neuron
    """
    
    n_trials, n_times = lfp_phase.shape
    spike_trial_plvs = np.zeros(n_trials) # PLV for each trial
    spike_trial_phases = np.zeros(n_trials) # mean phase for each trial
    n_trial_spikes = np.zeros(n_trials) # number of spikes in each trial
    
    i = 0
    for _, row in neuron_df.iterrows(): # iterate over trials for this neuron
        spk_times = np.round(row['spk_times'] * fs_lfp / fs_spk).astype(int) # convert spike times to LFP time indices
        trial_id = row['trial_id']
        
        # focus only on delay period
        spk_times = spk_times[(spk_times >= times_fs['stim1_off']) & (spk_times <= times_fs['stim2_on'])]
        n_trial_spikes[i] = len(spk_times)
        
        # filter by spike threshold
        if n_trial_spikes[i] < (threshold * (times_fs['stim2_on'] - times_fs['stim1_off']) / fs_lfp): 
            spike_trial_phases[i] = np.nan
            spike_trial_plvs[i] = np.nan
            i += 1
            continue
        
        # get phase values at these spike times in this trial
        phase_values = lfp_phase[trial_id, spk_times]
        spike_trial_phases[i] = circmean(phase_values, high=np.pi, low=-np.pi) # mean phase for this trial
        spike_trial_plvs[i] = np.abs(np.mean(np.exp(1j * phase_values))) # PLV for this trial
        
        i += 1

    return spike_trial_phases, spike_trial_plvs, n_trial_spikes


def get_null_sfc(n_trial_spikes, lfp_phase, times_fs, n_shuffles=100):
    # n_trial_spikes is (trials x 1) array of number of spikes in each trial
    # lfp_phase is (trials x time) array of LFP phase values
    # n_shuffles is number of shuffles to perform for null distribution
    
    n_trials = len(n_trial_spikes)
    null_trial_plvs = np.zeros((n_trials, n_shuffles)) # null PLVs for each trial and shuffle
    null_trial_meanphases = np.zeros((n_trials, n_shuffles)) # null mean phases for each trial and shuffle
    
    for tr in range(n_trials):
        n_spikes = int(n_trial_spikes[tr])
        if n_spikes <= 1:
            null_trial_plvs[tr, :] = np.nan
            null_trial_meanphases[tr, :] = np.nan
            continue
        for s in range(n_shuffles):
            shuffled_spk_times = np.random.choice(np.arange(times_fs['stim1_off'], times_fs['stim2_on']), size=n_spikes, replace=False).astype(int) # only generate during delay period
            shuffled_phase_values = lfp_phase[tr, shuffled_spk_times]
            null_trial_plvs[tr, s] = np.abs(np.mean(np.exp(1j * shuffled_phase_values))) # PLV for this shuffle
            null_trial_meanphases[tr, s] = circmean(shuffled_phase_values) # mean phase for this shuffle

    return null_trial_plvs, null_trial_meanphases


def compute_zscore_plvs(spike_trial_plvs, null_trial_plvs):
    '''
    spike_trial_plvs: array of shape (n_neurons, n_trials,) with spike PLVs for each trial
        - some will be NaN if not enough spikes
    null_trial_plvs: array of shape (n_neurons, n_trials, n_shuffles) with null PLVs for each trial and shuffle
        - some will be NaN if not enough spikes
    returns: zscore_plvs: array of shape (n_neurons, n_trials) with z-scored PLVs for each trial
    '''
    
    # compute z-score of spike PLVs relative to null distribution
    mean_null_plvs = np.nanmean(null_trial_plvs, axis=2) # mean across shuffles for each trial
    std_null_plvs = np.nanstd(null_trial_plvs, axis=2) # std across shuffles for each trial
    
    zscore_plvs = (spike_trial_plvs - mean_null_plvs) / std_null_plvs
    return zscore_plvs