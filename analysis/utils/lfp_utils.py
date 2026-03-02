# Functions for calculating spectrogram using wavelet transform

import numpy as np
from scipy.signal import spectrogram, resample, iirnotch, filtfilt
from mne.time_frequency import tfr_array_morlet
import matplotlib.pyplot as plt
import scipy.io
import load_data as ld
from bootstrap_method import *

import pdb


def get_band_ranges():
    # Returns the frequency ranges for different bands
    # theta: 4-8Hz
    # alpha: 9-12Hz
    # beta: 13-30Hz
    # gamma1: 30-55Hz
    # gamma2: 70-140Hz
    freq_bands = {'theta': [4,8], 'alpha': [9,12], 'beta': [13,30], 'gamma1': [30,55], 'gamma2': [70,140]}
    return freq_bands


def calc_band_power(s, f, band='all', zscore_band=False, baseline_times=None):
    """
    Calculate the band power for a specific frequency band.

    Parameters
    ----------
    s : np.ndarray
        The power spectrogram. Shape [N_trials, F, T].
    f : np.ndarray
        The frequency bins. Shape [F].
    band : list
        The frequency band to calculate power for. Should be a list of [low_freq, high_freq], or 'all','theta','alpha','beta','gamma1','gamma2'.

    Returns
    -------
    band_power : float
        The average power in the specified frequency band. Shape [N_trials, T] or dict of [N_trials, T] if band='all'.
    """

    band_ranges = get_band_ranges()
    if band == 'all':
        band_power = {}
        band_sem = {}
        for band in band_ranges:
            f_band_idx = np.where((f >= band_ranges[band][0]) & (f <=band_ranges[band][1]))[0]
            band_power[band] = np.nanmean(s[:, f_band_idx, :], axis=1) # (N_trials, T)
            # print(band_power[band].shape, f_band_idx)
            band_sem[band] = np.nanstd(s[:, f_band_idx, :], axis=2)/np.sqrt(s[:, f_band_idx,:].shape[1])
            if baseline_times is not None and zscore_band:
                b0, b1 = baseline_times
                baseline = band_power[band][:, np.arange(b0, b1)]
                mean = np.mean(baseline, axis=1, keepdims=True)
                std = np.std(baseline, axis=1, keepdims=True) + 1e-8
                band_power[band] = (band_power[band] - mean) / std
                # print(band_power[band].shape, b0, b1)
        return band_power, band_sem
    elif isinstance(band, str):
        f_band_idx = np.where((f >= band_ranges[band][0]) & (f <=band_ranges[band][1]))[0]
        band_power = np.nanmean(s[:, f_band_idx,:], axis=1)
        band_sem = np.nanstd(s[:, f_band_idx,:], axis=1)/np.sqrt(s[:, f_band_idx,:].shape[1])
    elif isinstance(band, (list, tuple)):
        if len(band) == 2:
            f_band_idx = np.where((f >= band[0]) & (f <=band[1]))[0]
            band_power = np.nanmean(s[:, f_band_idx,:], axis=1)
            band_sem = np.nanstd(s[:, f_band_idx,:], axis=1)/np.sqrt(s[:, f_band_idx,:].shape[1])
        elif len(band) == 1:
            f_band_idx = np.where((f >= band[0]-0.5) & (f <=band[0]+0.5))[0]
            band_power = np.nanmean(s[:, f_band_idx,:], axis=1)
            band_sem = np.nanstd(s[:, f_band_idx,:], axis=1)/np.sqrt(s[:, f_band_idx,:].shape[1])
    else:
        raise ValueError("band must be 'all', a band name, or a list/tuple of [low_freq, high_freq]")

    if zscore_band:
        if baseline_times is None or len(baseline_times) != 2:
            raise ValueError("baseline_times must be provided when zscore_input=True.")
        b0, b1 = baseline_times
        baseline = band_power[:, np.arange(b0, b1)]
        mean = np.mean(baseline, axis=1, keepdims=True)
        std = np.std(baseline, axis=1, keepdims=True) + 1e-8
        band_power = (band_power - mean) / std

    return band_power, band_sem


def compute_lfp_power(epsps, fs, freq_range = [4, 100], num_freqs = 40, n_cycles='varied'):
    """
    Compute LFP power in specified frequency range using Morlet wavelet transform.
    
    Parameters:
    epsps : array
        The excitatory post-synaptic potentials (shape: trials, n_timepoints).
    settings : dict
        Dictionary containing settings such as 'fs' (sampling rate).
    freq_range : list
        List of frequencies to compute power for.
    num_freqs : int
        Number of frequency bins.
    n_cycles : int
        Number of cycles for the Morlet wavelet.
    """
    
    freqs = np.geomspace(freq_range[0], freq_range[1], num_freqs)
    if isinstance(n_cycles, str) and n_cycles == 'varied':
        n_cycles = np.geomspace(3, 10, len(freqs)).astype(np.float32)
        

    # function takes (n_trials, n_channels, n_times)
    coeffs = tfr_array_morlet(epsps[:,np.newaxis,:], sfreq=fs, freqs=freqs, n_cycles=n_cycles, output='complex') # coeffs is (n_trials, n_channels, n_freqs, n_times)
    coeffs = np.squeeze(coeffs) # (n_trials, n_freqs, n_times)
    
    power = np.abs(coeffs)**2  # power is (n_trials, n_freqs, n_times)
    log_power = np.log(power + 1e-10)  # log-power, add small constant to avoid log(0)
    phase = np.angle(coeffs)  # phase is (n_trials, n_freqs, n_times)

    return np.array(log_power), freqs, np.array(phase)


def compute_bandpower(power, freqs, settings, band=[4], zscore=True, baseline=[10, 50]):
    """
    Compute average power in a specified frequency band.
    
    Parameters:
    power : array
        Power values (shape: trials, n_freqs, n_times).
    settings : dict
        Dictionary containing settings such as 'fs' (sampling rate).
    band : list
        Frequency band to compute power for.
    """
    # fs = settings['fs']
    # num_freqs = power.shape[1]
    
    # find indices of frequencies within the band
    if len(band) == 1:
        band_indices = np.where((freqs >= band[0]-0.5) & (freqs <= band[0]+0.5))[0]
    elif len(band) == 2:
        band_indices = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
    else:
        raise ValueError("Band must be a list of one or two elements.")
    
    # average power over the band frequencies
    band_power = np.mean(power[:, band_indices, :], axis=1)  # shape: (n_trials, n_times)
    
    if zscore:
        # z-score normalization using baseline period
        baseline_indices = np.where((np.arange(power.shape[2]) >= baseline[0]) & (np.arange(power.shape[2]) <= baseline[1]))[0]
        baseline_mean = np.mean(band_power[:, baseline_indices], axis=1, keepdims=True)
        baseline_std = np.std(band_power[:, baseline_indices], axis=1, keepdims=True)
        band_power = (band_power - baseline_mean) / baseline_std
    
    return band_power  # shape: (n_trials, n_times)


def plot_bandpower_bootstrap(model_name, bands1, bands2, condn_phrase, condn_num='', 
                            nboot=1000, CI_int=(2.5, 97.5), random_seed=820,
                            ax=None, title=None, colors = ['green', 'red'],
                            all_models_dir='/home/nuttidalab/Documents/renee/all_DMS_models/'):
    """
    bands1 is trials x time (e.g. correct trials)
    bands2 is trials x time (e.g. incorrect trials)
    
    """

    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4))
    
    times_fs, times_real, fs_dict = ld.get_times_dict('ds', condn_phrase, condn_num, 
                   model_name=model_name, all_models_dir=all_models_dir)
    trial_labels, trial_perfs = ld.load_bhv_data(model_name, condn_phrase, condn_num, all_models_dir=all_models_dir)
    
    t1_avg, t1_CI, t2_avg, t2_CI, _, _, p_diff = fnc_time_bootstrap_optimized_retX(bands1, bands2,
                                                            nboot, CI_int, random_seed=random_seed)
    ax.plot(t1_avg.mean(axis=1), label=f'Correct trials, prev_n={bands1.shape[0]}', color=colors[0])
    ax.fill_between(np.arange(bands1.shape[1]), 
                    t1_CI[:,0], t1_CI[:,1], alpha=0.3, color=colors[0])
    ax.plot(t2_avg.mean(axis=1), label=f'Incorrect trials, prev_n={bands2.shape[0]}', color=colors[1])
    ax.fill_between(np.arange(bands2.shape[1]), 
                        t2_CI[:,0], t2_CI[:,1], alpha=0.3, color=colors[1])
    ax.scatter(np.where(p_diff < 0.05)[0], np.ones(np.sum(p_diff < 0.05))*np.max(t1_avg)*1.1, 
               marker='s', s=10, color='black', label='p<0.05')
    ax.axvspan(times_fs['stim1_on'], times_fs['stim1_off'], color='gray', alpha=0.3)
    ax.axvspan(times_fs['stim2_on'], times_fs['stim2_off'], color='gray', alpha=0.3)
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f'Model {model_name[-6:]}, perf: {np.mean(trial_perfs):.2f}')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Band power')
    # ax.legend()
    
    return ax, np.mean(t1_avg, axis=1), np.mean(t2_avg, axis=1), t1_CI, t2_CI, p_diff
    
    
    
def plot_bandpower_rate_bootstrap(model_name, bands1, bands2, condn_phrase, condn_num='', 
                            nboot=1000, CI_int=(2.5, 97.5), random_seed=820,
                            ax=None, title=None, colors = ['green', 'red'],
                            all_models_dir='/home/nuttidalab/Documents/renee/all_DMS_models/'):
    """
    bands1 is trials x time (e.g. correct trials)
    bands2 is trials x time (e.g. incorrect trials)
    
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4))
        
    trial_labels, trial_perfs, _ = ld.load_bhv_rate_data(model_name, condn_phrase, condn_num, all_models_dir=all_models_dir)
    
    t1_avg, t1_CI, t2_avg, t2_CI, _, _, p_diff = fnc_time_bootstrap_optimized_retX(bands1, bands2,
                                                            nboot, CI_int, random_seed=random_seed)
    ax.plot(t1_avg.mean(axis=1), label=f'Low load, prev_n={bands1.shape[0]}', color=colors[0])
    ax.fill_between(np.arange(bands1.shape[1]), 
                    t1_CI[:,0], t1_CI[:,1], alpha=0.3, color=colors[0])
    ax.plot(t2_avg.mean(axis=1), label=f'High load, prev_n={bands2.shape[0]}', color=colors[1])
    ax.fill_between(np.arange(bands2.shape[1]), 
                        t2_CI[:,0], t2_CI[:,1], alpha=0.3, color=colors[1])
    ax.scatter(np.where(p_diff < 0.05)[0], np.ones(np.sum(p_diff < 0.05))*np.max(t1_avg)*1.1, 
               marker='s', s=10, color='black', label='p<0.05')
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f'Model {model_name[-6:]}, perf: {np.mean(trial_perfs):.2f}')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Band power')
    # ax.legend()
    
    return ax, np.mean(t1_avg, axis=1), np.mean(t2_avg, axis=1), t1_CI, t2_CI, p_diff


def plot_bandpower_rate_loaddiff_bootstrap(bands1, bands2,
                            nboot=1000, CI_int=(2.5, 97.5), random_seed=820,
                            ax=None, title=None, colors = ['green', 'red'],
                            bands1_label = 'Low load', bands2_label = 'High load'):
    """
    bands1 is models x time (e.g. low load)
    bands2 is models x time (e.g. high load)
    
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4))
    
    t1_avg, t1_CI, t2_avg, t2_CI, diff_avg, diff_CI, p_diff = fnc_time_bootstrap_optimized_retX(bands1, bands2,
                                                            nboot, CI_int, random_seed=random_seed)
    ax.plot(t1_avg.mean(axis=1), label=f'{bands1_label}', color=colors[0])
    ax.fill_between(np.arange(bands1.shape[1]), 
                    t1_CI[:,0], t1_CI[:,1], alpha=0.3, color=colors[0])
    ax.plot(t2_avg.mean(axis=1), label=f'{bands2_label}', color=colors[1])
    ax.fill_between(np.arange(bands2.shape[1]), 
                        t2_CI[:,0], t2_CI[:,1], alpha=0.3, color=colors[1])
    # significance markers above the max of the two curves
    ax.scatter(np.where(p_diff < 0.05)[0], np.ones(np.sum(p_diff < 0.05))*np.max([t1_avg.mean(axis=1), t2_avg.mean(axis=1)])*1.1, 
               marker='s', s=10, color='black', label='p<0.05')
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f'{bands1_label} vs {bands2_label}')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Band power')
    # ax.legend()
    
    return ax, np.mean(t1_avg, axis=1), np.mean(t2_avg, axis=1), t1_CI, t2_CI, np.mean(diff_avg, axis=1), diff_CI, p_diff


    
def plot_band_power_bootstrap(band_trials1, band_trials2, t, nboot=1000, CI_int=(2.5, 97.5), random_seed=820,
                              adjust_bands2=False, baseline_idx=[],
                              type1_label='exc neurons', type2_label='inh neurons', p_sig=0.05, title = [],
                              event_times = [], 
                              rt1=np.nan, rt2=np.nan, t_skip=5, 
                              band_range = [], time_range = [], c = []):
    '''
    band_trials is a dict with keys as bands and values as the band power (trials x time)
    f is the frequency vector
    t is the time vector
    event_times if given, is a dict with keys as event names and values as the time of the event
    rt1 and rt2 are the response times for the two trial types (if exists)
    band_range is a list of two numbers, the range of the y-axis for the band power plots
    time_range is a list of two numbers, the range of the x-axis for the band power plots
    c is a list of colors for the two trial types
    '''

    type_label = [type1_label,type2_label]
    if len(c) == 0:
        c = ['b','g','r','c','m','y','k']
    

    if len(time_range) == 2:
        time_start = [time_range[0] if ~np.isnan(time_range[0]) else t[0]][0]
        time_end = [time_range[1] if ~np.isnan(time_range[1]) else t[-1]][0]
        # time_start_idx = [np.where(t <= time_range[0])[0][-1] if ~np.isnan(time_range[0]) else 0][0]
        # time_end_idx = [np.where(t <= time_range[1])[0][-1] if ~np.isnan(time_range[1]) else len(t)][0]
    elif len(time_range) == 0:
        time_start = t[0]
        time_end = t[-1]
    
    # plot band powers separately, with each band as a subplot
    fig, axs = plt.subplots(len(band_trials1),1,figsize=(10,18), sharey=True)
    for j, band in enumerate(band_trials1.keys()):
        band1 = band_trials1[band]
        band2 = band_trials2[band]

        if adjust_bands2:
            band1_blmean = np.nanmean(band1[:,baseline_idx[0]:baseline_idx[1]])
            band2_blmean = np.nanmean(band2[:,baseline_idx[0]:baseline_idx[1]])
            diff = band1_blmean - band2_blmean
            band2 = band2 + diff

        t1_avg, t1_CI, t2_avg, t2_CI, _, _, p_diff = fnc_time_bootstrap_optimized_retX(band1, band2,
                                                                nboot=nboot, CI_int=CI_int, random_seed=random_seed)
        t1_avg = np.nanmean(t1_avg, axis=1)
        t2_avg = np.nanmean(t2_avg, axis=1)
        axs[j].plot(t, t1_avg, color = c[0], label=f'{band} {type_label[0]}')
        axs[j].fill_between(t, t1_CI[:,0], t1_CI[:,1], alpha=0.3, color = c[0])
        axs[j].plot(t, t2_avg, color = c[1], label=f'{band} {type_label[1]}')
        axs[j].fill_between(t, t2_CI[:,0], t2_CI[:,1], alpha=0.3, color = c[1])
        # tick_positions = range(len(t))[::t_skip]
        # tick_labels = np.round(t[::t_skip],3)
        # axs[j].set_xticks(tick_positions)
        # axs[j].set_xticklabels(tick_labels)
        # axs[j].set_xticks(np.arange(times['fixation'],times['end'],1))
        axs[j].set_title(f'{band} band power')
        for key in event_times:
            axs[j].axvline(x=event_times[key], color='r', linestyle='--')
        axs[j].set_xlim([time_start, time_end])
        if ~np.isnan(rt1):
            # axs[j].axvline(x=get_fft_time(rt1, t), color=c[0], linestyle='--', label='Response Time')
            axs[j].axvline(x=rt1, color=c[0], linestyle='--', label='Response Time')
        if ~np.isnan(rt2):
            # axs[j].axvline(x=get_fft_time(rt2, t), color=c[1], linestyle='--', label='Response Time')
            axs[j].axvline(x=rt2, color=c[1], linestyle='--', label='Response Time')
        # axs[j].set_xlim([time_start_idx, time_end_idx])
        axs[j].set_xlim([time_start, time_end])
        axs[j].legend()

        # time_vector = np.array(range(0, len(t1_avg)))
        significant_timepoints = t[p_diff < p_sig]
        if j == 0:
            ymin, ymax = axs[j].get_ylim()
        axs[j].scatter(significant_timepoints,
                        np.zeros_like(significant_timepoints) + ymax + (ymax-ymin)/10, color='k', label='_nolegend_', marker='s', s=20)
        
        # axs[j].set_ylim(ymin - (ymax-ymin)/10, ymax + (ymax-ymin)/5)

        if j == len(band_trials1)-1:
            axs[j].set_xlabel('time (s)')

        if len(band_range) > 0:
            if isinstance(band_range[0], int):
                # print('setting range')
                axs[j].set_ylim(band_range)
            else:
                axs[j].set_ylim(band_range[j])

    if len(title) > 0:
        fig.suptitle(title)
    
    # plt.savefig('/home/renee/WM_letters/results/2025_SummitAI/timecourse.svg', format='svg')

    plt.show()
    
    return