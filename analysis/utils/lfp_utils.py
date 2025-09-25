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


def calc_band_power(s, f, band='all', zscore_band=True, baseline_times=None):
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


def compute_lfp_power(epsps, fs, freq_range = [4, 100], num_freqs = 40, n_cycles=7):
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

    # function takes (n_trials, n_channels, n_times)
    coeffs = tfr_array_morlet(epsps[:,np.newaxis,:], sfreq=fs, freqs=freqs, n_cycles=n_cycles, output='complex') # coeffs is (n_trials, n_channels, n_freqs, n_times)
    coeffs = np.squeeze(coeffs) # (n_trials, n_freqs, n_times)
    
    power = np.abs(coeffs)**2  # power is (n_trials, n_freqs, n_times)
    phase = np.angle(coeffs)  # phase is (n_trials, n_freqs, n_times)

    return np.array(power), freqs, np.array(phase)

def plot_trial_bandpower(lfp_data, trial_id, condn_phrase, condn_num, 
                            ax=None, 
                            all_models_dir='/home/nuttidalab/Documents/renee/all_DMS_models/'):
    return 0