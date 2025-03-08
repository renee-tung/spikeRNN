# Functions for calculating spectrogram

import numpy as np
from scipy.signal import spectrogram, resample, iirnotch, filtfilt
import matplotlib.pyplot as plt
import scipy.io
# import load_data as ld
from bootstrap_method import *

import pdb

# # bootstrapping params
# random_seed = 820
# nboot = 1000
# CI_int = (2.5, 97.5)  # 95% confidence interval

def get_band_ranges():
    # Returns the frequency ranges for different bands
    # theta: 4-8Hz
    # alpha: 9-12Hz
    # beta: 13-30Hz
    # gamma1: 30-55Hz
    # gamma2: 70-140Hz
    freq_bands = {'theta': [4,8], 'alpha': [9,12], 'beta': [13,30], 'gamma1': [30,55], 'gamma2': [70,140]}
    return freq_bands

def calc_band_power(s, f, band='all'):
    # Calculates the power in each frequency band
    # s is the spectrogram (freq x time)
    # f is the frequency vector
    # band is the band to calculate power in
    # band can be 'all', 'theta', 'alpha', 'beta', 'gamma1', 'gamma2'

    band_ranges = get_band_ranges()
    if band == 'all':
        band_power = {}
        band_sem = {}
        for band in band_ranges:
            f_band_idx = np.where((f >= band_ranges[band][0]) & (f <=band_ranges[band][1]))[0]
            band_power[band] = np.nanmean(s[f_band_idx,:], axis=0)
            band_sem[band] = np.nanstd(s[f_band_idx,:], axis=0)/np.sqrt(s[f_band_idx,:].shape[0])
    elif isinstance(band, str):
        f_band_idx = np.where((f >= band_ranges[band][0]) & (f <=band_ranges[band][1]))[0]
        band_power = np.nanmean(s[f_band_idx,:], axis=0)
        band_sem = np.nanstd(s[f_band_idx,:], axis=0)/np.sqrt(s[f_band_idx,:].shape[0])

    return band_power, band_sem

def calc_band_power_multiple(s, f, band='all'):
    # Calculates the power in each frequency band
    # s is the spectrogram (trials x freq x time)
    # f is the frequency vector
    # band is the band to calculate power in
    # band can be 'all', 'theta', 'alpha', 'beta', 'gamma1', 'gamma2'

    band_ranges = get_band_ranges()
    if band == 'all':
        band_power = {}
        band_sem = {}
        for band in band_ranges:
            f_band_idx = np.where((f >= band_ranges[band][0]) & (f <=band_ranges[band][1]))[0]
            band_power[band] = np.nanmean(s[:,f_band_idx,:], axis=1)
            band_sem[band] = np.nanstd(s[:,f_band_idx,:], axis=1)/np.sqrt(s[:,f_band_idx,:].shape[1])
    elif isinstance(band, str):
        f_band_idx = np.where((f >= band_ranges[band][0]) & (f <=band_ranges[band][1]))[0]
        band_power = np.nanmean(s[:,f_band_idx,:], axis=1)
        band_sem = np.nanstd(s[:,f_band_idx,:], axis=1)/np.sqrt(s[:,f_band_idx,:].shape[1])

    return band_power, band_sem

def calc_band_power_pair(s1, s2, f, adjust_s2 = False, baseline_idx = []):
    # Calculates the power in each frequency band
    # s is the spectrogram (freq x time)
    # f is the frequency vector
    # band is the band to calculate power in
    # band can be 'all', 'theta', 'alpha', 'beta', 'gamma1', 'gamma2'

    band_ranges = get_band_ranges()
    # if band == 'all':
    band_power_1 = {}
    band_power_2 = {}
    for band in band_ranges:
        f_band_idx = np.where((f >= band_ranges[band][0]) & (f <=band_ranges[band][1]))[0]
        band1 = np.squeeze(np.nanmean(s1[f_band_idx,:], axis=0))
        band2 = np.squeeze(np.nanmean(s2[f_band_idx,:], axis=0))
        if adjust_s2:
            band1_blmean = np.nanmean(band1[baseline_idx[0]:baseline_idx[1]])
            band2_blmean = np.nanmean(band2[baseline_idx[0]:baseline_idx[1]])
            diff = band1_blmean - band2_blmean
            band2 = band2 + diff
        band_power_1[band] = band1
        band_power_2[band] = band2

    return band_power_1, band_power_2

def calc_band_power_bootstrap(s1, s2, f, band='all', nboot=1000, CI_int=(2.5, 97.5), random_seed=820,
                              adjust_s2 = False, baseline_idx = []):
    # Calculates the power in each frequency band using bootstrapping
    # s1 and s2 are the spectrograms (trials x freq x time)
    # f is the frequency vector
    # band is the band to calculate power in
    # band can be 'all', 'theta', 'alpha', 'beta', 'gamma1', 'gamma2'
    # nboot is the number of bootstraps
    # CI_int is the confidence interval bounds

    band_ranges = get_band_ranges()
    if band == 'all':
        band_err = {}
        band_pdiff = {}
        band_avg = {}
        for band in band_ranges:
            f_band_idx = np.where((f >= band_ranges[band][0]) & (f <=band_ranges[band][1]))[0]
            band1 = np.squeeze(np.nanmean(s1[:,f_band_idx,:], axis=1))
            band2 = np.squeeze(np.nanmean(s2[:,f_band_idx,:], axis=1))
            if adjust_s2:
                band1_blmean = np.nanmean(band1[:,baseline_idx[0]:baseline_idx[1]])
                band2_blmean = np.nanmean(band2[:,baseline_idx[0]:baseline_idx[1]])
                diff = band1_blmean - band2_blmean
                band2 = band2 + diff
            
            t1_avg, t1_CI, t2_avg, t2_CI, _, _, p_diff = fnc_time_bootstrap_optimized_retX(band1, band2, 
                                                                  nboot, CI_int, random_seed=random_seed)
            band_err[band] = [t1_CI, t2_CI]
            band_avg[band] = [t1_avg, t2_avg]
            band_pdiff[band] = p_diff
    elif isinstance(band, str):
        f_band_idx = np.where((f >= band_ranges[band][0]) & (f <=band_ranges[band][1]))[0]
        t1_avg, t1_CI, t2_avg, t2_CI, _, _, p_diff = fnc_time_bootstrap_optimized_retX(np.squeeze(np.nanmean(s1[:,f_band_idx,:], axis=1)), 
                                                            np.squeeze(np.nanmean(s2[:,f_band_idx,:], axis=1)), 
                                                            nboot, CI_int, random_seed=random_seed)
        band_err = [t1_CI, t2_CI]
        band_avg = [t1_avg, t2_avg]
        band_pdiff = p_diff

    return band_avg, band_err, band_pdiff
    
def notch_filter_eeg(eeg, fs, f0=50, Q=30):
    # Notch filter for removing 50Hz noise
    # eeg is the EEG signal (channels x time)
    # fs is the sampling frequency
    # f0 is the center frequency
    # Q is the quality factor
    # returns the filtered signal

    w0 = f0/(fs/2)
    b, a = iirnotch(w0, Q)
    if len(eeg.shape) == 1:
        eeg_filt = filtfilt(b, a, eeg)
    else:
        eeg_filt = filtfilt(b, a, eeg, axis=1)
    return eeg_filt

def zscore_spect(s, t, baseline_times):
    # z-score each trial's spectrogram
    # s is trials x freq x time
    # baseline is time indices for baseline period [start, end]
    bl_idx = [np.where(t <= baseline_times[0])[0][-1], np.where(t >= baseline_times[1])[0][0]]
    baseline = s[:, :, bl_idx[0]:bl_idx[1]]
    return (s - np.nanmean(baseline, axis=2)[:, :, np.newaxis])/np.nanstd(baseline, axis=2)[:, :, np.newaxis]

def get_fft_time(times, t):
    # time is real time, t is the vector returned from spectrogram function
    # time can be a single number or array
    # returns the index of the time in the fft
    if isinstance(times, (int, float)):
        time_idx = np.where(t <= times)[0][-1]
    else:
        time_idx = np.array([np.where(t <= this_t)[0][-1] for this_t in times])

    return time_idx


def calc_trial_spect(signal, fs, nperseg, noverlap, nfft, f_cutoff =0,
                     plot=1, fixation=0, stim_on=0, stim_off=0, probe=0, response=0,
                     signal_label = 'scalp EEG', spect_range = [],
                     t_skip=5, f_skip=2, plot_range = []):
    # Calculate the spect for a single trial (or mean of trial types if signal has multiple trials)
    # Signal is trials x time

    f, t, s = spectrogram(signal, fs=fs, window=('tukey', 0.25),
                          nperseg=nperseg, noverlap=noverlap, nfft=nfft, scaling='density') 
    
    if len(s.shape) > 2:
        s = np.nanmean(s, axis=0)
    
    if f_cutoff == 0:
            f_cutoff_idx = len(f)
    else:
        f_cutoff_idx = np.where(f <= f_cutoff)[0][-1]
    
    if plot:
        fft_times = {
            'stim_on_idx': np.where(t <= stim_on)[0][-1],
            'stim_off_idx': np.where(t <= stim_off)[0][-1],
            'probe_idx': np.where(t <= probe)[0][-1],
            'response_idx': np.where(t <= response)[0][-1]
        }
        fs_times = ld.get_event_indices(fs)

        fig, axs = plt.subplots(3,1,figsize=(16,14))

        # plot raw signal
        axs[0].plot(signal)
        axs[0].set_title(f'{signal_label} signal')
        for key in fs_times:
            axs[0].axvline(x=fs_times[key], color='r', linestyle='--')
        
        # plot spectrogram
        if len(spect_range) > 0:
            im = axs[1].imshow(10*np.log(s[:f_cutoff_idx,:]), 
                                aspect='auto', origin='lower', vmin=spect_range[0], vmax=spect_range[1])
        else:
            im = axs[1].imshow(10*np.log(s[:f_cutoff_idx,:]), 
                                aspect='auto', origin='lower')
        axs[1].set_xlabel('time (s)')
        axs[1].set_xticks(range(len(t))[::t_skip],t[::t_skip])
        axs[1].set_ylabel('frequency (Hz)')
        axs[1].set_yticks(range(len(f[:f_cutoff_idx]))[::f_skip],f[:f_cutoff_idx][::f_skip])
        cbar = fig.colorbar(im)
        cbar.ax.set_title('dB/Hz')
        axs[1].set_title(f'Time-Frequency plot, nperseg={nperseg}, noverlap={noverlap}, nfft={nfft}')
        for key in fft_times:
            axs[1].axvline(x=fft_times[key], color='r', linestyle='--', label=key)
        axs[1].legend()

        # plotting all band power
        band_power, band_sem = calc_band_power(s, f, band='all')
        for band in band_power:
            axs[2].plot(band_power[band], label=band)
            axs[2].fill_between(range(len(band_power[band])), band_power[band] - band_sem[band], band_power[band] + band_sem[band], alpha=0.3)
        axs[2].set_xlabel('time (s)')
        axs[2].set_xticks(range(len(t))[::t_skip],t[::t_skip])
        axs[2].set_title('band power')
        for key in fft_times:
            axs[2].axvline(x=fft_times[key], color='r', linestyle='--')
        axs[2].legend()

        plt.show()

    return f[:f_cutoff_idx], t, s[:f_cutoff_idx,:]    


def compare_avg_spect(signal, fs, nperseg, noverlap, nfft, trials1_ind, trials2_ind,
                      f_cutoff=0, fixation=0, stim_on=0, stim_off=0, probe=0,
                      plot=1, t_skip=5, f_skip=2, trials1_label='low load', trials2_label='high load',
                      EEG_range = [], spect_range = [], band_range = []):
    
    # plots average spectrogram depending on trials
    
    # signal is (trials x time)
    # trial_idxs is the indices of the neurons to average over

    f, t, s = spectrogram(signal, fs=fs, window=('tukey', 0.25),
                          nperseg=nperseg, noverlap=noverlap, nfft=nfft, scaling='density') 

    s_trials1 = np.nanmean(s[trials1_ind,:,:], axis=0)
    s_trials2 = np.nanmean(s[trials2_ind,:,:], axis=0)

    times = ld.get_real_event_times()
    t = t + times['fixation'] # align to probe time

    if f_cutoff == 0:
            f_cutoff_idx = len(f)
    else:
        f_cutoff_idx = np.where(f <= f_cutoff)[0][-1]

    if plot:
        type_label = [trials1_label,trials2_label]
        inds = [trials1_ind, trials2_ind]
        s_list = [s_trials1, s_trials2]

        fig, axs = plt.subplots(4,1,figsize=(16,20))
        for i, type in enumerate(type_label):
            
            fft_times = {
                'stim_on_idx': np.where(t <= stim_on)[0][-1],
                'stim_off_idx': np.where(t <= stim_off)[0][-1],
                'probe_idx': np.where(t <= probe)[0][-1],
                # 'response_idx': np.where(t <= response)[0][-1]
            }
            fs_times = ld.get_event_indices(fs)

            # plot raw signal
            axs[0].plot(np.mean(signal[inds[i],:], axis=0), label=f'{type}, n={len(inds[i])}')
            axs[0].fill_between(range(len(np.mean(signal[inds[i],:], axis=0))), 
                                np.mean(signal[inds[i],:], axis=0) - np.std(signal[inds[i],:], axis=0), 
                                np.mean(signal[inds[i],:], axis=0) + np.std(signal[inds[i],:], axis=0), alpha=0.3)
            axs[0].set_title(f'averaged EEG signal')
            for key in fs_times:
                axs[0].axvline(x=fs_times[key], color='r', linestyle='--')
            if len(EEG_range) > 0:
                axs[0].set_ylim(EEG_range)
            
            # plot spectrogram
            if len(spect_range) > 0:
                im = (axs[1+i].imshow(10*np.log(s_list[i][:f_cutoff_idx,:]),
                                     aspect='auto', origin='lower', vmin=spect_range[0], vmax=spect_range[1]))
            else:
                im = (axs[1+i].imshow(10*np.log(s_list[i][:f_cutoff_idx,:]), 
                                     aspect='auto', origin='lower'))
            axs[1+i].set_xlabel('time (s)')
            axs[1+i].set_xticks(range(len(t))[::t_skip],np.round(t[::t_skip],3))
            axs[1+i].set_ylabel('frequency (Hz)')
            axs[1+i].set_yticks(range(len(f[:f_cutoff_idx]))[::f_skip],f[:f_cutoff_idx][::f_skip])
            for key in fft_times:
                axs[1+i].axvline(x=fft_times[key], color='r', linestyle='--', label=key)
            axs[1+i].legend()
            cbar = fig.colorbar(im)
            cbar.ax.set_title('dB/Hz')
            axs[1+i].set_title(f'{type} time-frequency plot, nperseg={nperseg}, noverlap={noverlap}, nfft={nfft}')

            # plotting all band power
            c = ['b','g','r','c','m','y','k']
            linestyle = ['-', '--', '-.', ':', '-', '--', '-.']
            band_power, band_sem = calc_band_power(s_list[i], f, band='all')
            for j, band in enumerate(band_power):
                axs[3].plot(band_power[band], label=f'{band} {type_label[i]}', color = c[j], linestyle = linestyle[i])
                axs[3].fill_between(range(len(band_power[band])), band_power[band] - band_sem[band], band_power[band] + band_sem[band], alpha=0.3)
            axs[3].set_xlabel('time (s)')
            axs[3].set_xticks(range(len(t))[::t_skip],np.round(t[::t_skip]))
            axs[3].set_title('band power')
            for key in fft_times:
                axs[3].axvline(x=fft_times[key], color='r', linestyle='--')
            axs[3].legend()
            if len(band_range) > 0:
                axs[3].set_ylim(band_range)

        axs[0].legend()
        axs[3].legend()
        plt.show()

        # # plot band powers separately, with each band as a subplot
        # fig, axs = plt.subplots(len(band_power),1,figsize=(16,20))
        # for i, type in enumerate(type_label):
        #     band_power, band_sem = calc_band_power(s_list[i], f, band='all')
        #     for j, band in enumerate(band_power):
        #         axs[j].plot(band_power[band], label=f'{band} {type_label[i]}')
        #         axs[j].fill_between(range(len(band_power[band])), band_power[band] - band_sem[band], band_power[band] + band_sem[band], alpha=0.3)
        #         axs[j].set_xlabel('time (s)')
        #         axs[j].set_xticks(range(len(t))[::t_skip],t[::t_skip])
        #         axs[j].set_title(f'{band} band power')
        #         for key in fft_times:
        #             axs[j].axvline(x=fft_times[key], color='r', linestyle='--')
        #         axs[j].legend()
        #         if len(band_range) > 0:
        #             axs[j].set_ylim(band_range)
        # plt.show()


    return f[:f_cutoff_idx], t, s[:f_cutoff_idx,:] 


def calc_avg_spect(signal, fs, nperseg, noverlap, nfft, f_cutoff, exc_ind, inh_ind,
                   stim1_on, stim1_off, stim2_on, stim2_off,
                   plot=1, t_skip=5, f_skip=2, trial_label='same',
                   IPSC_range = [], spect_range = [], beta_range = []):
    
    # plots average spectrogram of excitatory and inhibitory neurons separately


    # signal is (neurons x time)
    # neuron_idxs is the indices of the neurons to average over

    f, t, s = spectrogram(signal, fs=fs, window=('tukey', 0.25), 
                          nperseg=nperseg, noverlap=noverlap, nfft=nfft, scaling='density')

    s_exc = np.nanmean(s[exc_ind,:,:], axis=0)
    s_inh = np.nanmean(s[inh_ind,:,:], axis=0)

    f_cutoff_idx = np.where(f <= f_cutoff)[0][-1]

    if plot:
        type_label = ['excitatory','inhibitory']
        inds = [exc_ind, inh_ind]
        s_list = [s_exc, s_inh]

        for i, type in enumerate(type_label):
        
            fig, axs = plt.subplots(3,1,figsize=(16,14))
            
            axs[0].plot(np.mean(signal[inds[i],:], axis=0))
            axs[0].fill_between(range(len(np.mean(signal[inds[i],:], axis=0))), 
                                np.mean(signal[inds[i],:], axis=0) - np.std(signal[inds[i],:], axis=0), 
                                np.mean(signal[inds[i],:], axis=0) + np.std(signal[inds[i],:], axis=0), alpha=0.3)
            axs[0].set_title(f'{trial_label} 25-trial-average IPSC, {type} neurons average (n={len(inds[i])})')
            axs[0].axvline(x=stim1_on*fs, color='r', linestyle='--')
            axs[0].axvline(x=stim1_off*fs, color='r', linestyle='--')
            axs[0].axvline(x=stim2_on*fs, color='r', linestyle='--')
            axs[0].axvline(x=stim2_off*fs, color='r', linestyle='--')
            if len(IPSC_range) > 0:
                axs[0].set_ylim(IPSC_range)

            stim1_on_idx = np.where(t <= stim1_on)[0][-1]
            stim1_off_idx = np.where(t <= stim1_off)[0][-1]
            stim2_on_idx = np.where(t <= stim2_on)[0][-1]
            stim2_off_idx = np.where(t <= stim2_off)[0][-1]
            
            if len(spect_range) > 0:
                im = axs[1].imshow(10*np.log(s_list[i][:f_cutoff_idx,:]), 
                                aspect='auto', origin='lower', vmin=spect_range[0], vmax=spect_range[1])
            else:
                im = axs[1].imshow(10*np.log(s_list[i][:f_cutoff_idx,:]), 
                                aspect='auto', origin='lower')
            axs[1].set_xlabel('time (s)')
            axs[1].set_xticks(range(len(t))[::t_skip],t[::t_skip])
            axs[1].set_ylabel('frequency (Hz)')
            axs[1].set_yticks(range(len(f[:f_cutoff_idx]))[::f_skip],f[:f_cutoff_idx][::f_skip])
            axs[1].axvline(x=stim1_on_idx, color='r', linestyle='--')
            axs[1].axvline(x=stim1_off_idx, color='r', linestyle='--')
            axs[1].axvline(x=stim2_on_idx, color='r', linestyle='--')
            axs[1].axvline(x=stim2_off_idx, color='r', linestyle='--')
            cbar = fig.colorbar(im)
            cbar.ax.set_title('dB/Hz')
            axs[1].set_title(f'Time-Frequency plot, nperseg={nperseg}, noverlap={noverlap}, nfft={nfft}')

            # plotting beta power (approx 13-30Hz)
            f_beta_idx = np.where((f >= 13) & (f <=30))[0]
            beta_power = 10*np.log(s_list[i][f_beta_idx,:])
            beta_mean = np.nanmean(beta_power, axis=0)
            beta_std = np.nanstd(beta_power, axis=0)
            beta_sem = np.sqrt(beta_std)/np.sqrt(beta_power.shape[0])

            axs[2].plot(beta_mean)
            axs[2].fill_between(range(len(beta_mean)), beta_mean - beta_sem, beta_mean + beta_sem, alpha=0.3)
            axs[2].axvline(x=stim1_on_idx, color='r', linestyle='--')
            axs[2].axvline(x=stim1_off_idx, color='r', linestyle='--')
            axs[2].axvline(x=stim2_on_idx, color='r', linestyle='--')
            axs[2].axvline(x=stim2_off_idx, color='r', linestyle='--')
            axs[2].set_xlabel('time (s)')
            axs[2].set_xticks(range(len(t))[::t_skip],t[::t_skip])
            axs[2].set_title('beta (13-30Hz) power')
            if len(beta_range) > 0:
                axs[2].set_ylim(beta_range)

            plt.show()

    return f, t, s, s_exc, s_inh



def plot_two_avg_spect(f, t, s_trials1_mean, s_trials2_mean, nperseg, noverlap, nfft,
                      fixation=0, stim_on=0, stim_off=0, probe=0, freq_range = [0, 150],
                      t_skip=5, f_skip=2, trials1_label='low load', trials2_label='high load',
                      spect_range = [], band_range = [], time_range = []):
    
    # plots average spectrogram depending on trials
    # also plots band powers with SEM over the bands (not over trials)
    
    # s is a signle mean spectrogram (freq x time)
    # trial_idxs is the indices of the neurons to average over

    type_label = [trials1_label,trials2_label]
    s_list = [s_trials1_mean, s_trials2_mean]

    fig, axs = plt.subplots(2,1,figsize=(16,12))
    for i, type in enumerate(type_label):
        
        fft_times = {
            'stim_on_idx': np.where(t <= stim_on)[0][-1],
            'stim_off_idx': np.where(t <= stim_off)[0][-1],
            'probe_idx': np.where(t <= probe)[0][-1],
            # 'response_idx': np.where(t <= response)[0][-1]
        }
        if len(time_range) == 2:
            time_start_idx = [np.where(t <= time_range[0])[0][-1] if ~np.isnan(time_range[0]) else 0][0]
            time_end_idx = [np.where(t <= time_range[1])[0][-1] if ~np.isnan(time_range[1]) else len(t)][0]
        elif len(time_range) == 0:
            time_start_idx = 0
            time_end_idx = len(t)

        freq_start_idx = np.where(f <= freq_range[0])[0][-1]
        freq_end_idx = np.where(f <= freq_range[1])[0][-1]

        # plot spectrogram
        if len(spect_range) > 0:
            im = (axs[i].imshow(10*np.log(s_list[i]),
                                    aspect='auto', origin='lower', vmin=spect_range[0], vmax=spect_range[1]))
        else:
            im = (axs[i].imshow(10*np.log(s_list[i]), 
                                    aspect='auto', origin='lower'))
        axs[i].set_xlabel('time (s)')
        axs[i].set_xticks(range(len(t))[::t_skip],np.round(t[::t_skip],3))
        axs[i].set_ylabel('frequency (Hz)')
        axs[i].set_yticks(range(len(f))[::f_skip],f[:][::f_skip])
        for key in fft_times:
            axs[i].axvline(x=fft_times[key], color='r', linestyle='--', label=key)
        axs[i].legend()
        cbar = fig.colorbar(im)
        cbar.ax.set_title('dB/Hz')
        axs[i].set_xlim([time_start_idx, time_end_idx])
        axs[i].set_title(f'{type} time-frequency plot, nperseg={nperseg}, noverlap={noverlap}, nfft={nfft}')
        axs[i].set_ylim(freq_start_idx, freq_end_idx)

    #     # plotting all band power
    #     c = ['b','g','r','c','m','y','k']
    #     linestyle = ['-', '--', '-.', ':', '-', '--', '-.']
    #     band_power, band_sem = calc_band_power(s_list[i], f, band='all')

    #     for j, band in enumerate(band_power):
    #         axs[2].plot(band_power[band], label=f'{band} {type_label[i]}', color = c[j], linestyle = linestyle[i])
    #         axs[2].fill_between(range(len(band_power[band])), band_power[band] - band_sem[band], band_power[band] + band_sem[band], 
    #                                 alpha=0.3, color = c[j])
    #     axs[2].set_xlabel('time (s)')
    #     axs[2].set_xticks(range(len(t))[::t_skip],np.round(t[::t_skip],3))
    #     axs[2].set_xlim([time_start_idx, time_end_idx])
    #     axs[2].set_title('band power')
    #     for key in fft_times:
    #         axs[2].axvline(x=fft_times[key], color='r', linestyle='--')
    #     axs[2].legend()
    #     if len(band_range) > 0:
    #         axs[2].set_ylim(band_range)

    # axs[2].legend()

    plt.show()


    return f, t


def compare_two_spect_groups_bootstrap(f, t, s_trials1, s_trials2, plot=1,
                      fixation=0, stim_on=0, stim_off=0, probe=0,
                      t_skip=5, f_skip=2, trials1_label='low load', trials2_label='high load',
                      nboot = 1000, CI_int = (2.5, 97.5), random_seed = 820,
                      p_sig = 0.05, title = [], plot_log=False, adjust_s2 = False, baseline_idx = [],
                      spect_range = [], band_range = [], time_range = [], c = []):
    
    # plots average spectrogram depending on trials
    
    # signal is (trials x time)
    # trial_idxs is the indices of the neurons to average over

    type_label = [trials1_label,trials2_label]

    # s_list_mean = [np.nanmean(s_trials1, axis=0), np.nanmean(s_trials2, axis=0)]
    if len(c) == 0:
        c = ['b','g','r','c','m','y','k']
        
    # fft_times = {
    #     'stim_on_idx': np.where(t <= stim_on)[0][-1],
    #     'stim_off_idx': np.where(t <= stim_off)[0][-1],
    #     'probe_idx': np.where(t <= probe)[0][-1],
    #     # 'response_idx': np.where(t <= response)[0][-1]
    # }

    times = ld.get_real_event_times()
    

    if len(time_range) == 2:
        time_start = [time_range[0] if ~np.isnan(time_range[0]) else times['fixation']][0]
        time_end = [time_range[1] if ~np.isnan(time_range[1]) else t[-1]][0]
        # time_start_idx = [np.where(t <= time_range[0])[0][-1] if ~np.isnan(time_range[0]) else 0][0]
        # time_end_idx = [np.where(t <= time_range[1])[0][-1] if ~np.isnan(time_range[1]) else len(t)][0]
    elif len(time_range) == 0:
        time_start = times['fixation']
        time_end = t[-1]
          
    # Calculate confidence intervals
    band_avg, band_err, band_pdiff = calc_band_power_bootstrap(s_trials1, s_trials2, f, 
                                                         band='all', nboot=nboot, CI_int=CI_int, random_seed=random_seed,
                                                         adjust_s2 = adjust_s2, baseline_idx = baseline_idx)
    
    # plot band powers separately, with each band as a subplot
    if plot:
        fig, axs = plt.subplots(len(band_err),1,figsize=(10,18), sharey=True)
        # x_values = np.linspace(times['fixation'],times['end'],s_trials1.shape[1])
        for i, type in enumerate(type_label):
            for j, band in enumerate(band_avg):
                band_bootstrap_avg = np.nanmean(band_avg[band][i], axis=1)
                if plot_log:
                    axs[j].plot(t, 10*np.log(band_bootstrap_avg), color = c[i], label=f'{band} {type}')
                    axs[j].fill_between(range(len(band_bootstrap_avg)), 
                                        10*np.log(band_err[band][i][:,0]), 
                                        10*np.log(band_err[band][i][:,1]),
                                        alpha=0.3, color = c[i])
                else:
                    axs[j].plot(t, band_bootstrap_avg, color = c[i], label=f'{band} {type}')
                    axs[j].fill_between(t, band_err[band][i][:,0], band_err[band][i][:,1],
                                        alpha=0.3, color = c[i])
                if j == len(band_avg)-1:
                    axs[j].set_xlabel('time (s)')
                # tick_positions = range(len(t))[::t_skip]
                # tick_labels = np.round(t[::t_skip],3)
                # axs[j].set_xticks(tick_positions)
                # axs[j].set_xticklabels(tick_labels)
                axs[j].set_xticks(np.arange(times['fixation'],times['end'],1))
                axs[j].set_title(f'{band} band power')
                axs[j].axvline(x=stim_on, color='r', linestyle='--')
                axs[j].axvline(x=stim_off, color='r', linestyle='--')
                axs[j].axvline(x=probe, color='r', linestyle='--')
                # for key in times:
                #     axs[j].axvline(x=times[key], color='r', linestyle='--')
                # axs[j].set_xlim([min(tick_positions) + 25, max(tick_positions) - 25])
                # axs[j].set_xlim([time_start_idx, time_end_idx])
                axs[j].set_xlim([time_start, time_end])
                axs[j].legend()
                if len(band_range) > 0:
                    if isinstance(band_range[0], int):
                        axs[j].set_ylim(band_range)
                    else:
                        axs[j].set_ylim(band_range[j])

                if i == 0:
                    # time_vector = np.array(range(0, len(band_bootstrap_avg)))
                    # significant_timepoints = time_vector[band_pdiff[band] < p_sig]
                    significant_timepoints = t[band_pdiff[band] < p_sig]
                    if j == 0:
                        ymin, ymax = axs[j].get_ylim()
                    axs[j].scatter(significant_timepoints,
                                    np.zeros_like(significant_timepoints) + ymax + (ymax-ymin)/10, color='k', label='_nolegend_', marker='s', s=20)
                    
        if len(title) > 0:
            fig.suptitle(title)
        plt.show()
        # plt.savefig('/home/renee/WM_letters/results/Northwell/EEG_load.svg', format='svg')

    return band_avg, band_err, band_pdiff

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