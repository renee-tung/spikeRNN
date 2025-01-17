# Functions for calculating spectrogram

import numpy as np
from scipy.signal import spectrogram, resample, iirnotch, filtfilt
import matplotlib.pyplot as plt
import scipy.io
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
            if len(s.shape) == 2:
                band_power[band] = np.nanmean(s[f_band_idx,:], axis=0)
                band_sem[band] = np.nanstd(s[f_band_idx,:], axis=0)/np.sqrt(s[f_band_idx,:].shape[0])
            elif len(s.shape) == 3:
                band_power[band] = np.nanmean(s[:,f_band_idx,:], axis=1)
                band_sem[band] = np.nanstd(s[:,f_band_idx,:], axis=1)/np.sqrt(s[:,f_band_idx,:].shape[1])
    elif isinstance(band, str):
        f_band_idx = np.where((f >= band_ranges[band][0]) & (f <=band_ranges[band][1]))[0]
        if len(s.shape) == 2:
            band_power = np.nanmean(s[f_band_idx,:], axis=0)
            band_sem = np.nanstd(s[f_band_idx,:], axis=0)/np.sqrt(s[f_band_idx,:].shape[0])
        elif len(s.shape) == 3:
            band_power = np.nanmean(s[:,f_band_idx,:], axis=1)
            band_sem = np.nanstd(s[:,f_band_idx,:], axis=1)/np.sqrt(s[:,f_band_idx,:].shape[1])

    return band_power, band_sem

def calc_band_power_bootstrap(s1, s2, f, band='all', nboot=1000, CI_int=(2.5, 97.5), random_seed=820):
    # Calculates the power in each frequency band using bootstrapping
    # s1 and s2 are the spectrograms (n x freq x time)
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
            
            t1_avg, t1_CI, t2_avg, t2_CI, _, _, p_diff = fnc_time_bootstrap_optimized_retX(np.squeeze(np.nanmean(s1[:,f_band_idx,:], axis=1)), 
                                                                  np.squeeze(np.nanmean(s2[:,f_band_idx,:], axis=1)), 
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


def compare_avg_spect(signal, fs, nperseg, noverlap, nfft, neur1_ind, neur2_ind,
                      f_cutoff=0, fixation=0, stim_on=0, stim_off=0, probe=0,
                      plot=1, t_skip=5, f_skip=2, trials1_label='exc neurons', trials2_label='inh neurons',
                      EEG_range = [], spect_range = [], band_range = []):
    
    # plots average spectrogram depending on neurons
    
    # signal is (neurons x time)
    # neur1_ind, neur2_ind is the indices of the neurons to average over
    # times are in real time (seconds)

    f, t, s = spectrogram(signal, fs=fs, window=('tukey', 0.25),
                          nperseg=nperseg, noverlap=noverlap, nfft=nfft, scaling='density') 

    s_trials1 = np.nanmean(s[neur1_ind,:,:], axis=0)
    s_trials2 = np.nanmean(s[neur2_ind,:,:], axis=0)

    if f_cutoff == 0:
            f_cutoff_idx = len(f)
    else:
        f_cutoff_idx = np.where(f <= f_cutoff)[0][-1]

    if plot:
        type_label = [trials1_label,trials2_label]
        inds = [neur1_ind, neur2_ind]
        s_list = [s_trials1, s_trials2]

        fig, axs = plt.subplots(4,1,figsize=(16,20))
        for i, type in enumerate(type_label):
            
            fft_times = {
                'stim_on_idx': np.where(t <= stim_on)[0][-1],
                'stim_off_idx': np.where(t <= stim_off)[0][-1],
                'probe_idx': np.where(t <= probe)[0][-1],
                # 'response_idx': np.where(t <= response)[0][-1]
            }

            # plot raw signal
            axs[0].plot(np.mean(signal[inds[i],:], axis=0), label=f'{type}, n={len(inds[i])}')
            axs[0].fill_between(range(len(np.mean(signal[inds[i],:], axis=0))), 
                                np.mean(signal[inds[i],:], axis=0) - np.std(signal[inds[i],:], axis=0), 
                                np.mean(signal[inds[i],:], axis=0) + np.std(signal[inds[i],:], axis=0), alpha=0.3)
            axs[0].set_title(f'averaged EEG signal')
            axs[0].axvline(x=stim_on*fs, color='r', linestyle='--')
            axs[0].axvline(x=stim_off*fs, color='r', linestyle='--')
            axs[0].axvline(x=probe*fs, color='r', linestyle='--')
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
            axs[1+i].set_xticks(range(len(t))[::t_skip],t[::t_skip])
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
            axs[3].set_xticks(range(len(t))[::t_skip],t[::t_skip])
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


    return f, t, s


def calc_avg_spect(signal, fs, nperseg, noverlap, nfft, f_cutoff, exc_ind, inh_ind,
                   stim1_on=0, stim1_off=0, stim2_on=0, stim2_off=0,
                   plot=0, t_skip=5, f_skip=2, trial_label='same',
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

def calc_spect(signal, fs, nperseg, noverlap, nfft):
    # Calculates the spectrogram

    f, t, s = spectrogram(signal, fs=fs, window=('tukey', 0.25), 
                          nperseg=nperseg, noverlap=noverlap, nfft=nfft, scaling='density')

    return f, t, s

def plot_two_avg_spect(f, t, s_trials1_mean, s_trials2_mean, nperseg, noverlap, nfft,
                      fixation=0, stim_on=0, stim_off=0, probe=0,
                      t_skip=5, f_skip=2, neur1_label='exc neurons', neur2_label='inh neurons',
                      spect_range = [], band_range = [], time_range = []):
    
    # plots average spectrogram depending on trials
    # also plots band powers with SEM over the bands (not over trials)
    
    # signal is (trials x time)
    # trial_idxs is the indices of the neurons to average over

    type_label = [neur1_label,neur2_label]
    s_list = [s_trials1_mean, s_trials2_mean]

    fig, axs = plt.subplots(3,1,figsize=(16,20))
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

        # plot spectrogram
        if len(spect_range) > 0:
            im = (axs[i].imshow(10*np.log(s_list[i]),
                                    aspect='auto', origin='lower', vmin=spect_range[0], vmax=spect_range[1]))
        else:
            im = (axs[i].imshow(10*np.log(s_list[i]), 
                                    aspect='auto', origin='lower'))
        axs[i].set_xlabel('time (s)')
        axs[i].set_xticks(range(len(t))[::t_skip],t[::t_skip])
        axs[i].set_ylabel('frequency (Hz)')
        axs[i].set_yticks(range(len(f))[::f_skip],f[:][::f_skip])
        for key in fft_times:
            axs[i].axvline(x=fft_times[key], color='r', linestyle='--', label=key)
        axs[i].legend()
        cbar = fig.colorbar(im)
        cbar.ax.set_title('dB/Hz')
        axs[i].set_xlim([time_start_idx, time_end_idx])
        axs[i].set_title(f'{type} time-frequency plot, nperseg={nperseg}, noverlap={noverlap}, nfft={nfft}')

        # plotting all band power
        c = ['b','g','r','c','m','y','k']
        linestyle = ['-', '--', '-.', ':', '-', '--', '-.']
        band_power, band_sem = calc_band_power(s_list[i], f, band='all')

        for j, band in enumerate(band_power):
            axs[2].plot(band_power[band], label=f'{band} {type_label[i]}', color = c[j], linestyle = linestyle[i])
            axs[2].fill_between(range(len(band_power[band])), band_power[band] - band_sem[band], band_power[band] + band_sem[band], 
                                    alpha=0.3, color = c[j])
        axs[2].set_xlabel('time (s)')
        axs[2].set_xticks(range(len(t))[::t_skip],t[::t_skip])
        axs[2].set_xlim([time_start_idx, time_end_idx])
        axs[2].set_title('band power')
        for key in fft_times:
            axs[2].axvline(x=fft_times[key], color='r', linestyle='--')
        axs[2].legend()
        if len(band_range) > 0:
            axs[2].set_ylim(band_range)

    axs[2].legend()

    plt.show()


    return f, t

def plot_band_power_bootstrap(band_trials1, band_trials2, t, nboot=1000, CI_int=(2.5, 97.5), random_seed=820,
                              type1_label='exc neurons', type2_label='inh neurons', p_sig=0.05, title = [],
                              stim_on=0, stim_off=0, probe=0, t_skip=5, 
                              band_range = [], time_range = []):
    
    # band_trials is a dict with keys as bands and values as the band power (trials x time)
    # f is the frequency vector
    # t is the time vector

    type_label = [type1_label,type2_label]
    c = ['b','g','r','c','m','y','k']

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

    # plot band powers separately, with each band as a subplot
    fig, axs = plt.subplots(len(band_trials1),1,figsize=(10,18))
    for j, band in enumerate(band_trials1.keys()):
        t1_avg, t1_CI, t2_avg, t2_CI, _, _, p_diff = fnc_time_bootstrap_optimized_retX(band_trials1[band], band_trials2[band],
                                                                nboot=nboot, CI_int=CI_int, random_seed=random_seed)
        t1_avg = np.nanmean(t1_avg, axis=1)
        t2_avg = np.nanmean(t2_avg, axis=1)
        axs[j].plot(t1_avg, color = c[0], label=f'{band} {type_label[0]}')
        axs[j].fill_between(range(len(t1_avg)), t1_CI[:,0], t1_CI[:,1], alpha=0.3, color = c[0])
        axs[j].plot(t2_avg, color = c[1], label=f'{band} {type_label[1]}')
        axs[j].fill_between(range(len(t2_avg)), t2_CI[:,0], t2_CI[:,1], alpha=0.3, color = c[1])
        axs[j].set_xticks(range(len(t))[::t_skip],t[::t_skip])
        axs[j].set_title(f'{band} band power')
        for key in fft_times:
            axs[j].axvline(x=fft_times[key], color='r', linestyle='--')
        axs[j].set_xlim([time_start_idx, time_end_idx])
        axs[j].legend()
        if len(band_range) > 0:
            axs[j].set_ylim(band_range)

        time_vector = np.array(range(0, len(t1_avg)))
        significant_timepoints = time_vector[p_diff < p_sig]
        visible_y = np.append(t1_CI[:,1][time_start_idx:time_end_idx], t2_CI[:,1][time_start_idx:time_end_idx]).flatten()
        if len(visible_y):
            axs[j].set_ylim(np.amin(visible_y), np.max(visible_y))
        ymin, ymax = axs[j].get_ylim()
        axs[j].scatter(significant_timepoints,
                        np.zeros_like(significant_timepoints) + ymax + (ymax-ymin)/10, color='k', label='_nolegend_', marker='s', s=25)
        axs[j].set_ylim(0, ymax + (ymax-ymin)/5)

        if j == len(band_trials1)-1:
            axs[j].set_xlabel('time (s)')

    if len(title) > 0:
        fig.suptitle(title)
    plt.show()
    
    return


def compare_two_spect_groups_bootstrap(f, t, s_trials1, s_trials2, plot=1,
                      fixation=0, stim_on=0, stim_off=0, probe=0,
                      t_skip=5, f_skip=2, neur1_label='exc neurons', neur2_label='inh neurons',
                      nboot = 1000, CI_int = (2.5, 97.5), random_seed = 820,
                      p_sig = 0.05, title = [],
                      spect_range = [], band_range = [], time_range = []):
    
    # plots average spectrogram depending on trials
    
    # signal s is (n x neurons x time)

    type_label = [neur1_label,neur2_label]

    # s_list_mean = [np.nanmean(s_trials1, axis=0), np.nanmean(s_trials2, axis=0)]

    c = ['b','g','r','c','m','y','k']
        
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
          
    # Calculate confidence intervals
    band_avg, band_err, band_pdiff = calc_band_power_bootstrap(s_trials1, s_trials2, f, 
                                                         band='all', nboot=nboot, CI_int=CI_int, random_seed=random_seed)
    
    # plot band powers separately, with each band as a subplot
    if plot:
        fig, axs = plt.subplots(len(band_err),1,figsize=(10,18))
        for i, type in enumerate(type_label):
            for j, band in enumerate(band_avg):
                band_bootstrap_avg = np.nanmean(band_avg[band][i], axis=1)
                axs[j].plot(band_bootstrap_avg, color = c[i], label=f'{band} {type}')
                axs[j].fill_between(range(len(band_bootstrap_avg)), band_err[band][i][:,0], band_err[band][i][:,1],
                                    alpha=0.3, color = c[i])
                
                tick_positions = range(len(t))[::t_skip]
                tick_labels = t[::t_skip]
                axs[j].set_xticks(tick_positions)
                axs[j].set_xticklabels(tick_labels)
                axs[j].set_title(f'{band} band power')
                for key in fft_times:
                    axs[j].axvline(x=fft_times[key], color='r', linestyle='--')
                # axs[j].set_xlim([min(tick_positions) + 25, max(tick_positions) - 25])
                axs[j].set_xlim([time_start_idx, time_end_idx])
                axs[j].legend()
                if len(band_range) > 0:
                    axs[j].set_ylim(band_range)

                if i == 0:
                    time_vector = np.array(range(0, len(band_bootstrap_avg)))
                    significant_timepoints = time_vector[band_pdiff[band] < p_sig]
                    # visible_y = band_err[band][i+1][:,1][time_start_idx:time_end_idx]
                    visible_y = np.append(np.nanmean(band_avg[band][i], axis=1)[time_start_idx:time_end_idx],
                                               np.nanmean(band_avg[band][i+1], axis=1)[time_start_idx:time_end_idx]).flatten()
                    if len(visible_y):
                        axs[j].set_ylim(np.amin(visible_y), np.max(visible_y))
                    ymin, ymax = axs[j].get_ylim()
                    axs[j].scatter(significant_timepoints,
                                    np.zeros_like(significant_timepoints) + ymax + (ymax-ymin)/10, color='k', label='_nolegend_', marker='s', s=10)
                    axs[j].set_ylim(0, ymax + (ymax-ymin)/5)
            axs[j].set_xlabel('time (s)')   
                 
        if len(title) > 0:
            fig.suptitle(title)
        plt.show()

    return band_avg, band_err, band_pdiff