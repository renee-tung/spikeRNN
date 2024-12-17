### Utils to help with plotting visualizations

import numpy as np
import matplotlib.pyplot as plt


from scipy.signal import spectrogram, resample
import tensorpac
from sklearn.decomposition import PCA

import pdb


'''
BEHAVIOR
'''

def generate_letters_type(T, stim_on, stim_dur, delay, load, label = 1):
    
    n_input_stim = int(2*load)
    u = np.zeros((n_input_stim, T))
     
    letters = np.arange(0, n_input_stim, 1) # 2*load letter choices
    stim_letters = np.random.choice(letters, load, replace=False) # [load] letter choices
    not_stim_letters = np.setdiff1d(letters, stim_letters) # half non-stim letter choices
    
    if label == 1:
        probe_letter = np.random.choice(stim_letters, 1)[0]
    else:
        probe_letter = np.random.choice(not_stim_letters, 1)[0]

    u[stim_letters, stim_on:stim_on+stim_dur] = 1 # stimulus presentation
    u[probe_letter, stim_on+stim_dur+delay:] = 1 # probe presentation

    return u, label


'''
NEURON FIRING RATES
'''

def plot_neuron_rates(r, exc_ind, inh_ind, stim_on, stim_dur, delay, vmin=0, vmax=1, sort=1, suptitle='Neuron rates'):
    # r is (time x neurons)
    # exc_ind and inh_ind are indices of excitatory and inhibitory neurons

    stim1_on = stim_on
    stim1_off = stim_on + stim_dur
    stim2_on = stim_on + stim_dur + delay
    # stim2_off = stim_on + 2*stim_dur + delay

    if sort:
        pca_exc = PCA(n_components=1).fit(r[:,exc_ind].T) # neurons x time (samples x fts)
        Xnew_exc = pca_exc.transform(r[:,exc_ind].T) # neurons x comp (samples x pcs)
        sort_idx = np.argsort(Xnew_exc[:,0])
        exc_ind = exc_ind[sort_idx]

        pca_inh = PCA(n_components=1).fit(r[:,inh_ind].T)
        Xnew_inh = pca_inh.transform(r[:,inh_ind].T)
        sort_idx = np.argsort(Xnew_inh[:,0])
        inh_ind = inh_ind[sort_idx]

    fig,axs = plt.subplots(1,2,figsize=(16,4))
    im1 = axs[0].imshow(r[:,exc_ind].T,cmap='binary',vmin=vmin, vmax=vmax, aspect='auto')
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('neurons')
    axs[0].set_title('excitatory neurons')
    axs[0].axvline(x=stim1_on, color='r', linestyle='--')
    axs[0].axvline(x=stim1_off, color='r', linestyle='--')
    axs[0].axvline(x=stim2_on, color='r', linestyle='--')
    # axs[0].axvline(x=stim2_off, color='r', linestyle='--')
    fig.colorbar(im1)

    im2 = axs[1].imshow(r[:,inh_ind].T,cmap='binary',vmin=vmin, vmax=vmax, aspect='auto')
    axs[1].set_xlabel('time')
    axs[1].set_ylabel('neurons')
    axs[1].set_title('inhibitory neurons')
    axs[1].axvline(x=stim1_on, color='r', linestyle='--')
    axs[1].axvline(x=stim1_off, color='r', linestyle='--')
    axs[1].axvline(x=stim2_on, color='r', linestyle='--')
    # axs[1].axvline(x=stim2_off, color='r', linestyle='--')
    fig.colorbar(im2)

    plt.suptitle(suptitle)
    plt.show()

    # print(f'{len(exc_ind)} excitatory neurons')
    # print(f'{len(inh_ind)} inhibitory neurons')

    return


def plot_neuron_raster(spk, exc_ind, inh_ind, stim_on, stim_dur, delay, sort=1, suptitle='Neuron raster'):
    # spk is (neurons x time)
    # exc_ind and inh_ind are indices of excitatory and inhibitory neurons

    stim1_on = stim_on
    stim1_off = stim_on + stim_dur
    stim2_on = stim_on + stim_dur + delay
    stim2_off = stim_on + 2*stim_dur + delay

    if sort:
        pca_exc = PCA(n_components=1).fit(spk[exc_ind, :]) # neurons x time (samples x fts)
        Xnew_exc = pca_exc.transform(spk[exc_ind, :]) # neurons x comp (samples x pcs)
        sort_idx = np.argsort(Xnew_exc[:,0])
        exc_ind = exc_ind[sort_idx]

        pca_inh = PCA(n_components=1).fit(spk[inh_ind, :])
        Xnew_inh = pca_inh.transform(spk[inh_ind, :])
        sort_idx = np.argsort(Xnew_inh[:,0])
        inh_ind = inh_ind[sort_idx]

    fig,axs = plt.subplots(1,2,figsize=(16,4))
    for i, neuron_spikes in enumerate(spk[exc_ind, :]):
        spike_times = np.where(neuron_spikes == 1)[0]  # Get the indices where the spike is 1
        axs[0].plot(spike_times, np.ones_like(spike_times) * (i + 1), 'k|', markersize=5)  # 'k|' plots vertical lines
    axs[0].set_xlabel('time (s)')
    axs[0].set_ylabel('neurons')
    axs[0].set_title('excitatory neurons')
    axs[0].axvline(x=stim1_on, color='r', linestyle='--')
    axs[0].axvline(x=stim1_off, color='r', linestyle='--')
    axs[0].axvline(x=stim2_on, color='r', linestyle='--')
    axs[0].axvline(x=stim2_off, color='r', linestyle='--')
    axs[0].set_xticks(range(0,spk.shape[1],10000),np.arange(0,spk.shape[1],10000)/20000)

    for i, neuron_spikes in enumerate(spk[inh_ind, :]):
        spike_times = np.where(neuron_spikes == 1)[0]  # Get the indices where the spike is 1
        axs[1].plot(spike_times, np.ones_like(spike_times) * (i + 1), 'k|', markersize=5)  # 'k|' plots vertical lines
    axs[1].set_xlabel('time (s)')
    axs[1].set_ylabel('neurons')
    axs[1].set_title('inhibitory neurons')
    axs[1].axvline(x=stim1_on, color='r', linestyle='--')
    axs[1].axvline(x=stim1_off, color='r', linestyle='--')
    axs[1].axvline(x=stim2_on, color='r', linestyle='--')
    axs[1].axvline(x=stim2_off, color='r', linestyle='--')
    axs[1].set_xticks(range(0,spk.shape[1],10000),np.arange(0,spk.shape[1],10000)/20000)

    plt.suptitle(suptitle)
    plt.show()

    # print(f'{len(exc_ind)} excitatory neurons')
    # print(f'{len(inh_ind)} inhibitory neurons')

    return

'''
PACs
'''

def calculate_erpac(ipscs, fs, neuron_inds, plot=1, settings = 0, neuron_label='exc', trial_label = '+1/+1 (same)',
                    f_pha=[4, 8], f_amp=(30, 200, 2, 1), dcomplex='hilbert', cycle=(3,6), width=7, method='circular'):
    times = np.arange(ipscs.shape[1])
    if isinstance(neuron_inds, int):
        data = ipscs[neuron_inds,:,:].squeeze().T
    else:
        data = np.nanmean(ipscs[neuron_inds,:,:], axis=0).squeeze().T # neuron mean, trials x times, transpose to get trials x times
    print(data.shape)
    # n_trials = data.shape[0]
    # pdb.set_trace()

    rp_obj = tensorpac.EventRelatedPac(f_pha=f_pha, f_amp=f_amp, dcomplex=dcomplex, 
                        cycle=cycle, width=width)
    erpac = rp_obj.filterfit(fs, data, method=method)

    if plot:
        plt.figure(figsize=(8, 6))
        rp_obj.pacplot(erpac.squeeze(), times/fs, rp_obj.yvec, xlabel='Time',
                    ylabel='Amplitude frequency (Hz)',
                    title=f'ERPAC for theta ({f_pha[0]}-{f_pha[1]}Hz) phase, {neuron_label} neurons, {trial_label}',
                    fz_labels=15, fz_title=18, vmin=0, vmax=1)
        plt.axvline(settings['stim_on']/fs*100, color='r', linestyle='--')
        plt.axvline((settings['stim_on']+settings['stim_dur'])/fs*100, color='r', linestyle='--')
        plt.axvline((settings['stim_on']+settings['stim_dur']+settings['delay'])/fs*100, color='r', linestyle='--')
        plt.axvline((settings['stim_on']+2*settings['stim_dur']+settings['delay'])/fs*100, color='r', linestyle='--')
        plt.show()

    return rp_obj, erpac.squeeze()


def plot_erpac(rp_obj, erpac, times, freqs, fs, settings, settings_real = 0, title = 'ERPAC', vmin=0, vmax=1, cmap='viridis',
               save=0, savename=None):
    # code to plot
    plt.figure(figsize=(8, 6))
    rp_obj.pacplot(erpac, times/fs, freqs, xlabel='Time',
                ylabel='Amplitude frequency (Hz)',
                title=title, cmap=cmap,
                fz_labels=15, fz_title=18, vmin=vmin, vmax=vmax)
    if settings_real == 1:
        plt.axvline(settings['stim1_on'], color='r', linestyle='--')
        plt.axvline(settings['stim1_off'], color='r', linestyle='--')
        plt.axvline(settings['stim2_on'], color='r', linestyle='--')
        plt.axvline(settings['stim2_off'], color='r', linestyle='--')
    else:
        plt.axvline(settings['stim_on']/fs*100, color='r', linestyle='--')
        plt.axvline((settings['stim_on']+settings['stim_dur'])/fs*100, color='r', linestyle='--')
        plt.axvline((settings['stim_on']+settings['stim_dur']+settings['delay'])/fs*100, color='r', linestyle='--')
        plt.axvline((settings['stim_on']+2*settings['stim_dur']+settings['delay'])/fs*100, color='r', linestyle='--')
    if save:
        plt.savefig(savename)
        plt.close()
    else:
        plt.show()

    return


def calculate_pac(ipscs, fs, time_onset, time_offset, neuron_inds, plot=1, 
                    idpac=(2,1,0), f_pha=(2, 14, 2, 1), f_amp=(30, 200, 2, 1), dcomplex='hilbert', cycle=(3, 6), width=7, n_bins=18,
                    neuron_label='exc', trial_label = '+1/+1 (same)', time_label = 'delay', vmax=0.004):
    # times = np.arange(ipscs.shape[1])
    data = np.mean(ipscs[neuron_inds,time_onset:time_offset,:], axis=0).T # neuron mean, trials x times
    data = data.reshape((-1)) # flatten data for calculating PAC
    # n_trials = data.shape[0]

    pac_obj = tensorpac.Pac(idpac=idpac, f_pha=f_pha, f_amp=f_amp, dcomplex=dcomplex, 
                        cycle=cycle, width=width, n_bins=n_bins)
    pac = pac_obj.filterfit(fs, data)

    if plot:
        plt.figure(figsize=(8, 6))
        pac_obj.comodulogram(pac, xlabel='Phase', ylabel='Amplitude frequency (Hz)',
                    title=f'PACs, {neuron_label} neurons, {trial_label}, during {time_label}',
                    fz_labels=15, fz_title=18, vmin=0, vmax=vmax)
        plt.show()

    return pac_obj, pac


def plot_pac(pac_obj, pac, title = 'PAC', vmin=0, vmax=1, cmap='viridis', 
             save=0, savename=None):
    # code to plot
    plt.figure(figsize=(8, 6))
    pac_obj.comodulogram(pac, xlabel='Phase', ylabel='Amplitude frequency (Hz)',
                title=title, cmap=cmap,
                fz_labels=15, fz_title=18, vmin=vmin, vmax=vmax)
    if save:
        plt.savefig(savename)
    else:
        plt.show()

    return

'''
SPECTROGRAMS
'''

def plot_all_power(signals, fs, nperseg, noverlap, nfft, exc_ind, inh_ind,
                   stim1_on, stim1_off, stim2_on, stim2_off, powers_to_plot,
                   trial_type_labels = ["+1/+1 same", "+1/-1 diff", "-1/+1 diff", "-1/-1 same"],
                   t_skip=5, plot_range = []):
    # signal is (neurons x time x n_trial_types)
    # neuron_idxs is the indices of the neurons to average over
    # power_range is a list, can be ["all"],"theta","alpha","beta","gamma1","gamma2", or self-input list of power ranges
    
    n_trial_types = signals.shape[2]    
    
    neuron_type_label = ["excitatory","inhibitory"]
    # trial_type_labels = ["+1/+1 same", "+1/-1 diff", "-1/+1 diff", "-1/-1 same"]
    colors = ['r','b','c','m']
    alphas = [1,0.7]
    linestyle = ['-', '--']

    # plotting power range of interest
    if isinstance(powers_to_plot[0], str):
        if powers_to_plot[0] == "all":
            powers_to_plot = ["theta","alpha","beta","gamma1","gamma2"]

    for power in powers_to_plot:
        print(power)
        if isinstance(power, str):
            if power == "theta":
                power_range = [4,8]
            elif power == "alpha":
                power_range = [9,12]
            elif power == "beta":
                power_range = [13,30]
            elif power == "gamma1":
                power_range = [30,55]
            elif power == "gamma2":
                power_range = [70,140]
        else:
            power_range = power

        plt.figure(figsize=(16,5))
        for n_trial_type in range(n_trial_types):
            f, t, s = spectrogram(signals[:,:,n_trial_type], fs=fs, window=('tukey', 0.25),
                                nperseg=nperseg, noverlap=noverlap, nfft=nfft, scaling='density')        
            
            s_exc = np.nanmean(s[exc_ind,:,:], axis=0)
            s_inh = np.nanmean(s[inh_ind,:,:], axis=0)
            
            s_list = [s_exc, s_inh]        
            
            for i, neuron_type in enumerate(neuron_type_label):            
                
                stim1_on_idx = np.where(t <= stim1_on)[0][-1]
                stim1_off_idx = np.where(t <= stim1_off)[0][-1]
                stim2_on_idx = np.where(t <= stim2_on)[0][-1]
                stim2_off_idx = np.where(t <= stim2_off)[0][-1]            

                # calculate power in specified band
                f_band_idx = np.where((f >= power_range[0]) & (f <=power_range[1]))[0]
                band_power = 10*np.log(s_list[i][f_band_idx,:])
                band_mean = np.nanmean(band_power, axis=0)
                band_std = np.nanstd(band_power, axis=0)
                band_sem = np.sqrt(band_std)/np.sqrt(band_power.shape[0])
                
                plt.plot(band_mean, label=f'{trial_type_labels[n_trial_type]}, {neuron_type}', c=colors[n_trial_type], alpha=alphas[i], linestyle=linestyle[i])
                plt.fill_between(range(len(band_mean)), band_mean - band_sem, band_mean + band_sem, color=colors[n_trial_type], alpha=0.3)
                plt.axvline(x=stim1_on_idx, color='r', linestyle='--')
                plt.axvline(x=stim1_off_idx, color='r', linestyle='--')
                plt.axvline(x=stim2_on_idx, color='r', linestyle='--')
                plt.axvline(x=stim2_off_idx, color='r', linestyle='--')
                plt.xlabel('time (s)')
                plt.xticks(range(len(t))[::t_skip],t[::t_skip])
                plt.title(f'{power} ({power_range[0]}-{power_range[1]}Hz) power')
                if len(plot_range) > 0:
                    plt.ylim(plot_range)    
            plt.legend()
        plt.show()    
    
    return


def calc_indv_spect(signal, fs, nperseg, noverlap, nfft, f_cutoff, inh=0,
               plot=1, neuron_nums=[0], t_skip=5, f_skip=2, trial_label='same',
               IPSC_range = [], spect_range = [], beta_range = []):
    
    # plots spectrogram of individual neurons
    
    # signal is (neurons x time)
    # neuron_num is list of neuron numbers to plot

    f, t, s = spectrogram(signal, fs=fs, window=('tukey', 0.25), 
                          nperseg=nperseg, noverlap=noverlap, nfft=nfft, scaling='density')

    f_cutoff_idx = np.where(f <= f_cutoff)[0][-1]

    if plot:
        type_label = ['excitatory','inhibitory']
        for neuron_num in neuron_nums:
            fig, axs = plt.subplots(3,1,figsize=(16,14))
            
            axs[0].plot(signal[neuron_num,:])
            axs[0].set_title(f'{trial_label} trial IPSC, neuron {neuron_num}, {type_label[inh[neuron_num][0]]}')
            if len(IPSC_range) > 0:
                axs[0].set_ylim(IPSC_range)
            
            if len(spect_range) > 0:
                im = axs[1].imshow(10*np.log(s[neuron_num,:f_cutoff_idx,:]), 
                                   aspect='auto', origin='lower', vmin=spect_range[0], vmax=spect_range[1])
            else:
                im = axs[1].imshow(10*np.log(s[neuron_num,:f_cutoff_idx,:]), 
                                   aspect='auto', origin='lower')
            axs[1].set_xlabel('time (s)')
            axs[1].set_xticks(range(len(t))[::t_skip],t[::t_skip])
            axs[1].set_ylabel('frequency (Hz)')
            axs[1].set_yticks(range(len(f[:f_cutoff_idx]))[::f_skip],f[:f_cutoff_idx][::f_skip])
            cbar = fig.colorbar(im)
            cbar.ax.set_title('dB/Hz')
            axs[1].set_title(f'Time-Frequency plot, nperseg={nperseg}, noverlap={noverlap}, nfft={nfft}')

            # plotting beta power (approx 13-30Hz)
            f_beta_idx = np.where((f >= 13) & (f <=30))[0]
            beta_power = 10*np.log(s[neuron_num,f_beta_idx,:])
            beta_mean = np.nanmean(beta_power, axis=0)
            beta_std = np.nanstd(beta_power, axis=0)
            beta_sem = np.sqrt(beta_std)/np.sqrt(beta_power.shape[0])

            axs[2].plot(beta_mean)
            axs[2].fill_between(range(len(beta_mean)), beta_mean - beta_sem, beta_mean + beta_sem, alpha=0.3)
            axs[2].set_xlabel('time (s)')
            axs[2].set_xticks(range(len(t))[::t_skip],t[::t_skip])
            axs[2].set_title('beta (13-30Hz) power')
            if len(beta_range) > 0:
                axs[2].set_ylim(beta_range)

            plt.show()

    return f[:f_cutoff_idx], t, s[:,:f_cutoff_idx,:]


def compare_avg_spect(signal, fs, nperseg, noverlap, nfft, f_cutoff, exc_ind, inh_ind,
                   stim1_on, stim1_off, stim2_on, stim2_off,
                   plot=1, t_skip=5, f_skip=2, trial_label='same',
                   IPSC_range = [], spect_range = [], beta_range = []):
    
    # plots average spectrogram of excitatory and inhibitory neurons, putting signal and power on same plot
    
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

        fig, axs = plt.subplots(4,1,figsize=(16,20))
        for i, type in enumerate(type_label):
            
            axs[0].plot(np.mean(signal[inds[i],:], axis=0), label=f'{type}, n={len(inds[i])}')
            axs[0].fill_between(range(len(np.mean(signal[inds[i],:], axis=0))), 
                                np.mean(signal[inds[i],:], axis=0) - np.std(signal[inds[i],:], axis=0), 
                                np.mean(signal[inds[i],:], axis=0) + np.std(signal[inds[i],:], axis=0), alpha=0.3)
            axs[0].set_title(f'{trial_label} 25-trial-average IPSC')
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
                im = (axs[1+i].imshow(10*np.log(s_list[i][:f_cutoff_idx,:]),
                                     aspect='auto', origin='lower', vmin=spect_range[0], vmax=spect_range[1]))
            else:
                im = (axs[1+i].imshow(10*np.log(s_list[i][:f_cutoff_idx,:]), 
                                     aspect='auto', origin='lower'))
            axs[1+i].set_xlabel('time (s)')
            axs[1+i].set_xticks(range(len(t))[::t_skip],t[::t_skip])
            axs[1+i].set_ylabel('frequency (Hz)')
            axs[1+i].set_yticks(range(len(f[:f_cutoff_idx]))[::f_skip],f[:f_cutoff_idx][::f_skip])
            axs[1+i].axvline(x=stim1_on_idx, color='r', linestyle='--')
            axs[1+i].axvline(x=stim1_off_idx, color='r', linestyle='--')
            axs[1+i].axvline(x=stim2_on_idx, color='r', linestyle='--')
            axs[1+i].axvline(x=stim2_off_idx, color='r', linestyle='--')
            cbar = fig.colorbar(im)
            cbar.ax.set_title('dB/Hz')
            axs[1+i].set_title(f'{type} time-frequency plot, nperseg={nperseg}, noverlap={noverlap}, nfft={nfft}')

            # plotting beta power (approx 13-30Hz)
            f_beta_idx = np.where((f >= 13) & (f <=30))[0]
            beta_power = 10*np.log(s_list[i][f_beta_idx,:])
            beta_mean = np.nanmean(beta_power, axis=0)
            beta_std = np.nanstd(beta_power, axis=0)
            beta_sem = np.sqrt(beta_std)/np.sqrt(beta_power.shape[0])

            axs[3].plot(beta_mean, label=type)
            axs[3].fill_between(range(len(beta_mean)), beta_mean - beta_sem, beta_mean + beta_sem, alpha=0.3)
            axs[3].axvline(x=stim1_on_idx, color='r', linestyle='--')
            axs[3].axvline(x=stim1_off_idx, color='r', linestyle='--')
            axs[3].axvline(x=stim2_on_idx, color='r', linestyle='--')
            axs[3].axvline(x=stim2_off_idx, color='r', linestyle='--')
            axs[3].set_xlabel('time (s)')
            axs[3].set_xticks(range(len(t))[::t_skip],t[::t_skip])
            axs[3].set_title('beta (13-30Hz) power')
            if len(beta_range) > 0:
                axs[3].set_ylim(beta_range)

        axs[0].legend()
        axs[3].legend()
        plt.show()

    return f, t, s, s_exc, s_inh


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



'''
MISC
'''


def upsample(signal, fs, upsample_fs):
    # signal is (neurons x time)

    down_time = np.linspace(0, 1, fs)
    up_time = np.linspace(0, 1, upsample_fs)

    n_neurons = signal.shape[0]
    up_signal = np.zeros((n_neurons,len(up_time)))
    for i in range(n_neurons):
        intp1 = scipy.interpolate.interp1d(down_time, signal[i,:], kind='linear')
        up_signal[i,:] = intp1(up_time)

    return up_signal

def downsample(signal, fs_high, fs_low, axis=0):
    '''
    Downsample signal from high to low sampling rate
    Use axis input for the time dimension
    '''
    
    t_high = np.arange(0, signal.shape[axis]/fs_high, 1/fs_high)
    num_samples = int(len(t_high) * fs_low / fs_high)
    signal_low = resample(signal, num_samples, axis=axis)

    return signal_low