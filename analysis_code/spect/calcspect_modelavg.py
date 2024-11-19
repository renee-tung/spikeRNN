# Script to make spectrogram plots across models

import os, scipy.io 
import numpy as np 
# %matplotlib ipympl
import matplotlib.pyplot as plt 
# from mpl_toolkits.mplot3d import axes3d

import sys
sys.path.append('/home/nuttidalab/Documents/spikeRNN/analysis_code/utils/')

import vis_utils as vu
import importlib

# modify these as needed
normalize_ipscs = 1 # usually will be 1
lesion_connections = 'ee' # if not lesioning put 0, else 'ii' etc
longer_delay = 400 # '' if standard delay (150), longer can be 200, 250, 400

models_types = ['good_models','bad_models']
# models_types = ['good_models']
norm_name = ['' if normalize_ipscs == 1 else '_raw']
lesion_name = ['' if lesion_connections == 0 else f'_lesion{lesion_connections}']
delay_name = ['' if longer_delay == '' else f'_delay{longer_delay}'][0]

for models_type in models_types:
    print(f'Running {models_type} models...')

    # load list of models
    models_dir = '/scratch/spikeRNN/models/DMS_OSF/' # dir where all models are located
    results_dir = f'{models_dir}{models_type}/' # dir where results are saved

    model_list_path = f'{results_dir}{models_type}_list.mat' # list (saved from matlab) of models of interest
    model_list_cell = scipy.io.loadmat(model_list_path)['stable_mods'][0] 
    model_list = [model_list_cell[i][0] for i in range(len(model_list_cell))]
    n_models = len(model_list)
    print(f'Total: {n_models} {models_type} models')

    freqbands_list = ['theta','alpha','beta','gamma1','gamma2']

    all_powers_pos_exc = []
    all_powers_pos_inh = []
    all_powers_neg_exc = []
    all_powers_neg_inh = []
    for n_model, model_fname in enumerate(model_list):

        print(f'Loading model {n_model+1}/{n_models}...')
        model_dir = os.path.join(models_dir, model_fname[:model_fname.rfind('.')]) # dir for this model

        # first load model to get exc/inh indices
        mat_data = scipy.io.loadmat(os.path.join(models_dir,model_fname)) # load model data
        exc = mat_data['exc']
        exc_ind = np.where(exc == 1)[0]
        inh = mat_data['inh']
        inh_ind = np.where(inh == 1)[0]
        del mat_data

        # now load IPSC data
        ipsc_savename = f'IPSCs_50travg{norm_name[0]}{lesion_name[0]}{longer_delay}.mat'
        ipsc_data = scipy.io.loadmat(os.path.join(model_dir, ipsc_savename)) # load IPSC data ***
        ipscs = [ipsc_data['ipscs_samepos'], ipsc_data['ipscs_sameneg']] # +1 first trials and -1 first trials
        T = ipsc_data['T'][0][0]
        stim_on = ipsc_data['stim_on'][0][0]
        stim_dur = ipsc_data['stim_dur'][0][0]
        delay = ipsc_data['delay'][0][0]
        del ipsc_data
        
        # params for FFT
        fs_spk = 20000
        fs_rate = 200
        nperseg=1000
        noverlap = nperseg//1.2
        nfft = nperseg*20
        f_cutoff = 200

        # get trial times in real time (seconds)
        stim_on_time = stim_on/fs_rate
        stim_dur_time = stim_dur/fs_rate
        delay_time = delay/fs_rate
        stim1_on = stim_on_time
        stim1_off = stim_on_time + stim_dur_time
        stim2_on = stim_on_time + stim_dur_time + delay_time
        stim2_off = stim_on_time + 2*stim_dur_time + delay_time


        f, t, s_pos = vu.calc_indv_spect(ipscs[0], fs_spk, nperseg, noverlap, nfft, f_cutoff, plot=0)
        _, _, s_neg = vu.calc_indv_spect(ipscs[1], fs_spk, nperseg, noverlap, nfft, f_cutoff, plot=0)
        del ipscs
        
        s_pos_exc = np.nanmean(s_pos[exc_ind,:,:], axis=0)
        s_pos_inh = np.nanmean(s_pos[inh_ind,:,:], axis=0)
        s_neg_exc = np.nanmean(s_neg[exc_ind,:,:], axis=0)
        s_neg_inh = np.nanmean(s_neg[inh_ind,:,:], axis=0)
        
        # save spectral data
        np.save(os.path.join(model_dir, f's_pos_exc{norm_name[0]}{lesion_name[0]}{delay_name}.npy'), s_pos_exc)
        np.save(os.path.join(model_dir, f's_pos_inh{norm_name[0]}{lesion_name[0]}{delay_name}.npy'), s_pos_inh)
        np.save(os.path.join(model_dir, f's_neg_exc{norm_name[0]}{lesion_name[0]}{delay_name}.npy'), s_neg_exc)
        np.save(os.path.join(model_dir, f's_neg_inh{norm_name[0]}{lesion_name[0]}{delay_name}.npy'), s_neg_inh)
        np.save(os.path.join(model_dir, 'f.npy'), f)
        if (n_model == 0) & (models_type == models_types[0]):
            np.save(os.path.join(models_dir, f't{delay_name}.npy'), t)
            print('saved t')

        # now get frequency bands (loop over bands)
        powers_pos_exc = np.zeros((len(freqbands_list), s_pos_exc.shape[1]))
        powers_pos_inh = np.zeros((len(freqbands_list), s_pos_inh.shape[1]))
        powers_neg_exc = np.zeros((len(freqbands_list), s_neg_exc.shape[1]))
        powers_neg_inh = np.zeros((len(freqbands_list), s_neg_inh.shape[1]))
        for i,power in enumerate(freqbands_list):
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

            # calculate power in specified band
            f_band_idx = np.where((f >= power_range[0]) & (f <=power_range[1]))[0]
            powers_pos_exc[i,:] = np.nanmean(10*np.log(s_pos_exc[f_band_idx,:]), axis=0)
            powers_pos_inh[i,:] = np.nanmean(10*np.log(s_pos_inh[f_band_idx,:]), axis=0)
            powers_neg_exc[i,:] = np.nanmean(10*np.log(s_neg_exc[f_band_idx,:]), axis=0)
            powers_neg_inh[i,:] = np.nanmean(10*np.log(s_neg_inh[f_band_idx,:]), axis=0)

        all_powers_pos_exc.append(powers_pos_exc)
        all_powers_pos_inh.append(powers_pos_inh)
        all_powers_neg_exc.append(powers_neg_exc)
        all_powers_neg_inh.append(powers_neg_inh)

        del s_pos, s_neg, s_pos_exc, s_pos_inh, s_neg_exc, s_neg_inh
        del powers_pos_exc, powers_pos_inh, powers_neg_exc, powers_neg_inh

    # convert to matrices
    all_powers_pos_exc = np.array(all_powers_pos_exc) # should be (n_models, n_freqbands, n_timepoints)
    all_powers_pos_inh = np.array(all_powers_pos_inh)
    all_powers_neg_exc = np.array(all_powers_neg_exc)
    all_powers_neg_inh = np.array(all_powers_neg_inh)

    # get band powers from each spectrogram
    powers_pos_mean = [np.mean(all_powers_pos_exc, axis=0), 
                    np.mean(all_powers_pos_inh, axis=0)]
    powers_pos_sem = [np.std(all_powers_pos_exc, axis=0)/np.sqrt(n_models),
                        np.std(all_powers_pos_inh, axis=0)/np.sqrt(n_models)]
    powers_neg_mean = [np.mean(all_powers_neg_exc, axis=0),
                    np.mean(all_powers_neg_inh, axis=0)]
    powers_neg_sem = [np.std(all_powers_neg_exc, axis=0)/np.sqrt(n_models),
                        np.std(all_powers_neg_inh, axis=0)/np.sqrt(n_models)]

    # plot data
    stim1_on_idx = np.where(t <= stim1_on)[0][-1]
    stim1_off_idx = np.where(t <= stim1_off)[0][-1]
    stim2_on_idx = np.where(t <= stim2_on)[0][-1]
    stim2_off_idx = np.where(t <= stim2_off)[0][-1]

    t_skip = 15

    for i, power in enumerate(freqbands_list):
        print(f'Plotting {power} power...')
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
        plt.figure(figsize=(16,5))
        for j, neuron_type in enumerate(['excitatory','inhibitory']):
            for k, trial_type in enumerate(['+1', '-1']):
                if trial_type == '+1':
                    band_mean = powers_pos_mean[j][i,:]
                    band_sem = powers_pos_sem[j][i,:]
                else:
                    band_mean = powers_neg_mean[j][i,:]
                    band_sem = powers_neg_sem[j][i,:]
                plt.plot(band_mean, label=f'{trial_type} stim1, {neuron_type}', c=['r','b'][k], alpha=1, linestyle=['-','--'][j])
                plt.fill_between(range(len(band_mean)), band_mean - band_sem, band_mean + band_sem, color=['r','b'][k], alpha=0.3)
        plt.axvline(x=stim1_on_idx, color='r', linestyle='--')
        plt.axvline(x=stim1_off_idx, color='r', linestyle='--')
        plt.axvline(x=stim2_on_idx, color='r', linestyle='--')
        plt.axvline(x=stim2_off_idx, color='r', linestyle='--')
        plt.xlabel('time (s)')
        plt.ylabel('average power (dB)')
        plt.xticks(range(len(t))[::t_skip],t[::t_skip])
        # plt.ylim([-10,10])
        if len(lesion_name[0]) > 0:
            lesion_name_title = f', lesioning {lesion_name[0][-2:]}'
        else:
            lesion_name_title = ''
        plt.title(f'{power} ({power_range[0]}-{power_range[1]}Hz) power {norm_name[0]}{lesion_name_title} {delay/fs_rate*1000}ms delay')
        plt.legend()
        plt.savefig(f'{results_dir}{power}_power{norm_name[0]}{lesion_name[0]}{delay_name}.png')