# Script to make pac plots across models
# 

import os, scipy.io 
import numpy as np 
# %matplotlib ipympl
import matplotlib.pyplot as plt 
import pickle as pk
# from mpl_toolkits.mplot3d import axes3d

import sys
sys.path.append('/home/nuttidalab/Documents/spikeRNN/analysis_code/utils/')

import vis_utils as vu
import importlib
import pdb

pdb.set_trace()
# modify these as needed
normalize_ipscs = 1 # usually will be 1
lesion_connections = 'ii' # if not lesioning put 0, else 'ii' etc
longer_delay = '' # '' if standard delay (150), longer can be 200, 250, 400


# models_types = ['good_models','bad_models']
models_types = ['good_models']
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

    for n_model, model_fname in enumerate(model_list):

        print(f'Loading model {n_model+1}/{n_models}...')
        model_dir = os.path.join(models_dir, model_fname[:model_fname.rfind('.')]) # dir for this model

        # skip if we've already saved this model's pacs
        if os.path.exists(os.path.join(model_dir, f'pac_objs{lesion_name[0]}{delay_name}.pkl')):
            print(f'Skipping model {n_model+1}/{n_models}...')
            continue

        # first load model to get exc/inh indices
        mat_data = scipy.io.loadmat(os.path.join(models_dir,model_fname)) # load model data
        exc = mat_data['exc']
        exc_ind = np.where(exc == 1)[0]
        inh = mat_data['inh']
        inh_ind = np.where(inh == 1)[0]
        del mat_data

        # now load IPSC data
        ipsc_savename = f'IPSCs_30trall{norm_name[0]}{lesion_name[0]}{longer_delay}.mat'
        ipsc_data = scipy.io.loadmat(os.path.join(model_dir, ipsc_savename)) # load IPSC data ***
        ipscs = [ipsc_data['ipscs_samepos'], ipsc_data['ipscs_sameneg']] # +1 first trials and -1 first trials
        T = ipsc_data['T'][0][0]
        stim_on = ipsc_data['stim_on'][0][0]
        stim_dur = ipsc_data['stim_dur'][0][0]
        delay = ipsc_data['delay'][0][0]
        del ipsc_data
        
        # params for pacs
        fs_spk = 20000
        fs_rate = 200
        trial_labels = ['+1','-1']
        neuron_labels = ['exc','inh']

        # get trial times in real time (seconds)
        stim_on_time = stim_on/fs_rate
        stim_dur_time = stim_dur/fs_rate
        delay_time = delay/fs_rate
        stim1_on = stim_on_time
        stim1_off = stim_on_time + stim_dur_time
        stim2_on = stim_on_time + stim_dur_time + delay_time
        stim2_off = stim_on_time + 2*stim_dur_time + delay_time

        # in fs_spk time
        delay_onset = int(stim1_off*fs_spk)
        delay_offset = int(stim2_on*fs_spk)

        # calculate all pacs and save

        all_pac_objs = {}
        all_pacs = {}
        for i, ipsc in enumerate(ipscs): # +1 and -1 trials
            pac_objs = {}
            pacs = {}
            for j, neuron_inds in enumerate([exc_ind, inh_ind]):
                pac_obj, pac = vu.calculate_pac(ipsc, fs_spk, delay_onset, delay_offset, neuron_inds, plot=0, 
                                        idpac=(2,1,0), f_pha=(2, 14, 2, 1), f_amp=(30, 200, 2, 1), 
                                        dcomplex='hilbert', cycle=(3, 6), width=7, n_bins=18)

                pac_objs[neuron_labels[j]] = pac_obj
                pacs[neuron_labels[j]] = pac
            all_pac_objs[trial_labels[i]] = pac_objs
            all_pacs[trial_labels[i]] = pacs

        #save
        times = np.arange(ipsc.shape[1])
        freqs = pac_obj.yvec
        with open(os.path.join(model_dir, f'pac_objs{lesion_name[0]}{delay_name}.pkl'), 'wb') as f:  
            pk.dump([all_pac_objs, all_pacs, times, freqs, stim1_on, stim1_off, stim2_on, stim2_off], f) # list of exc and inh pac_objs and pacs, with time and freq

        print(f'Saved pacs for model {n_model+1}/{n_models}...')
        del ipscs, all_pac_objs, all_pacs, times, freqs, stim1_on, stim1_off, stim2_on, stim2_off # clear up space