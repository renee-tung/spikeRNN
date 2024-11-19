# Script to make PAC plots across models

import os, scipy.io 
import numpy as np 
import matplotlib.pyplot as plt 
import pickle as pk

import vis_utils as vu
import pdb

# modify these as needed
normalize_ipscs = 1 # usually will be 1
lesion_connections = 0 # if not lesioning put 0, else 'ii' etc
longer_delay = '' # '' if standard delay (150), longer can be 200 or 250

models_types = ['good_models','bad_models']
# models_types = ['good_models']
norm_name = ['' if normalize_ipscs == 1 else '_raw']
lesion_name = ['' if lesion_connections == 0 else f'_lesion{lesion_connections}']
delay_name = ['' if longer_delay == '' else f'_delay{longer_delay}'][0]

all_mean_pacs = []
all_mean_diff_pacs = []
for models_type in models_types: # for good and bad models
    print(f'Running {models_type} models...')

    # load list of models
    models_dir = '/home/nuttidalab/Documents/spikeRNN/models/DMS_OSF/' # dir where all models are located
    results_dir = f'{models_dir}{models_type}/' # dir where results are saved

    model_list_path = f'{results_dir}{models_type}_list.mat' # list (saved from matlab) of models of interest
    model_list_cell = scipy.io.loadmat(model_list_path)['stable_mods'][0] 
    model_list = [model_list_cell[i][0] for i in range(len(model_list_cell))]
    n_models = len(model_list)
    print(f'Total: {n_models} {models_type} models')

    mean_pacs = []
    mean_diff_pacs = []
    for n_model, model_fname in enumerate(model_list): # for each individual model
        
        print(f'Loading model {n_model+1}/{n_models}...')
        model_dir = os.path.join(models_dir, model_fname[:model_fname.rfind('.')]) # dir for this model

        # params for pacs
        fs_spk = 20000
        fs_rate = 200
        trial_labels = ['+1','-1']
        neuron_labels = ['exc','inh']

        # load pac data
        with open (os.path.join(model_dir, f'pac_objs{lesion_name[0]}{delay_name}.pkl'), 'rb') as f:
            [all_pac_objs, all_pacs, times, freqs, stim1_on, stim1_off, stim2_on, stim2_off] = pk.load(f)

        settings = {
            'stim1_on': stim1_on,
            'stim1_off': stim1_off,
            'stim2_on': stim2_on,
            'stim2_off': stim2_off,
        }

        # calculate mean pac across models, and mean difference btwn +1/-1 trials
        for i, neuron_label in enumerate(neuron_labels): # excitatory and inhibitory neurons
            for j, trial_label in enumerate(trial_labels): # +1 and -1 trials
                pac = np.squeeze(all_pacs[trial_label][neuron_label])
                if n_model == 0:
                    mean_pacs.append(np.zeros_like(pac))
                    mean_diff_pacs.append(np.zeros_like(pac))
                    pac_obj = all_pac_objs[trial_label][neuron_label] # load one pac_obj (should all be usable since same timescale)
                mean_pacs[i*len(neuron_labels) + j] += pac/n_models # add to mean. list is [exc+1, exc-1, inh+1, inh-1]
                mean_diff_pacs[i] += np.squeeze((all_pacs[trial_labels[0]][neuron_label] - all_pacs[trial_labels[1]][neuron_label]))/n_models # difference between +1 and -1 trials for each neuron type. list is [exc diff, inh diff]


    # plot mean pacs and mean difference between trial types, for the model type
    for i, neuron_label in enumerate(neuron_labels): # excitatory and inhibitory neurons
        for j, trial_label in enumerate(trial_labels): # +1 and -1 trials
            vu.plot_pac(pac_obj, mean_pacs[i*len(neuron_labels) + j], 
                    title = f'Mean delay PAC, {neuron_label}, {trial_label}', vmin=0, vmax=.006,
                    save=1, savename = f'{results_dir}pac_mean_{trial_label}{neuron_label}{norm_name[0]}{lesion_name[0]}{delay_name}.png') # plot mean pac
        vu.plot_pac(pac_obj, mean_diff_pacs[i], 
                    title = f'Mean delay PAC difference, {neuron_label}, +1 minus -1',
                    vmin=-.002, vmax=.002, cmap='bwr',
                    save=1, savename = f'{results_dir}pac_trialdiff_mean_{neuron_label}{norm_name[0]}{lesion_name[0]}{delay_name}.png') # plot mean difference between +1 and -1 trials for each neuron type
    all_mean_pacs.append(mean_pacs)
    all_mean_diff_pacs.append(mean_diff_pacs)

results_dir = f'{models_dir}models_good+bad/'
# plot difference between good and bad models; and then difference between +1/-1 trials between good and bad models
for i, neuron_label in enumerate(neuron_labels):
    for j, trial_label in enumerate(trial_labels):
        vu.plot_pac(pac_obj, all_mean_pacs[0][i*len(neuron_labels) + j] - all_mean_pacs[1][i*len(neuron_labels) + j], 
                title = f'Mean delay PAC difference, {neuron_label}, {trial_label}, good minus bad',
                vmin=-.002, vmax=.002, cmap='bwr',
                save=1, savename = f'{results_dir}pac_goodbaddiff_{trial_label}{neuron_label}{norm_name[0]}{lesion_name[0]}{delay_name}.png')
    vu.plot_pac(pac_obj, all_mean_diff_pacs[0][i] - all_mean_diff_pacs[1][i], 
                title = f'Mean delay PAC difference, {neuron_label}, +1 minus -1, then good minus bad',
                vmin=-.002, vmax=.002, cmap='bwr',
                save=1, savename = f'{results_dir}pac_goodbaddiff_trialdiff_{neuron_label}{norm_name[0]}{lesion_name[0]}{delay_name}.png')



