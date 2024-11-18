# Script to make ERPAC plots across models (good vs bad)

import os, scipy.io 
import numpy as np 
import matplotlib.pyplot as plt 
import pickle as pk

import vis_utils as vu
import pdb

import sys
sys.path.append('/home/nuttidalab/Documents/spikeRNN/utils/')
from bootstrap_method import *


# modify these as needed
normalize_ipscs = 1 # usually will be 1
lesion_connections = 0 # if not lesioning put 0, else 'ii' etc
longer_delay = '' # '' if standard delay (150), longer can be 200 or 250

models_types = ['good_models','bad_models']
# models_types = ['good_models']
norm_name = ['' if normalize_ipscs == 1 else '_raw']
lesion_name = ['' if lesion_connections == 0 else f'_lesion{lesion_connections}']
delay_name = ['' if longer_delay == '' else f'_delay{longer_delay}'][0]

# bootstrapping params
random_seed = 820
nboot = 1000
CI_int = (2.5, 97.5)  # 95% confidence interval

all_mean_erpacs = []
all_mean_diff_erpacs = []
all_delay_erpacs = []
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

    mean_erpacs = []
    mean_diff_erpacs = []
    delay_erpacs = []
    for n_model, model_fname in enumerate(model_list): # for each individual model
        
        print(f'Loading model {n_model+1}/{n_models}...')
        model_dir = os.path.join(models_dir, model_fname[:model_fname.rfind('.')]) # dir for this model

        # params for erpacs
        fs_spk = 20000
        fs_rate = 200
        trial_labels = ['+1','-1']
        neuron_labels = ['exc','inh']

        # load erpac data
        with open (os.path.join(model_dir, f'erpac_objs{lesion_name[0]}{delay_name}.pkl'), 'rb') as f:
            [all_rp_objs, all_erpacs, times, freqs, stim1_on, stim1_off, stim2_on, stim2_off] = pk.load(f)

        settings = {
            'stim1_on': stim1_on,
            'stim1_off': stim1_off,
            'stim2_on': stim2_on,
            'stim2_off': stim2_off,
        }

        delay_on = int(stim1_off*fs_spk) # calculate delay period times (in fs_spk time)
        delay_off = int(stim2_on*fs_spk)

        # calculate mean erpac across models, mean difference btwn +1/-1 trials, mean over delay period
        this_delay_erpacs = []
        for i, neuron_label in enumerate(neuron_labels): # excitatory and inhibitory neurons
            for j, trial_label in enumerate(trial_labels): # +1 and -1 trials
                erpac = all_erpacs[trial_label][neuron_label]
                if n_model == 0:
                    mean_erpacs.append(np.zeros_like(erpac))
                    mean_diff_erpacs.append(np.zeros_like(erpac))
                    rp_obj = all_rp_objs[trial_label][neuron_label] # load one rp_obj (should all be usable since same timescale)
                mean_erpacs[i*len(neuron_labels) + j] += erpac/n_models # add to mean. list is [exc+1, exc-1, inh+1, inh-1]
                mean_diff_erpacs[i] += (all_erpacs[trial_labels[0]][neuron_label] - all_erpacs[trial_labels[1]][neuron_label])/n_models # difference between +1 and -1 trials for each neuron type. list is [exc diff, inh diff]
                this_delay_erpacs.append(np.mean(erpac[:,delay_on:delay_off], axis=1)) # mean over delay period
        delay_erpacs.append(np.array(this_delay_erpacs)) # appending a 4 x n_freqs array for each model


    # plot mean erpacs and mean difference between trial types, for the model type
    for i, neuron_label in enumerate(neuron_labels): # excitatory and inhibitory neurons
        for j, trial_label in enumerate(trial_labels): # +1 and -1 trials
            vu.plot_erpac(rp_obj, mean_erpacs[i*len(neuron_labels) + j], times, freqs, fs_spk, settings, settings_real=1,
                    title = f'Mean ERPAC, {neuron_label}, {trial_label}', vmin=.15, vmax=.35,
                    save=1, savename = f'{results_dir}erpac_mean_{trial_label}{neuron_label}{norm_name[0]}{lesion_name[0]}{delay_name}.png') # plot mean erpac
        vu.plot_erpac(rp_obj, mean_diff_erpacs[i], times, freqs, fs_spk, settings, settings_real=1,
                    title = f'Mean ERPAC difference, {neuron_label}, +1 minus -1',
                    vmin=-.5, vmax=.5, cmap='bwr',
                    save=1, savename = f'{results_dir}erpac_trialdiff_mean_{neuron_label}{norm_name[0]}{lesion_name[0]}{delay_name}.png') # plot mean difference between +1 and -1 trials for each neuron type
    all_mean_erpacs.append(mean_erpacs)
    all_mean_diff_erpacs.append(mean_diff_erpacs)

    # store mean delay erpacs across models of this model type
    all_delay_erpacs.append(np.array(delay_erpacs)) # appending a n_models x 4 x n_freqs array for each model type

results_dir = f'{models_dir}models_good+bad/'
# plot difference between good and bad models; and then difference between +1/-1 trials between good [0] and bad [1] models
for i, neuron_label in enumerate(neuron_labels):
    for j, trial_label in enumerate(trial_labels):
        vu.plot_erpac(rp_obj, all_mean_erpacs[0][i*len(neuron_labels) + j] - all_mean_erpacs[1][i*len(neuron_labels) + j], times, freqs, fs_spk, settings, settings_real=1,
                title = f'Mean ERPAC difference, {neuron_label}, {trial_label}, good minus bad',
                vmin=-.15, vmax=.15, cmap='bwr',
                save=1, savename = f'{results_dir}erpac_goodbaddiff_{trial_label}{neuron_label}{norm_name[0]}{lesion_name[0]}{delay_name}.png')
    vu.plot_erpac(rp_obj, all_mean_diff_erpacs[0][i] - all_mean_diff_erpacs[1][i], times, freqs, fs_spk, settings, settings_real=1,
                title = f'Mean ERPAC difference, {neuron_label}, +1 minus -1, then good minus bad',
                vmin=-.25, vmax=.25, cmap='bwr',
                save=1, savename = f'{results_dir}erpac_goodbaddiff_trialdiff_{neuron_label}{norm_name[0]}{lesion_name[0]}{delay_name}.png')


## plot ERPACs across frequencies, averaging over delay period, for good and bad models — yes could've put in the last for loop oh well
for i, neuron_label in enumerate(neuron_labels):
    for j, trial_label in enumerate(trial_labels):
        good_erpac = np.squeeze(all_delay_erpacs[0][:, i*len(neuron_labels) + j, :]) # n_models x 1 x n_freqs (then squeeze dims)
        bad_erpac = np.squeeze(all_delay_erpacs[1][:, i*len(neuron_labels) + j, :])
        good_erpac_delay = np.mean(good_erpac, axis=0)
        bad_erpac_delay = np.mean(bad_erpac, axis=0)
        good_erpac_delay_sem = np.std(good_erpac, axis=0)/np.sqrt(good_erpac.shape[0])
        bad_erpac_delay_sem = np.std(bad_erpac, axis=0)/np.sqrt(bad_erpac.shape[0])

        # bootstrap
        _, _, _, pdiff = fnc_time_bootstrap_optimized(good_erpac, # n x time
                                                    bad_erpac, 
                                                    nboot, CI_int, n_jobs=10, random_seed=random_seed)
        x_vector = freqs
        significant_timepoints = x_vector[pdiff < 0.05]

        # plot
        plt.figure(figsize=(16,4))
        plt.plot(freqs, good_erpac_delay, label='good', color='r')
        plt.fill_between(freqs, good_erpac_delay-good_erpac_delay_sem, good_erpac_delay+good_erpac_delay_sem, color='r', alpha=.2)
        plt.plot(freqs, bad_erpac_delay, label='bad', color='b')
        plt.fill_between(freqs, bad_erpac_delay-bad_erpac_delay_sem, bad_erpac_delay+bad_erpac_delay_sem, color='b', alpha=.2)
        plt.ylim([.21,.265])
        plt.scatter(significant_timepoints, np.zeros_like(significant_timepoints)+.260, color='k', label='_nolegend_', marker='s')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('ERPAC')
        plt.title(f'Mean ERPAC during delay, {neuron_label}, {trial_label}')
        plt.legend()
        plt.savefig(f'{results_dir}erpac_delay_{trial_label}{neuron_label}{norm_name[0]}{lesion_name[0]}{delay_name}.png')
        plt.close()

# now plot all of above on one figure
colors = ['r','b', 'm','c']
fig, axs = plt.subplots(2, 2, figsize=(16,8))
for i, neuron_label in enumerate(neuron_labels):
    for j, trial_label in enumerate(trial_labels):
        this_idx = i*len(neuron_labels) + j
        good_erpac = np.squeeze(all_delay_erpacs[0][:, this_idx, :]) # n_models x 1 x n_freqs (then squeeze dims)
        bad_erpac = np.squeeze(all_delay_erpacs[1][:, this_idx, :])
        good_erpac_delay = np.mean(good_erpac, axis=0)
        bad_erpac_delay = np.mean(bad_erpac, axis=0)
        good_erpac_delay_sem = np.std(good_erpac, axis=0)/np.sqrt(good_erpac.shape[0])
        bad_erpac_delay_sem = np.std(bad_erpac, axis=0)/np.sqrt(bad_erpac.shape[0])

        # bootstrap
        _, _, _, pdiff = fnc_time_bootstrap_optimized(good_erpac, # n x time
                                                    bad_erpac, 
                                                    nboot, CI_int, n_jobs=10, random_seed=random_seed)
        x_vector = freqs
        significant_timepoints = x_vector[pdiff < 0.05]

        # plot
        if neuron_label == 'exc':
            ax = axs[0,0]
            ax.set_title('Excitatory neurons')
            ax.set_ylabel('ERPAC')
        else:
            ax = axs[0,1]
            ax.set_title('Inhibitory neurons')
        
        ax.plot(freqs, good_erpac_delay, label=f'good {neuron_label} {trial_label}', color=colors[this_idx])
        ax.fill_between(freqs, good_erpac_delay-good_erpac_delay_sem, good_erpac_delay+good_erpac_delay_sem, color=colors[this_idx], alpha=.2)
        ax.plot(freqs, bad_erpac_delay, label=f'bad {neuron_label} {trial_label}', color=colors[this_idx], linestyle='--')
        ax.fill_between(freqs, bad_erpac_delay-bad_erpac_delay_sem, bad_erpac_delay+bad_erpac_delay_sem, color=colors[this_idx], alpha=.2)
        ax.scatter(significant_timepoints, np.zeros_like(significant_timepoints)+.260, color='k', label='_nolegend_', marker='s')

        if trial_label == '+1':
            ax = axs[1,0]
            ax.set_title('Trial +1')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('ERPAC')
        else:
            ax = axs[1,1]
            ax.set_title('Trial -1')
            ax.set_xlabel('Frequency (Hz)')
        
        ax.plot(freqs, good_erpac_delay, label=f'good {neuron_label} {trial_label}', color=colors[this_idx])
        ax.fill_between(freqs, good_erpac_delay-good_erpac_delay_sem, good_erpac_delay+good_erpac_delay_sem, color=colors[this_idx], alpha=.2)
        ax.plot(freqs, bad_erpac_delay, label=f'bad {neuron_label} {trial_label}', color=colors[this_idx], linestyle='--')
        ax.fill_between(freqs, bad_erpac_delay-bad_erpac_delay_sem, bad_erpac_delay+bad_erpac_delay_sem, color=colors[this_idx], alpha=.2)

# Add legends for each subplot
for ax in axs.flat:
    ax.legend()
    ax.set_ylim([.21,.265])

plt.tight_layout()
plt.savefig(f'{results_dir}erpac_delay_all_panel{norm_name[0]}{lesion_name[0]}{delay_name}.png')
plt.close()

# now plot combining +1/-1 trials
fig, axs = plt.subplots(1, 2, figsize=(16,4))
for i, neuron_label in enumerate(neuron_labels):
    this_idx = i*len(neuron_labels)
    good_erpac = np.reshape(all_delay_erpacs[0][:, this_idx:this_idx+1, :], (all_delay_erpacs[0].shape[0],-1)) # n_models x 2*n_freqs (then squeeze dims)
    bad_erpac = np.reshape(all_delay_erpacs[1][:, this_idx:this_idx+1, :], (all_delay_erpacs[1].shape[0],-1))
    good_erpac_delay = np.mean(good_erpac, axis=0)
    bad_erpac_delay = np.mean(bad_erpac, axis=0)
    good_erpac_delay_sem = np.std(good_erpac, axis=0)/np.sqrt(good_erpac.shape[0])
    bad_erpac_delay_sem = np.std(bad_erpac, axis=0)/np.sqrt(bad_erpac.shape[0])

    # bootstrap
    _, _, _, pdiff = fnc_time_bootstrap_optimized(good_erpac, # n x time
                                                bad_erpac, 
                                                nboot, CI_int, n_jobs=10, random_seed=random_seed)
    x_vector = freqs
    significant_timepoints = x_vector[pdiff < 0.05]

    # plot
    ax = axs[i]
    ax.set_title(f'{neuron_label} neurons')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('ERPAC')
    ax.plot(freqs, good_erpac_delay, label=f'good {neuron_label}', color='r')
    ax.fill_between(freqs, good_erpac_delay-good_erpac_delay_sem, good_erpac_delay+good_erpac_delay_sem, color='r', alpha=.2)
    ax.plot(freqs, bad_erpac_delay, label=f'bad {neuron_label}', color='b')
    ax.fill_between(freqs, bad_erpac_delay-bad_erpac_delay_sem, bad_erpac_delay+bad_erpac_delay_sem, color='b', alpha=.2)
    ax.scatter(significant_timepoints, np.zeros_like(significant_timepoints)+.260, color='k', label='_nolegend_', marker='s')
# Add legends for each subplot 
for ax in axs.flat:
    ax.set_ylim([.21,.265])
    ax.legend()
plt.tight_layout()
plt.savefig(f'{results_dir}erpac_delay_combinetrials{norm_name[0]}{lesion_name[0]}{delay_name}.png')
plt.close()