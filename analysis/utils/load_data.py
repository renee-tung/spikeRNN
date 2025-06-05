
# Name: Renee Tung
# Contact: rjt2155@cumc.columbia.edu  
# Last updated: April 2025  
# Filename: load_data.py  
# Description: Script to load data generated from models
# Note: First, xx.
#.      Second, xx  
#       This script depends on xx

'''
FUNCTIONS FOR LOADING SAVED DATA
'''

import os
import numpy as np
import pandas as pd
import pickle as pk
import h5py
import scipy.io as sio

import single_neuron_utils as sn



def get_immediate_subdirs(dir_name):
    return [name for name in os.listdir(dir_name)
            if os.path.isdir(os.path.join(dir_name, name))]

def find_files_ext(dir_name, ext):
    '''
    Find all files with the given extension in the given directory
    '''
    files = []
    for root, dirs, filenames in os.walk(dir_name):
        for filename in filenames:
            if filename.endswith(ext):
                files.append(os.path.join(root, filename))
    return files

def find_files_keywords(dir_name, keywords):
    '''
    Find all files with all of the given keywords in the given directory
    '''
    files = []
    for root, dirs, filenames in os.walk(dir_name):
        for filename in filenames:
            if all(keyword in filename for keyword in keywords):
                files.append(os.path.join(root, filename))

    if len(files) == 0:
        print(f'No files found with keywords {keywords} in {dir_name}')

    return files


def rmv_lowfr_neurons(neuron_idxs, N, spk_df=None, rates_data=None, lfp_data=None, 
                      connectivity_df=None, W=None, exc_ind=None, inh_ind = None, remove_lowfr=True):
    '''
    Remove low firing rate neurons from the data
    neuron_idxs: list of neuron indices to remove
    N: total number of neurons (pre-removal)
    '''
    results = {}
    
    if not remove_lowfr:
        return None
    
    # get new neuron indices after removing low firing rate neurons
    idxs_old = np.arange(N)
    idxs_old = np.setdiff1d(idxs_old, neuron_idxs)  # keep neurons that are not in neuron_idxs
    idxs_new = np.arange(len(idxs_old))  # new indices for remaining neurons
    idx_map = np.zeros(N)  # map old indices to new indices
    idx_map[:] = np.nan  # fill with nan to indicate removed neurons
    for new_idx, old_idx in enumerate(idxs_old):
        idx_map[old_idx] = new_idx
    results['idxs_old'] = idxs_old
    results['neuron_idxs'] = neuron_idxs
    results['N'] = N - len(neuron_idxs)  # new number of neurons after removal
    results['idxs_new'] = idxs_new  # new indices for remaining neurons
    results['idx_map'] = idx_map  # map from old indices to new indices
    
    if spk_df is not None:
        # Remove low firing rate neurons from spike data
        spk_df = spk_df[~spk_df['cell_id'].isin(neuron_idxs)]
        results['spk_df'] = spk_df
        
    if rates_data is not None:
        # Remove low firing rate neurons from rates data (time x neurons x trials)
        rates_data = np.delete(rates_data, neuron_idxs, axis=1)
        results['rates_data'] = rates_data
    
    if lfp_data is not None:
        # Remove low firing rate neurons from LFP data (time x neurons x trials)
        lfp_data = np.delete(lfp_data, neuron_idxs, axis=1)
        results['lfp_data'] = lfp_data
    
    if connectivity_df is not None:
        # Remove low firing rate neurons from connectivity data
        connectivity_df = connectivity_df[~connectivity_df['presyn_id'].isin(neuron_idxs)]
        connectivity_df = connectivity_df[~connectivity_df['postsyn_id'].isin(neuron_idxs)]
        results['connectivity_df'] = connectivity_df
        
    if W is not None:
        # Remove low firing rate neurons from weights matrix
        W = np.delete(W, neuron_idxs, axis=0)
        W = np.delete(W, neuron_idxs, axis=1)
        results['W'] = W
        
    if exc_ind is not None:
        # Remove low firing rate neurons from exc and inh indices
        exc_id = np.setdiff1d(exc_ind, neuron_idxs)
        results['exc_id'] = exc_id
        exc_ind = idx_map[exc_id]  # map old indices to new indices
        results['exc_ind'] = exc_ind.astype(int)
    
    if inh_ind is not None:
        inh_id = np.setdiff1d(inh_ind, neuron_idxs)
        results['inh_id'] = inh_id
        inh_ind = idx_map[inh_id]
        results['inh_ind'] = inh_ind.astype(int)
        
    return results
        
        

def load_lowfr_idxs(model_name, condn_phrase, condn_num='', threshold=2,
                   all_models_dir='/home/nuttidalab/Documents/renee/all_DMS_models/'): #all_models_dir='/scratch/all_DMS_models/',):
    '''
    Load the low firing rate neurons for a given model and condition
    '''
    # Get the data path
    model_dir = os.path.join(all_models_dir, model_name)
    condn_key = f'{condn_phrase}{condn_num}'
    lowfr_path = find_files_keywords(model_dir, ['lowfr', condn_key, str(threshold)])
    
    if not lowfr_path:
        _,spk_df,_ = load_neural_data(model_name, condn_phrase, condn_num, load_spikes=True)
        times_spk, _, _ = get_times_dict('spk', condn_phrase, condn_num, model_name)
        neuron_rmv_idxs = sn.calc_lowfr_neurons(spk_df, times_spk, threshold=threshold)
        np.save(os.path.join(model_dir, f'lowfr_{condn_key}_{threshold}.npy'), neuron_rmv_idxs)
    else:
        lowfr_path = lowfr_path[0]
        neuron_rmv_idxs = np.load(lowfr_path)
    
    return neuron_rmv_idxs


def load_neural_data(model_name, condn_phrase, condn_num='', 
                     all_models_dir='/home/nuttidalab/Documents/renee/all_DMS_models/', #all_models_dir='/scratch/all_DMS_models/',
                     load_LFP=False, load_spikes=False, load_rates=False, remove_lowfr=False):
    '''
    Load neural data from a saved model
    condn_phrase is a string specifying the condition to load, eg 'delay150'
    '''
    # Get the data path
    model_dir = os.path.join(all_models_dir, model_name)
    condn_key = f'{condn_phrase}{condn_num}'
    neuraldata_path = find_files_keywords(model_dir, ['neuraldata', condn_key])[0]
    
    exc_ind, inh_ind = get_celltype_label(model_name, condn_phrase, condn_num)

    # Load the data
    f = h5py.File(neuraldata_path, 'r')

    if load_LFP:
        lfp_data = f['all_lfp'][:] # time x neurons x trials
        # lfp_data = lfp_data.transpose(1, 2, 0) # neurons x trials x time
        N = lfp_data.shape[1]  # number of neurons
    else:
        lfp_data = None

    if load_spikes:
        refs = f['all_spk_times']['spk_times']#[:]
        spk_times_list = []
        for i in range(refs.shape[0]):
            inner_refs = f[refs[i][0]][:]  # This gets you the array of inner object refs
            spike_times = [f[ref][:] for ref in inner_refs.flat]  # Flatten in case it's (1, N) shape
            spk_times_list.append(spike_times)
        
        exc_ind_set = set(exc_ind)  # Faster lookup
        data_rows = []

        for n_trial, trial_spk_times in enumerate(spk_times_list):
            for n_cell, cell_spk_times in enumerate(trial_spk_times):
                cell_type = 'exc' if n_cell in exc_ind_set else 'inh'
                data_rows.append({
                    'cell_id': n_cell,
                    'trial_id': n_trial,
                    'cell_type': cell_type,
                    'spk_times': cell_spk_times.flatten(),
                    'model_name': model_name,
                    condn_phrase: condn_num,
                })

        spk_times_df = pd.DataFrame(data_rows)
        N = len(spk_times_df['cell_id'].unique())  # number of neurons

    else:
        spk_times_df = None

    if load_rates:
        rates = f['all_rates'][:] # time x neurons x trials
        # rates = rates.transpose(1, 2, 0) # neurons x trials x time
        N = rates.shape[1]  # number of neurons
    else:
        rates = None
        
    if remove_lowfr:
        neuron_rmv_idxs = load_lowfr_idxs(model_name, condn_phrase, condn_num, threshold=2)
        results = rmv_lowfr_neurons(neuron_rmv_idxs, N, spk_df=spk_times_df, rates_data=rates, lfp_data=lfp_data, 
                      connectivity_df=None, W=None, exc_ind=exc_ind, inh_ind = inh_ind)
        return results

    return lfp_data, spk_times_df, rates
    

    
def load_bhv_data(model_name, condn_phrase, condn_num='', 
                  all_models_dir='/home/nuttidalab/Documents/renee/all_DMS_models/'): #all_models_dir='/scratch/all_DMS_models/',):
    '''
    Load behavioral data from a saved model
    condn_phrase is a string specifying the condition to load, eg 'delay150'
    '''
    # Get the data path
    model_dir = os.path.join(all_models_dir, model_name)
    condn_key = f'{condn_phrase}{condn_num}'
    bhvdata_path = find_files_keywords(model_dir, ['bhvdata', condn_key])[0]
    if bhvdata_path == []:
        print(f'No behavioral data found for {model_name} with condition {condn_phrase}{condn_num}')
        return None, None

    # Load the data
    bhv_mat = sio.loadmat(bhvdata_path)
    trial_labels = bhv_mat['all_trial_labels']
    trial_perfs = bhv_mat['all_trial_perfs'].flatten()

    return trial_labels, trial_perfs

def load_timing_data(model_name, condn_phrase, condn_num='', 
                     all_models_dir='/home/nuttidalab/Documents/renee/all_DMS_models/'): #all_models_dir='/scratch/all_DMS_models/',):
    '''
    Load timing data from a saved model
    condn_phrase is a string specifying the condition to load, eg 'delay150'
    '''
    # Get the data path
    model_dir = os.path.join(all_models_dir, model_name)
    condn_key = f'{condn_phrase}{condn_num}'
    timingdata_path = find_files_keywords(model_dir, ['timingdata', condn_key])[0]

    # Load the data
    timing_mat = sio.loadmat(timingdata_path)
    
    times_rate = {
    'T': timing_mat['T'][0][0],
    'stim_on': timing_mat['stim_on'][0][0],
    'stim_dur': timing_mat['stim_dur'][0][0],
    'delay': timing_mat['delay'][0][0],
    'fs': timing_mat['fs_rate'][0][0],
    }

    fs_dict = {
        'rate': timing_mat['fs_rate'][0][0], # refers to rate network
        'spk': timing_mat['fs_spk'][0][0],
        'ds': timing_mat['fs_ds'][0][0], # applies to lfp and to firing rate
    }

    times_real = {
        'stim1_on': times_rate['stim_on']/times_rate['fs'],
        'stim1_off': (times_rate['stim_on'] + times_rate['stim_dur'])/times_rate['fs'],
        'stim2_on': (times_rate['stim_on'] + times_rate['stim_dur'] + times_rate['delay'])/times_rate['fs'],
        'stim2_off': (times_rate['stim_on'] + 2*times_rate['stim_dur'] + times_rate['delay'])/times_rate['fs'],
        'T': times_rate['T']/times_rate['fs'],
        'delay': times_rate['delay']/times_rate['fs'],
        'stim_dur': times_rate['stim_dur']/times_rate['fs'],
        }

    return times_rate, times_real, fs_dict

def get_times_dict(fs_wanted, condn_phrase, condn_num='', 
                   model_name='', all_models_dir='/home/nuttidalab/Documents/renee/all_DMS_models/'): #all_models_dir='/scratch/all_DMS_models/'
    '''
    Get the timing data dict for a specified fs
    fs_wanted is the sampling rate, can be 'rate', 'spk', 'ds', or a float
    '''
    if model_name == '':
        model_name = get_immediate_subdirs(all_models_dir)[0]

    # Get timing data
    times_rate, times_real, fs_dict = load_timing_data(model_name, condn_phrase, condn_num, all_models_dir)

    # Get the fs for the specified condition
    if isinstance(fs_wanted, str):
        fs = fs_dict[fs_wanted]
    else:
        fs = fs_wanted
    
    times_fs = {}
    for key in times_real.keys():
        times_fs[key] = times_real[key] * fs
    times_fs['fs'] = fs
    
    return times_fs, times_real, fs_dict


def get_model(model_name, all_models_dir='/home/nuttidalab/Documents/renee/all_DMS_models/'): #all_models_dir='/scratch/all_DMS_models/',):
    '''
    Get the model data for a given model
    '''
    # Get the model path
    model_path = f'{all_models_dir}/{model_name}.mat'
    mat_data = sio.loadmat(model_path)
    
    return mat_data


def get_celltype_label(model_name, condn_phrase='', condn_num='', remove_lowfr=False):
    '''
    Get the cell type labels (E or I) for a given model
    '''
    mat_data = get_model(model_name)

    exc = mat_data['exc']
    exc_ind = np.where(exc == 1)[0]
    inh = mat_data['inh']
    inh_ind = np.where(inh == 1)[0]
    
    if remove_lowfr:
        neuron_rmv_idxs = load_lowfr_idxs(model_name, condn_phrase, condn_num, threshold=2)
        results = rmv_lowfr_neurons(neuron_rmv_idxs, N=len(exc), spk_df=None, rates_data=None, lfp_data=None, 
                      connectivity_df=None, W=None, exc_ind=exc_ind, inh_ind = inh_ind)
        exc_ind = results['exc_ind']
        exc_id = results['exc_id']
        inh_ind = results['inh_ind']
        inh_id = results['inh_id']
        return exc_ind, inh_ind, exc_id, inh_id
    
    return exc_ind, inh_ind


def get_timescales(model_name):
    '''
    Get the timescales for a given model
    '''
    mat_data = get_model(model_name)
    mean_decay = mat_data['mean_decay'][0][0]
    taus_decay_ms = mat_data['taus_decay_ms'][0]
    nan_idx = np.isnan(taus_decay_ms)
    auto_N = mat_data['auto_N'][0]
    taus_decay_ms = taus_decay_ms[~nan_idx]
    auto_N = auto_N[~nan_idx]

    return mean_decay, taus_decay_ms, auto_N


def get_connectivity_df(model_name, condn_phrase, condn_num, remove_lowfr=False):
    '''
    Get the connectivity data for a given model
    '''
    mat_data = get_model(model_name)
    final_w = get_weights(model_name)
    exc_ind = np.where(mat_data['exc'] == 1)[0]
    
    # Initialize list for flattened rows
    flattened_rows = []

    # Iterate through all neuron connections
    n_neurons = final_w.shape[0]
    for presyn_id in range(n_neurons):
        for postsyn_id in range(n_neurons):
            weight = final_w[postsyn_id, presyn_id]
            if weight != 0:
                # Determine the cell type of presynaptic and postsynaptic neurons
                presyn_type = 'exc' if presyn_id in exc_ind else 'inh'
                postsyn_type = 'exc' if postsyn_id in exc_ind else 'inh'
                
                flattened_rows.append({
                    'presyn_id': presyn_id,
                    'postsyn_id': postsyn_id,
                    'weight': weight,
                    'presyn_type': presyn_type,
                    'postsyn_type': postsyn_type
                })

    # Convert the list of dictionaries to a DataFrame
    connectivity_df = pd.DataFrame(flattened_rows)
    
    if remove_lowfr:
        neuron_rmv_idxs = load_lowfr_idxs(model_name, condn_phrase, condn_num, threshold=2)
        results = rmv_lowfr_neurons(neuron_rmv_idxs, n_neurons, connectivity_df=connectivity_df)
        connectivity_df = results['connectivity_df']
    
    return connectivity_df


def get_weights(model_name, condn_phrase, condn_num, remove_lowfr=False,
                all_models_dir='/home/nuttidalab/Documents/renee/all_DMS_models/'): #all_models_dir='/scratch/all_DMS_models/',):
    '''
    Get the weights for a given model
    '''
    mat_data = get_model(model_name, all_models_dir)
    
    w = mat_data['w']
    m = mat_data['m']
    scaling_factor = mat_data['opt_scaling_factor'][0][0]
    final_w = np.matmul(w, m) / scaling_factor
    taus_gaus = mat_data['taus_gaus']
    taus = mat_data['taus'].flatten()
    taus_sig = (1/(1+np.exp(-taus_gaus))*(taus[1] - taus[0])) + taus[0]
    
    if remove_lowfr:
        neuron_rmv_idxs = load_lowfr_idxs(model_name, condn_phrase, condn_num, threshold=2)
        results = rmv_lowfr_neurons(neuron_rmv_idxs, len(w), W = final_w)
        final_w = results['W']
        
    
    return final_w


def get_models_by_perf(low_cutoff, high_cutoff, condn_phrase, condn_num='', 
                       all_models_dir='/home/nuttidalab/Documents/renee/all_DMS_models/'):
    
    model_dirs = get_immediate_subdirs(all_models_dir)
    model_perfs = []
    model_names = []
    for i, model_name in enumerate(model_dirs):
        _, trial_perfs = load_bhv_data(model_name, condn_phrase, condn_num)
        model_perf = np.mean(trial_perfs)
        if model_perf > low_cutoff and model_perf < high_cutoff:
            model_perfs.append(model_perf)
            model_names.append(model_name)
    print(f'Found {len(model_perfs)} models with performance between {low_cutoff} and {high_cutoff}')
    
    return model_names, model_perfs



def get_trialtype_colors():
    stims = np.array([[-1,-1], [-1,1], [1,-1], [1,1]])
    colors = ['#6E439A','#2B1644', '#236975','#49BEA3']
    return stims, colors