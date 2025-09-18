'''
UTIL FUNCTIONS FOR SINGLE NEURON ANALYSIS
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from scipy import stats
import pandas as pd
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.decomposition import PCA
# from umap import UMAP
from dPCA.dPCA import dPCA
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans
import pdb

import load_data as ld
import single_neuron_utils as sn



def plot_trajectory_pca(rates_data, trial_labels, settings, pca_obj=None,
                    cut_off = 50, ds=10, smooth=10, plot=1,
                    linestyle='-', alpha=0.9,
                    ax=None, title=None,):
    """
    Plot the trajectory of the population activity for a given model and condition
    
    rates_data: trials x neurons x time
    trial_labels: trials x 2
    """
    
    if ax is None and plot:
        # make a new 3d plot
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111, projection='3d')

    stims, colors = ld.get_trialtype_colors()
       
    
    # get avg firing rate for each trial type
    new_T = int(settings['T'] - cut_off)
    if new_T <= 0:
        raise ValueError("cut_off is too large, resulting in non-positive time length for trials.")
    trial_types, trial_idxs = np.unique(trial_labels, axis=0, return_inverse=True)
    mean_rates = np.zeros((len(trial_types), new_T, rates_data.shape[1])) # trial_types x time x neurons
    
    # get mean rates for each trial type
    for i, trial_type in enumerate(trial_types):
        trials_idx = (trial_idxs == i)
        trials_rate = rates_data[trials_idx, :, cut_off:] # trials x neurons x time

        mean_rates[i,:,:] = np.mean(trials_rate, axis=0).T # avg across trials, neurons x time
    
    # reshape and combine time axes for the dif trial types
    rates_reshape = np.transpose(mean_rates, (2,0,1)) # neurons x trials x time
    rates_reshape = rates_reshape.reshape(rates_reshape.shape[0], -1) # neurons x trials*time
    rates_reshape = rates_reshape.T # trials*time x neurons
        
    if pca_obj is None: # create a PCA object with 3 components if there wasn't one specified
        pca_obj = PCA(n_components=3)
        pca_obj.fit(rates_reshape)
    if not plot:
        return pca_obj
    rates_pca = pca_obj.transform(rates_reshape) # trials*time x 3
    varexp = pca_obj.explained_variance_ratio_ # 3 x 1
    print(f'PCA explained variance: {varexp}, total: {varexp.sum()}')
    rates_comp = rates_pca.reshape(len(trial_types), -1, 3) # trials x time x 3
    if smooth is not None:
        rates_comp = gaussian_filter1d(rates_comp, sigma=smooth, axis=1)  # smooth over time axis
    
    # Stimulus times in original timepoints, then downsampled
    stim1_on = int(settings['stim_on'] - cut_off)
    stim1_off = int(settings['stim_on'] + settings['stim_dur'] - cut_off)
    stim2_on = int(settings['stim_on'] + settings['stim_dur'] + settings['delay'] - cut_off)
    stim2_off = int(settings['stim_on'] + 2*settings['stim_dur'] + settings['delay'] - cut_off)

    # downsample
    if ds is not None:
        rates_comp = rates_comp[:, ::ds, :]  # downsample
        stim1_on = int(stim1_on // ds)  # adjust stim times for downsampling
        stim1_off = int(stim1_off // ds)
        stim2_on = int(stim2_on // ds)
        stim2_off = int(stim2_off // ds)

    # plot the trajectory in 3D
    for i, trial_type in enumerate(trial_types):
        color_idx = matching_stim_idx(stims, trial_type)
        ax.plot(rates_comp[i, :, 0], rates_comp[i, :, 1], rates_comp[i, :, 2], 
                color=colors[color_idx], label=str(trial_type), linestyle=linestyle, alpha=alpha)
        
        # plot over stim times with thicker lines
        ax.plot(rates_comp[i, stim1_on:stim1_off, 0],
                rates_comp[i, stim1_on:stim1_off, 1],
                rates_comp[i, stim1_on:stim1_off, 2],
                color=colors[color_idx], linewidth=5, linestyle=linestyle, alpha=alpha)
        ax.plot(rates_comp[i, stim2_on:stim2_off, 0],
                rates_comp[i, stim2_on:stim2_off, 1],
                rates_comp[i, stim2_on:stim2_off, 2],
                color=colors[color_idx], linewidth=5, linestyle=linestyle, alpha=alpha)
        
        # plot a star at the end of the trajectory
        ax.scatter(rates_comp[i, -1, 0], rates_comp[i, -1, 1], rates_comp[i, -1, 2],
                   color=colors[color_idx], s=100, marker='*', 
                   edgecolor='k', linewidth=1.5, alpha=alpha)
        # plot a circle at the start of the trajectory
        ax.scatter(rates_comp[i, 0, 0], rates_comp[i, 0, 1], rates_comp[i, 0, 2],
                   color=colors[color_idx], s=50, marker='o', 
                   edgecolor=colors[color_idx], linewidth=1.5, alpha=alpha)
        

    if title is not None:
        ax.set_title(f'{title}')#, {model_name[-6:]}, {condn_phrase} {condn_num}')
    else:
        ax.set_title(f'Avg model trajectory')
    ax.legend()
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3');
    
    return ax, pca_obj

def plot_trajectory_comps(settings, pcs, trial_labels, 
                          cut_off=50, ds=10, ax=None, title=None,
                          linestyle='-', alpha=0.9,):
    """ plot trajectory given components and trial labels """
    
    if ax is None:
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111, projection='3d')
    
    stims, colors = ld.get_trialtype_colors()
    
    
    if ds is not None:
        pcs = pcs[:, ::ds, :]  # downsample
        cut_off = int(cut_off / ds)  # adjust cut_off for downsampling
        times_ms = {k: int(v / ds) for k, v in times_ms.items()}  # adjust times for downsampling
        
    trial_types, trial_idxs = np.unique(trial_labels, axis=0, return_inverse=True)
    
    # plot the trajectory in 3D
    for i, trial_type in enumerate(trial_types):
        color_idx = matching_stim_idx(stims, trial_type)
        ax.plot(pcs[i, :, 0], pcs[i, :, 1], pcs[i, :, 2], 
                color=colors[color_idx], label=str(trial_type), linestyle=linestyle, alpha=alpha)
        
        # plot over stim times with thicker lines
        stim1_on = int(times_ms['stim1_on'] - cut_off)
        stim1_off = int(times_ms['stim1_off'] - cut_off)
        stim2_on = int(times_ms['stim2_on'] - cut_off)
        stim2_off = int(times_ms['stim2_off'] - cut_off)
        ax.plot(pcs[i, stim1_on:stim1_off, 0],
                pcs[i, stim1_on:stim1_off, 1],
                pcs[i, stim1_on:stim1_off, 2],
                color=colors[color_idx], linewidth=5, linestyle=linestyle, alpha=alpha)
        ax.plot(pcs[i, stim2_on:stim2_off, 0],
                pcs[i, stim2_on:stim2_off, 1],
                pcs[i, stim2_on:stim2_off, 2],
                color=colors[color_idx], linewidth=5, linestyle=linestyle, alpha=alpha)
        
        # plot a star at the end of the trajectory
        ax.scatter(pcs[i, -1, 0], pcs[i, -1, 1], pcs[i, -1, 2],
                   color=colors[color_idx], s=100, marker='*', 
                   edgecolor='k', linewidth=1.5, alpha=alpha)
        # plot a circle at the start of the trajectory
        ax.scatter(pcs[i, 0, 0], pcs[i, 0, 1], pcs[i, 0, 2],
                   color=colors[color_idx], s=50, marker='o', 
                   edgecolor=colors[color_idx], linewidth=1.5, alpha=alpha)
        

    if title is not None:
        ax.set_title(f'{title}')#, {model_name[-6:]}, {condn_phrase} {condn_num}')
    else:
        ax.set_title(f'Avg trajectory for model')
    ax.legend()
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3');


# NOT UPDATED YET
def plot_trajectory_dpca(model_name, condn_phrase, condn_num, 
                    dpca_obj = None, plot_type='t', ds=10,
                    neuron_ids=None, rates_data=None, trial_labels=None, cut_off = 50,
                    ax=None, title=None):
    """
    ***NOT UPDATED***
    Plot the trajectory of the population activity for a given model and condition
    plot_type can be 't' for time, 's' for stimulus, or 'st' for both.
    """
    # Load the data
    if rates_data is None:
        _, _, rates_data = ld.load_neural_data(model_name, condn_phrase, condn_num, load_rates=True)
    if trial_labels is None:
        trial_labels, _ = ld.load_bhv_data(model_name, condn_phrase, condn_num)
    
    if neuron_ids is not None: # trim the population activity to the specified neurons
        rates_data = rates_data[:, neuron_ids, :]

    if ax is None:
        # make a new 3d plot
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111, projection='3d')
        
    times_ms, times_real, fs_dict = ld.get_times_dict('ds', condn_phrase, condn_num, model_name=model_name) #ds is fs=1000, ms

    stims, colors = ld.get_trialtype_colors()
       
    
    # get avg firing rate for each trial type
    new_T = int(times_ms['T'] - cut_off)
    if new_T <= 0:
        raise ValueError("cut_off is too large, resulting in non-positive time length for trials.")
    trial_types, trial_idxs = np.unique(trial_labels, axis=0, return_inverse=True)
    mean_rates = np.zeros((len(trial_types), new_T, rates_data.shape[1])) # trial_types x time x neurons
    
    # get mean rates for each trial type
    for i, trial_type in enumerate(trial_types):
        trials_idx = (trial_idxs == i)
        trials_rate = rates_data[cut_off:, :, trials_idx] # time x neurons x trials

        mean_rates[i,:,:] = np.mean(trials_rate, axis=2) # avg across trials, time x neurons
    
    # reshape and combine time axes for the dif trial types
    rates_reshape = np.transpose(mean_rates, (2,0,1)) # neurons x trials x time
    
    if dpca_obj is None: # create a dPCA object with 3 components if there wasn't one specified
        dpca_obj = dPCA(labels='st', n_components=3)
        dpca_obj.fit(rates_reshape)
    rates_fit = dpca_obj.transform(rates_reshape) # 3 x trials x time
    # print(rates_fit['s'].shape, rates_fit['t'].shape, rates_fit['st'].shape)
    rates_comp  = rates_fit[plot_type].transpose((1,2,0)) # trials x time x 3
    # gaussian smooth
    rates_comp = gaussian_filter1d(rates_comp, sigma=50, axis=1)  # smooth over time axis
    # print(rates_comp.shape)
    
    # downsample and adjust all times
    if ds is not None:
        # downsample, preserving first time point
        rates_comp = rates_comp[:, ::ds, :]  # downsample
        cut_off = int(cut_off / ds)  # adjust cut_off for downsampling
        times_ms = {k: int(v / ds) for k, v in times_ms.items()}  # adjust times for downsampling
    
    # plot the trajectory in 3D
    for i, trial_type in enumerate(trial_types):
        color_idx = matching_stim_idx(stims, trial_type)
        
        ax.plot(rates_comp[i, :, 0], rates_comp[i, :, 1], rates_comp[i, :, 2], 
                color=colors[color_idx], label=str(trial_type), alpha=.9)
        
        # plot over stim times with thicker lines
        stim1_on = int(times_ms['stim1_on'] - cut_off)
        stim1_off = int(times_ms['stim1_off'] - cut_off)
        stim2_on = int(times_ms['stim2_on'] - cut_off)
        stim2_off = int(times_ms['stim2_off'] - cut_off)
        ax.plot(rates_comp[i, stim1_on:stim1_off, 0],
                rates_comp[i, stim1_on:stim1_off, 1],
                rates_comp[i, stim1_on:stim1_off, 2],
                color=colors[color_idx], linewidth=5, alpha=.9)
        ax.plot(rates_comp[i, stim2_on:stim2_off, 0],
                rates_comp[i, stim2_on:stim2_off, 1],
                rates_comp[i, stim2_on:stim2_off, 2],
                color=colors[color_idx], linewidth=5, alpha=.9)
        
        # plot a star at the end of the trajectory
        ax.scatter(rates_comp[i, -1, 0], rates_comp[i, -1, 1], rates_comp[i, -1, 2],
                   color=colors[color_idx], s=100, marker='*', 
                   edgecolor='k', linewidth=1.5)
        # plot a circle at the start of the trajectory
        ax.scatter(rates_comp[i, 0, 0], rates_comp[i, 0, 1], rates_comp[i, 0, 2],
                   color=colors[color_idx], s=50, marker='o', 
                   edgecolor=colors[color_idx], linewidth=1.5)
        

    if title is not None:
        ax.set_title(f'{title}')#, {model_name[-6:]}, {condn_phrase} {condn_num}')
    else:
        ax.set_title(f'Avg trajectory for {model_name[-6:]}, {condn_phrase} {condn_num}')
    ax.legend()
    
    ax.set_xlabel('dPC1')
    ax.set_ylabel('dPC2')
    ax.set_zlabel('dPC3');
    
    return ax, dpca_obj




def matching_stim_idx(trial_labels, stim_label):
    """
    Get the indices of trials that match the given stimulus label.
    trial_labels is a 2D array where each row is a trial and each column is a label.
    stim_label is a 1D array representing the stimulus label to match.
    Returns the indices of trials that match the stimulus label.
    """
    stim_idx = np.where(np.all(trial_labels == stim_label, axis=1))[0][0]
    return stim_idx