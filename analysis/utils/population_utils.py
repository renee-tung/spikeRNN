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
from umap import UMAP
from sklearn.cluster import KMeans
import pdb

import load_data as ld
import single_neuron_utils as sn
from bootstrap_method import *



def plot_trajectory(model_name, condn_phrase, condn_num, pca_obj = None,
                    neuron_ids=None, rates_data=None, cut_off = 50,
                    ax=None, title=None):
    """
    Plot the trajectory of the population activity for a given model and condition
    """
    # Load the data
    if rates_data is None:
        _, _, rates_data = ld.load_neural_data(model_name, condn_phrase, condn_num, load_rates=True)
    
    if neuron_ids is not None: # trim the population activity to the specified neurons
        rates_data = rates_data[:, neuron_ids, :]

    if ax is None:
        # make a new 3d plot
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111, projection='3d')
        
    trial_labels, trial_perfs = ld.load_bhv_data(model_name, condn_phrase, condn_num)
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
    rates_reshape = rates_reshape.reshape(rates_reshape.shape[0], -1) # neurons x trials*time
    rates_reshape = rates_reshape.T # trials*time x neurons
        
    # PCA the mean trial frs together
    if pca_obj is None: # create a PCA object with 3 components if there wasn't one specified
        pca_obj = PCA(n_components=3)
        pca_obj.fit(rates_reshape)
    rates_pca = pca_obj.transform(rates_reshape) # trials*time x 3
    varexp = pca_obj.explained_variance_ratio_ # 3 x 1
    print(f'PCA explained variance: {varexp}, total: {varexp.sum()}')
    rates_pca = rates_pca.reshape(len(trial_types), -1, 3) # trials x time x 3
    
    # plot the trajectory in 3D
    for i, trial_type in enumerate(trial_types):
        color_idx = matching_stim_idx(stims, trial_type)
        ax.plot(rates_pca[i, :, 0], rates_pca[i, :, 1], rates_pca[i, :, 2], 
                color=colors[color_idx], label=str(trial_type), alpha=.9)
        
        # plot over stim times with thicker lines
        stim1_on = int(times_ms['stim1_on'] - cut_off)
        stim1_off = int(times_ms['stim1_off'] - cut_off)
        stim2_on = int(times_ms['stim2_on'] - cut_off)
        stim2_off = int(times_ms['stim2_off'] - cut_off)
        ax.plot(rates_pca[i, stim1_on:stim1_off, 0],
                rates_pca[i, stim1_on:stim1_off, 1],
                rates_pca[i, stim1_on:stim1_off, 2],
                color=colors[color_idx], linewidth=5, alpha=.9)
        ax.plot(rates_pca[i, stim2_on:stim2_off, 0],
                rates_pca[i, stim2_on:stim2_off, 1],
                rates_pca[i, stim2_on:stim2_off, 2],
                color=colors[color_idx], linewidth=5, alpha=.9)
        
        # plot a star at the end of the trajectory
        ax.scatter(rates_pca[i, -1, 0], rates_pca[i, -1, 1], rates_pca[i, -1, 2],
                   color=colors[color_idx], s=100, marker='*', 
                   edgecolor='k', linewidth=1.5)
        # plot a circle at the start of the trajectory
        ax.scatter(rates_pca[i, 0, 0], rates_pca[i, 0, 1], rates_pca[i, 0, 2],
                   color=colors[color_idx], s=50, marker='o', 
                   edgecolor=colors[color_idx], linewidth=1.5)
        

    if title is not None:
        ax.set_title(f'{title}')#, {model_name[-6:]}, {condn_phrase} {condn_num}')
    else:
        ax.set_title(f'Avg trajectory for {model_name[-6:]}, {condn_phrase} {condn_num}')
    ax.legend()
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3');
    
    return ax, pca_obj



def plot_trajectory_animation(model_name, condn_phrase, condn_num, pca_obj=None,
                    neuron_ids=None, rates_data=None, cut_off=50,
                    ax=None, title=None, save_path=None):
    """
    Plot or animate the trajectory of population activity in PCA space.
    If `as_animation` is True, animates instead of static plot.
    """

    # Load and preprocess data
    if rates_data is None:
        _, _, rates_data = ld.load_neural_data(model_name, condn_phrase, condn_num, load_rates=True)
    if neuron_ids is not None:
        rates_data = rates_data[:, neuron_ids, :]

    trial_labels, trial_perfs = ld.load_bhv_data(model_name, condn_phrase, condn_num)
    times_ms, times_real, fs_dict = ld.get_times_dict('ds', condn_phrase, condn_num, model_name=model_name)
    stims, colors = ld.get_trialtype_colors()

    new_T = int(times_ms['T'] - cut_off)
    if new_T <= 0:
        raise ValueError("cut_off is too large.")

    trial_types, trial_idxs = np.unique(trial_labels, axis=0, return_inverse=True)
    mean_rates = np.zeros((len(trial_types), new_T, rates_data.shape[1]))

    for i, trial_type in enumerate(trial_types):
        trials_idx = (trial_idxs == i)
        trials_rate = rates_data[cut_off:, :, trials_idx]
        mean_rates[i,:,:] = np.mean(trials_rate, axis=2)

    rates_reshape = np.transpose(mean_rates, (2,0,1)).reshape(rates_data.shape[1], -1).T

    if pca_obj is None:
        pca_obj = PCA(n_components=3)
        pca_obj.fit(rates_reshape)
    rates_pca = pca_obj.transform(rates_reshape)
    varexp = pca_obj.explained_variance_ratio_
    print(f'PCA explained variance: {varexp}, total: {varexp.sum()}')

    rates_pca = rates_pca.reshape(len(trial_types), new_T, 3)
    
    # downsample in time
    rates_pca = rates_pca[:, ::10, :]  # downsample by factor of 10

    # If animating, skip static plot and go to animation block
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')

    lines = []
    for i in range(rates_pca.shape[0]):
        line, = ax.plot([], [], [], color=colors[i], label=str(trial_types[i]))
        lines.append(line)

    ax.set_xlim(rates_pca[..., 0].min(), rates_pca[..., 0].max())
    ax.set_ylim(rates_pca[..., 1].min(), rates_pca[..., 1].max())
    ax.set_zlim(rates_pca[..., 2].min(), rates_pca[..., 2].max())
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title(title or f'Animated Trajectory: {model_name[-6:]}, {condn_phrase} {condn_num}')
    ax.legend()

    def update(frame):
        for i in range(rates_pca.shape[0]):
            lines[i].set_data(rates_pca[i, :frame, 0], rates_pca[i, :frame, 1])
            lines[i].set_3d_properties(rates_pca[i, :frame, 2])
        return lines

    ani = FuncAnimation(fig, update, frames=new_T, interval=50, blit=False)

    if save_path is not None:
        ani.save(save_path, writer='ffmpeg', fps=20)
        print(f'Animation saved to {save_path}')
    else:
        return HTML(ani.to_jshtml())



def matching_stim_idx(trial_labels, stim_label):
    """
    Get the indices of trials that match the given stimulus label.
    trial_labels is a 2D array where each row is a trial and each column is a label.
    stim_label is a 1D array representing the stimulus label to match.
    Returns the indices of trials that match the stimulus label.
    """
    stim_idx = np.where(np.all(trial_labels == stim_label, axis=1))[0][0]
    return stim_idx