'''
UTIL FUNCTIONS FOR CONNECTIVITY ANALYSIS
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import stats
import pandas as pd

import load_data as ld
# from bootstrap_method import *


'''
VISUALIZE CONNECTIVITY
'''

def plot_num_synaptic_connections(connectivity_df, neuron_inds, synapse_type, ax=None, title=None):
    """
    Plot the number pre or post synaptic connections of a group of neurons.
    
    Parameters:
    connectivity_df : pandas DataFrame
        The connectivity dataframe.
    neuron_inds : list
        The indices of the neurons to plot.
    ax : matplotlib axis
        The axis to plot on.
    title : str
        The title of the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    
    for n_neuron, neuron_idx in enumerate(neuron_inds):
        if synapse_type == 'pre':

            # get the number of presynaptic connections for this neuron
            presyn_df = connectivity_df[connectivity_df['postsyn_id'] == neuron_idx]
            # get the number of presynaptic connections for this neuron
            n_inh = np.sum(presyn_df['presyn_type'] == 'inh')
            n_exc = np.sum(presyn_df['presyn_type'] == 'exc')
            num_presyn = len(connectivity_df[connectivity_df['postsyn_id'] == neuron_idx])
            if n_inh + n_exc != num_presyn:
                print(f'Neuron {neuron_idx} has {num_presyn} presynaptic connections, but {n_inh} inh and {n_exc} exc')
        elif synapse_type == 'post':
            # get the number of postsynaptic connections for this neuron
            postsyn_df = connectivity_df[connectivity_df['presyn_id'] == neuron_idx]
            # get the number of postsynaptic connections for this neuron
            n_inh = np.sum(postsyn_df['postsyn_type'] == 'inh')
            n_exc = np.sum(postsyn_df['postsyn_type'] == 'exc')
            num_postsyn = len(connectivity_df[connectivity_df['presyn_id'] == neuron_idx])
            if n_inh + n_exc != num_postsyn:
                print(f'Neuron {neuron_idx} has {num_postsyn} postsynaptic connections, but {n_inh} inh and {n_exc} exc')
        else:
            raise ValueError('synapse_type must be "pre" or "post"')
        
        # plot the number of presynaptic connections for this neuron
        if n_inh > 0:
            ax.bar(n_neuron, n_inh, color='blue', alpha=0.5)
        if n_exc > 0:
            ax.bar(n_neuron, n_exc, bottom=n_inh, color='red', alpha=0.5)

    # set the x ticks to be the neuron indices
    ax.set_xticks(np.arange(len(neuron_inds)))
    ax.set_xticklabels(neuron_inds, rotation=90)
    # set the y label
    ax.set_ylabel(f'Number of {synapse_type}synaptic connections')
    
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f'Number of {synapse_type}synaptic Connections')



'''
VISUALIZE MATRIX
'''

def plot_connectivity_matrix(w, ax = None, title=None, cmap='bwr', vmin=-0.1, vmax=0.1):
    """
    Plot a connectivity matrix with a colorbar and title.
    
    Parameters:
    ax : matplotlib axis
        The axis to plot on.
    w : numpy array
        The connectivity matrix to plot.
    title : str
        The title of the plot.
    cmap : str
        The colormap to use.
    vmin : float
        The minimum value for the colormap.
    vmax : float
        The maximum value for the colormap.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(w, cmap=cmap, vmin=vmin, vmax=vmax)
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('W[a,b], b->a')
    plt.xlabel('b, outgoing weights, presynaptic neuron')
    plt.ylabel('a, incoming weights, postsynaptic neuron')
    plt.colorbar(im, ax=ax)


def plot_connectivity_matrix_by_type(w, exc_ind, inh_ind, title=None, cmap='bwr', vmin=-0.1, vmax=0.1):
    """
    Plot a connectivity matrix with a colorbar and title, separating excitatory and inhibitory neurons.
    
    Parameters:
    ax : matplotlib axis
        The axis to plot on.
    w : numpy array
        The connectivity matrix to plot.
    exc_ind : numpy array
        Indices of excitatory neurons.
    inh_ind : numpy array
        Indices of inhibitory neurons.

    """
    
    # Create a mask for excitatory and inhibitory neurons
    ee_mask = np.ix_(exc_ind, exc_ind)
    ie_mask = np.ix_(exc_ind, inh_ind)
    ei_mask = np.ix_(inh_ind, exc_ind)
    ii_mask = np.ix_(inh_ind, inh_ind)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0, 0].imshow(w[ee_mask], vmin=-0.1, vmax=0.1, cmap='bwr', aspect='auto')
    axs[0, 0].set_title('Excitatory to Excitatory')
    axs[0, 0].set_xlabel('Excitatory Neurons')
    axs[0, 0].set_ylabel('Excitatory Neurons')
    axs[0, 1].imshow(w[ei_mask], vmin=-0.1, vmax=0.1, cmap='bwr', aspect='auto')
    axs[0, 1].set_title('Excitatory to Inhibitory')
    axs[0, 1].set_xlabel('Excitatory Neurons')
    axs[0, 1].set_ylabel('Inhibitory Neurons')
    axs[1, 0].imshow(w[ie_mask], vmin=-0.1, vmax=0.1, cmap='bwr', aspect='auto')
    axs[1, 0].set_title('Inhibitory to Excitatory')
    axs[1, 0].set_xlabel('Inhibitory Neurons')
    axs[1, 0].set_ylabel('Excitatory Neurons')
    axs[1, 1].imshow(w[ii_mask], vmin=-0.1, vmax=0.1, cmap='bwr', aspect='auto')
    axs[1, 1].set_title('Inhibitory to Inhibitory')
    axs[1, 1].set_xlabel('Inhibitory Neurons')
    axs[1, 1].set_ylabel('Inhibitory Neurons')
    plt.colorbar(axs[0, 0].imshow(w[ee_mask], vmin=-0.1, vmax=0.1, cmap='bwr'), ax=axs, orientation='vertical')
    # plt.tight_layout()
    plt.show()
    
    if title is not None:
        plt.title(title)