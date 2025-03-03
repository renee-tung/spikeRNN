'''
Script to calculate and save synaptic X (synX) and fixed points for all good and bad models
'''


import os, scipy.io 
import numpy as np 
# %matplotlib ipympl
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import axes3d
import sys
import pickle as pk
from sklearn.decomposition import PCA

sys.path.append('/home/nuttidalab/Documents/spikeRNN/rate/')
# import model
from model import generate_input_stim_xor, eval_tf

sys.path.append('/home/nuttidalab/Documents/spikeRNN/analysis_code/utils/')
import trajectory_utils as tu
import importlib

sys.path.append('/home/nuttidalab/Documents/spikeRNN/analysis_code/trajectory/fixed_pt/')
import load_RNN_model as lrm

sys.path.append('/home/nuttidalab/Documents/spikeRNN/analysis_code/utils/fixed-point-finder')
from FixedPointFinderTorch import FixedPointFinderTorch as FixedPointFinder
from plot_utils import plot_fps # in fixedpointfinder folder

models_dir = '/scratch/spikeRNN/models/DMS_OSF/'

models_types = ['good_models','bad_models']

settings = {
        'T': 500, # trial duration (in steps)
        'stim_on': 200, # input stim onset (in steps)
        'stim_dur': 50, # input stim duration (in steps)
        'delay': 150, # delay b/w the two stimuli (in steps)
        'DeltaT': 1, # sampling rate
        'taus': 20, # decay time-constants (in steps)
        'task': 'xor', # task name
        }

for models_type in models_types:
    results_dir = f'{models_dir}{models_type}/' # dir where results are saved

    model_list_path = f'{results_dir}{models_type}_list.mat' # list (saved from matlab) of models of interest
    model_list_cell = scipy.io.loadmat(model_list_path)['stable_mods'][0] 
    model_list = [model_list_cell[i][0] for i in range(len(model_list_cell))]

    for RNN_model_file in model_list:
        model_results_dir = f'{models_dir}{RNN_model_file[:-4]}/'

        print(f'Loading model: {RNN_model_file}')
        rnn_path = os.path.join(models_dir, RNN_model_file) 
        model = lrm.load_RNN_model(rnn_path)

        if not os.path.exists(f'{model_results_dir}delay{settings["delay"]}_synX.npy'):
            print('Generating synX')
            xx_trials, trial_stim_labels = tu.generate_synX(rnn_path, settings, 
                                                            save=1, model_results_dir = model_results_dir)
        else:
            print('synX already calculated')
            # print('Loading synX')
            # xx_trials = np.load(f'{model_results_dir}delay{settings["delay"]}_synX.npy')
            # trial_stim_labels = np.load(f'{model_results_dir}trial_stim_labels.npy')

        # Get the fixed points
        if os.path.exists(f'{model_results_dir}delay{settings["delay"]}_fixed_pt_results.pkl'):
            print('fixed point optimization results already calculated')
            # print('loading fixed point optimization results')
            # fixed_pt_results = pk.load(open(f'{model_results_dir}delay{settings["delay"]}_fixed_pt_results.pkl', 'rb'))
        else:
            print('Running fixed point optimization')

            _ = tu.calculate_fixed_pts(RNN_model_file = RNN_model_file, xx_trials = xx_trials,
                                                    trial_stim_labels = trial_stim_labels, settings = settings,
                                                    save=1, models_dir = models_dir)
            # fixed_pt_results = pk.load(open(f'{model_results_dir}fixed_pt_results.pkl', 'rb'))
