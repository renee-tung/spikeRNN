import pdb
import sys
import argparse
import numpy as np
import os
import pickle
import scipy.io as si
from sklearn.decomposition import PCA
import h5py
import matplotlib.pyplot as plt
rootpath = os.path.join(os.getcwd(), '..')
sys.path.append(rootpath)
from load_RNN_model import load_RNN_model
PATH_TO_FIXED_POINT_FINDER = '/home/nuttidalab/Documents/spikeRNN/analysis_code/utils/fixed-point-finder/'
sys.path.insert(0, PATH_TO_FIXED_POINT_FINDER)
from FixedPointFinderTorch import FixedPointFinderTorch as FixedPointFinder
from plot_utils import plot_fps # in fixedpointfinder folder

'''
Borrowed code from Tomas
'''


def get_period_pca(xx_trials,period,nComponents):
    # Which times the trajectory points will be drawn from
    
	# if period == 'instruction':
    #     times = np.arange(29,79) # Instruction
    # elif period == 'stim1':
    #     times = np.arange(79,129) # Stim1
    # elif period == 'stim2':
    #     times = np.arange(179,229) # Stim2 

	stim_on = 200
	stim_dur = 50
	delay=10
	T = 500    
	
	if period == 'stim1':
		times = np.arange(stim_on, stim_on+stim_dur) # Stim1
	elif period == 'delay':
		times = np.arange(stim_on+stim_dur,stim_on+stim_dur+delay) # Maintenance
	elif period == 'stim2':
		times = np.arange(stim_on+stim_dur+delay,stim_on+2*stim_dur+delay) # Stim2   
		
	# Getting period time PCA
	period_avg = np.mean(np.squeeze(xx_trials[:,times,:]),axis=1)
	pca = PCA(n_components=nComponents)
	pca.fit(period_avg)
	pca_period = pca
	return pca_period


def main():
	# Step 1: Take a pre-trained network
	stim_on = 200
	stim_dur = 50
	delay=50
	T = 500  

	# modality_type = 'bad_models'
	models_type = 'good_models'

	models_dir = '/scratch/spikeRNN/models/DMS_OSF/' # dir where all models are located
	results_dir = f'{models_dir}{models_type}/' # dir where general results are saved
	model_results_dir = f'{models_dir}{RNN_model_file[:-4]}/' # dir where results for this model are saved

	model_list_path = f'{results_dir}{models_type}_list.mat' # list (saved from matlab) of models of interest
	model_list_cell = si.loadmat(model_list_path)['stable_mods'][0] 
	model_list = [model_list_cell[i][0] for i in range(len(model_list_cell))]
	RNN_model_file = model_list[0] # just grab the first model

	# if modality_type == 'good_models':
	# 	base_folder = 'scratch/spikeRNN/models/DMS_OSF/good_models/'
	# 	RNN_model_file = 'Task_instr_N_1000_Taus_4.0_25.0_Act_sigmoid_2023_04_24_013958'
		
	rnn_path = os.path.join(models_dir, RNN_model_file)
	model = load_RNN_model(rnn_path)
	# Getting RNN activity
	synX_filename = r'synX.npy' # 'synX.mat' for RNN activity 
	synX_path = os.path.join(model_results_dir,synX_filename)
	xx_trials = np.load(synX_path) # Get matrix of simulated nTrials x nTimes x nNeurons trajectories

	# Getting trial information 
	trial_filename = r'trial_stim_labels.npy'
	trial_path = os.path.join(model_results_dir,trial_filename)
	trial_stim_labels = np.load(trial_path)

	# STEP 2: Find, analyze, and visualize the fixed points of the trained RNN
	#find_fixed_points(model, valid_predictions)
	fpf_hps = {
			'max_iters': 30000,
			'lr_init': 0.1,
			'outlier_distance_scale': 10.0,			
			'tol_unique':10, #75
			'verbose': True, 
			'super_verbose': True}
	
	# task_types = ['null','pro','anti']
	task_types = [1, -1] # for the first stimulus

	for task_type in task_types:
		# Getting initial conditions for fixed point optimization at the end of the instruction period
		
		stim1_times = np.arange(stim_on,stim_on+stim_dur)
		delay_times = np.arange(stim_on+stim_dur,stim_on+stim_dur+delay) # Maintenance
		
		# Selecting trials from this particular trial type 
		xx_trials_stim1 = xx_trials[trial_stim_labels[:,0] == task_type,:,:][:,stim1_times,:]
		xx_trials_delay = xx_trials[trial_stim_labels[:,0] == task_type,:,:][:,delay_times,:]
		
		# Selecting a random number of initial conditions from eligible points
		n_initial = 200
		initial_states = np.zeros((n_initial,xx_trials_delay.shape[2])) # n_initial x nNeurons
		for iI in np.arange(n_initial):
			rand_trial = np.random.randint(xx_trials_delay.shape[0]) 		
			rand_time = np.random.randint(xx_trials_delay.shape[1])
			initial_states[iI,:] = xx_trials_delay[rand_trial,-1,:]
			
		# Initializing input vector u
		inputs = np.zeros((1, 2))

		# Choosing input vector u for each trial type/modality type		
		inputs[0, 0] = 0 # no input during maintenance
				
		# Defining and running fixed point optimization
		fpf = FixedPointFinder(model, **fpf_hps)
		unique_fps,all_fps = fpf.find_fixed_points(initial_states, inputs)    
		fp_dict = {'xstar': unique_fps.xstar, 'is_stable':unique_fps.is_stable, \
				'J_xstar':unique_fps.J_xstar, 'eigval_J_xstar': unique_fps.eigval_J_xstar, \
					'eigvec_J_xstar':unique_fps.eigvec_J_xstar}
		with open(model_results_dir+'fixed_points_'+str(task_type)+'.pk', 'wb') as handle:
			#pickle.dump(unique_fps.xstar, handle, protocol=pickle.HIGHEST_PROTOCOL)		
			pickle.dump(fp_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
	main()