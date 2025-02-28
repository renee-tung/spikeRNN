import numpy as np
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
import vis_utils as vu
import trajectory_utils as tu
import importlib

sys.path.append('/home/nuttidalab/Documents/spikeRNN/analysis_code/trajectory/fixed_pt/')
import load_RNN_model as lrm

sys.path.append('/home/nuttidalab/Documents/spikeRNN/analysis_code/utils/fixed-point-finder')
from FixedPointFinderTorch import FixedPointFinderTorch as FixedPointFinder
from plot_utils import plot_fps # in fixedpointfinder folder



def generate_xor_type(T, stim_on, stim_dur, delay, stim1=1, stim2=-1):
     '''
     Generate an xor trial input u, where we specify which stims to use
     '''
     
     u = np.zeros((2, T))
     
     u[0, stim_on:stim_on+stim_dur] = stim1
     u[1, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay] = stim2

     if stim1 == stim2:
          label = 'same'
     else: 
          label = 'diff'
     
     return u, label


def generate_synX(model_dir, settings, 
                  stim1s = [-1, 1], stim2s = [-1, 1],
                  n_trials = 1000, save=0, model_results_dir = []):
    '''
    
    '''
     
    n_trials_per_condition = int(n_trials/(len(stim1s)*len(stim2s)))
    
    synX = np.zeros((n_trials, settings['T'], 200)) # nTrials x nTimes x nNeurons
    trial_stim_labels = np.zeros((n_trials,2))
    for i, stim1 in enumerate(stim1s):
        for j, stim2 in enumerate(stim2s):
            for k in range(n_trials_per_condition):
                    trial_idx = (i*len(stim2s)+j)*n_trials_per_condition + k

                    u, label = generate_xor_type(settings['T'], settings['stim_on'], 
                                                 settings['stim_dur'], settings['delay'], stim1, stim2)
                    x, r, o, _ = eval_tf(model_dir=model_dir, settings=settings, u=u)

                    synX[trial_idx,:,:] = x.T
                    trial_stim_labels[trial_idx,0] = stim1
                    trial_stim_labels[trial_idx,1] = stim2

    if save:
        np.save(f'{model_results_dir}delay{settings["delay"]}_synX.npy', synX)
        np.save(f'{model_results_dir}trial_stim_labels.npy', trial_stim_labels)

    return synX, trial_stim_labels


def calculate_fixed_pts(RNN_model_file, xx_trials, trial_stim_labels, settings, 
                        save=0, models_dir = '/scratch/spikeRNN/models/DMS_OSF/'):
    '''
    '''

    model_results_dir = f'{models_dir}{RNN_model_file[:-4]}/'

    print(f'Loading model: {RNN_model_file}')
    rnn_path = os.path.join(models_dir, RNN_model_file) 
    model = lrm.load_RNN_model(rnn_path)


    fixed_pt_results = {}

    task_types = [1, -1] # for the first stimulus
    time_fp_labels = ['start','middle','end'] # relative to the delay period
    n_initial = 200

    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']
    delay=settings['delay']
    T = settings['T']

    fpf_hps = {
                'max_iters': 30000,
                'lr_init': 0.1,
                'outlier_distance_scale': 10.0,			
                'tol_unique':10, #75
                'verbose': True, 
                'super_verbose': True}

    for task_type in task_types:
        for time_fp_label in time_fp_labels:
            print(f'Finding fixed points for task type {task_type} and {time_fp_label} of delay period')
            stim1_times = np.arange(stim_on,stim_on+stim_dur)
            delay_times = np.arange(stim_on+stim_dur,stim_on+stim_dur+delay)
            xx_trials_stim1 = xx_trials[trial_stim_labels[:,0] == task_type,:,:][:,stim1_times,:]
            xx_trials_delay = xx_trials[trial_stim_labels[:,0] == task_type,:,:][:,delay_times,:]

            initial_states = np.zeros((n_initial,xx_trials_delay.shape[2])) # n_initial x nNeurons
            for iI in np.arange(n_initial):
                rand_trial = np.random.randint(xx_trials_delay.shape[0])
                if time_fp_label == 'start':
                    initial_states[iI,:] = xx_trials_stim1[rand_trial,-1,:] # last time point of the stimulus
                elif time_fp_label == 'middle':
                    mid_delay = int((delay_times[-1] - delay_times[0])/2)
                    initial_states[iI,:] = xx_trials_delay[rand_trial,mid_delay,:] # middle of delay period
                elif time_fp_label == 'end':
                    initial_states[iI,:] = xx_trials_delay[rand_trial,-2,:] # end of delay period
            
            # Choosing input vector u for each trial type/modality type
            inputs = np.zeros((1, 2))
            inputs[0, 0] = 0 # no stim for maintenance period

            # Defining and running fixed point optimization
            fpf = FixedPointFinder(model, **fpf_hps)
            unique_fps,all_fps = fpf.find_fixed_points(initial_states, inputs)    
            # fp_dict = {'xstar': unique_fps.xstar, 'is_stable':unique_fps.is_stable, \
            #         'J_xstar':unique_fps.J_xstar, 'eigval_J_xstar': unique_fps.eigval_J_xstar, \
            #             'eigvec_J_xstar':unique_fps.eigvec_J_xstar}
            fixed_pt_results[(task_type, time_fp_label)] = {
                'xstar': unique_fps.xstar,
                'is_stable': unique_fps.is_stable,
                'J_xstar': unique_fps.J_xstar,
                'eigval_J_xstar': unique_fps.eigval_J_xstar,
                'eigvec_J_xstar': unique_fps.eigvec_J_xstar
            }

    if save:
        fixed_pt_results = pk.dump(fixed_pt_results, open(f'{model_results_dir}fixed_pt_results.pkl', 'wb'))

    return fixed_pt_results


def plot_fixed_pts(fixed_pt_results, model_results_dir, settings,
                   xx_trials, trial_stim_labels, plot_examples=0,
                   skip_start = 0, skip_mid = 0, skip_end = 0):
    '''
    '''
    task_types = [1, -1] # for the first stimulus
    time_fp_labels = ['start','middle','end'] # relative to the delay period

    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']
    delay=settings['delay']
    T = settings['T']

    xx_trials_thrudelay = xx_trials[:,:stim_on+stim_dur+delay,:] # nTrials x nTimes x nNeurons

    n_neurons = xx_trials_thrudelay.shape[2]
    n_times = xx_trials_thrudelay.shape[1]

    xx_concatenated = xx_trials_thrudelay.reshape(-1,n_neurons)
    n_xx = xx_concatenated.shape[0]

    # concatenate the fixed points to the trials
    for task_type in task_types:
        for time_fp_label in time_fp_labels:
            fp_dict = fixed_pt_results[(task_type, time_fp_label)]
            xx_concatenated = np.concatenate((xx_concatenated, fp_dict['xstar']),axis=0)

    n_fixed_pts = xx_concatenated.shape[0] - n_xx

    # do pca
    pca = PCA(n_components=3)
    pca.fit(xx_concatenated)
    xx_pca = pca.transform(xx_concatenated)

    # split up the pca results back into trials
    xx_pca_split = xx_pca[:-n_fixed_pts,:]
    xx_pca_split = xx_pca_split.reshape(-1,n_times,3)

    xx_pca_pos_mean = np.mean(xx_pca_split[trial_stim_labels[:,0] == 1],axis=0)
    xx_pca_neg_mean = np.mean(xx_pca_split[trial_stim_labels[:,0] == -1],axis=0)

    xx_pca_pos = xx_pca_split[trial_stim_labels[:,0] == 1]
    xx_pca_neg = xx_pca_split[trial_stim_labels[:,0] == -1]

    fixedpt_pca = xx_pca[-n_fixed_pts:,:]

    # plot the pca results
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')

    # first plot mean of the trial types
    ax.plot(xx_pca_pos_mean[:,0],xx_pca_pos_mean[:,1],xx_pca_pos_mean[:,2],'b', label='+1')
    ax.plot(xx_pca_neg_mean[:,0],xx_pca_neg_mean[:,1],xx_pca_neg_mean[:,2],'g', label= '-1')

    if plot_examples:
        n_samples = 5
        pos_samp_idx = np.random.randint(xx_pca_pos.shape[0],size=n_samples)
        ax.plot(xx_pca_pos[pos_samp_idx,:,0].T,
                xx_pca_pos[pos_samp_idx,:,1].T,
                xx_pca_pos[pos_samp_idx,:,2].T,'b', alpha=0.3)
        neg_samp_idx = np.random.randint(xx_pca_neg.shape[0],size=n_samples)
        ax.plot(xx_pca_neg[neg_samp_idx,:,0].T,
                xx_pca_neg[neg_samp_idx,:,1].T,
                xx_pca_neg[neg_samp_idx,:,2].T,'g', alpha=0.3)
        
    shapes = ['o','x']
    label = ['unstable','stable']
    colors = ['b','g']
    # alphas = [1, 0.75, 0.5]
    sizes = [100, 50, 15]

    fixed_pt_count = 0
    for i, task_type in enumerate(task_types): # +1 or -1
        color = colors[i]

        for j, time_fp_label in enumerate(time_fp_labels): # start, middle, end
            # alpha = alphas[j]
            fp_dict = fixed_pt_results[(task_type, time_fp_label)]
            n_fp = len(fp_dict['xstar'])
            fp_pca = fixedpt_pca[fixed_pt_count:fixed_pt_count+n_fp,:]
            fixed_pt_count += n_fp
            is_stability = fp_dict['is_stable']

            if skip_start and j == 0:
                continue
            if skip_mid and j == 1:
                continue
            if skip_end and j == 2:
                continue

            print(f'{task_type} {time_fp_label} fp count: {n_fp}, stability: {is_stability}')

            if n_fp > 0:
                for k in range(n_fp):
                    # for fixed points types (stable/unstable) that haven't been seen before, plot label
                    if (k == 0) or (is_stability[k] != is_stability[:k]).all():
                        ax.scatter(fp_pca[k,0],fp_pca[k,1],fp_pca[k,2],
                                color=color,marker=shapes[is_stability[k]], 
                                label=f'{task_type} {label[is_stability[k]]} {time_fp_label} fp', s=sizes[j])
                    else:
                        ax.scatter(fp_pca[k,0],fp_pca[k,1],fp_pca[k,2],
                                color=color,marker=shapes[is_stability[k]], s=sizes[j])
        
    plt.legend()
    plt.show()

