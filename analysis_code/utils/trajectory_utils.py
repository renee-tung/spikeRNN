import numpy as np
import os, scipy.io 
import numpy as np 
# %matplotlib ipympl
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import pickle as pk
import torch
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


def get_perf(model_path, settings, n_trials = 100, plot=0):
    '''
    calculate rate model performance on n_trials trials
    '''
    resp_onset = settings['stim_on'] + 2*settings['stim_dur'] + settings['delay']
    eval_amp_threshold = 0.7
    eval_perf = np.zeros(n_trials)
    if plot:
        plt.figure(figsize=(10, 5))
    for i in range(n_trials):
        u, label = generate_input_stim_xor(settings)
        _, _, o, _ = eval_tf(model_dir=model_path, settings=settings, u=u)
        
        if label == 'same':
            if plot:
                plt.plot(o, c='r')
            if np.max(o[resp_onset:]) > eval_amp_threshold:
                eval_perf[i] = 1
        else:
            if plot:
                plt.plot(o, c='b')
            if np.min(o[resp_onset:]) < -eval_amp_threshold:
                eval_perf[i] = 1
    if plot:
        plt.axvline(x=settings['stim_on'], c='k', linestyle='--')
        plt.axvline(x=settings['stim_on']+settings['stim_dur'], c='k', linestyle='--')
        plt.axvline(x=settings['stim_on']+settings['stim_dur']+settings['delay'], c='k', linestyle='--')
        plt.axvline(x=settings['stim_on']+2*settings['stim_dur']+settings['delay'], c='k', linestyle='--')
        plt.xlabel('Time')
        plt.ylabel('Output')
        plt.title(f'Rate model performance, {np.nanmean(eval_perf)}')
        plt.show()
        
    return np.nanmean(eval_perf)


def generate_synX(model_dir, settings, 
                  stim1s = [-1, 1], stim2s = [-1, 1],
                  n_trials = 1000, save=0, save_flag='', model_results_dir = []):
    '''
    generate trials of synaptic current matrix X for specified stims
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
        if len(save_flag) == 0:
            np.save(f'{model_results_dir}delay{settings["delay"]}_synX.npy', synX)
        else:
            np.save(f'{model_results_dir}delay{settings["delay"]}_synX_{save_flag}.npy', synX)
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
        fixed_pt_results = pk.dump(fixed_pt_results, open(f'{model_results_dir}delay{settings["delay"]}_fixed_pt_results.pkl', 'wb'))

    return fixed_pt_results


def plot_fixed_pts(fixed_pt_results, model_results_dir, settings,
                   xx_trials, trial_stim_labels, plot_examples=0,
                   skip_start = 0, skip_mid = 0, skip_end = 0, title=''):
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

    xx_trials_thrudelay = xx_trials_thrudelay.reshape(-1,n_neurons) # time x neurons

    # do pca
    pca_delay = PCA(n_components=3)
    pca_delay.fit(xx_trials_thrudelay)
    xx_pca = pca_delay.transform(xx_trials_thrudelay)
    xx_pca = xx_pca.reshape(-1,n_times,3)
    print(xx_pca.shape)


    xx_pca_pos_mean = np.mean(xx_pca[trial_stim_labels[:,0] == 1],axis=0)
    xx_pca_neg_mean = np.mean(xx_pca[trial_stim_labels[:,0] == -1],axis=0)

    xx_pca_pos = xx_pca[trial_stim_labels[:,0] == 1]
    xx_pca_neg = xx_pca[trial_stim_labels[:,0] == -1]

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
    sizes = [100, 50, 15]

    for i, task_type in enumerate(task_types): # +1 or -1
        color = colors[i]

        for j, time_fp_label in enumerate(time_fp_labels): # start, middle, end
            fp_dict = fixed_pt_results[(task_type, time_fp_label)]
            n_fp = len(fp_dict['xstar'])
            fixed_pt_pca = pca_delay.transform(fp_dict['xstar'])
            is_stability = fp_dict['is_stable']

            if skip_start and j == 0:
                continue
            if skip_mid and j == 1:
                continue
            if skip_end and j == 2:
                continue

            print(f'{task_type} {time_fp_label} fp count: {n_fp}, stability: {is_stability}')

            if n_fp > 0:
                for i_fp, fp in enumerate(fixed_pt_pca):
                    # for fixed points types (stable/unstable) that haven't been seen before, plot label
                    if (i_fp == 0) or (is_stability[i_fp] != is_stability[:i_fp]).all():
                        ax.scatter(fp[0],fp[1],fp[2],
                                color=color,marker=shapes[is_stability[i_fp]], 
                                label=f'{task_type} {label[is_stability[i_fp]]} {time_fp_label} fp', s=sizes[j])
                    else:
                        ax.scatter(fp[0],fp[1],fp[2],
                                color=color,marker=shapes[is_stability[i_fp]], s=sizes[j])
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    if len(title) > 0:
        ax.set_title(title)
        
    plt.legend()
    plt.show()


def non_linearity_r(x):
    #r = tf.expand_dims(tf.cast(tf.clip_by_value(tf.nn.relu(x), 0, 1),tf.float32),1)
    #r = tf.expand_dims(tf.cast(tf.math.sigmoid(x), tf.float32), 1)
    r = torch.unsqueeze(torch.sigmoid(x), 1).float()
    return r

# Defining function of state space evolution for state space analysis
# x: synaptic current, taus_sig: neural time constant
# ww: neural recurrent weight matrix, w_in: input weight matrix, stim: condition/instruction input
def q_fun(x, taus_sig, ww, w_in, stim):
    r = non_linearity_r(x)            
    #F = tf.multiply((1 - 1./taus_sig), np.expand_dims(x,1)) + \
    #            tf.multiply((1./taus_sig), ((tf.matmul(ww, r)) + tf.matmul(w_in, tf.expand_dims(stim, 1)))) 
    
    F = ((1 - 1. / taus_sig) * x.unsqueeze(1)) + \
    ((1. / taus_sig) * ((torch.matmul(ww, r)) + torch.matmul(w_in, stim.unsqueeze(1))))

        
    # Transposing to energy signal    
    q = 0.5*(torch.sum(np.power(F-np.expand_dims(x,1),2))).item()        
    return q

def get_period_pca(xx_trials,period,settings,nComponents):
    # Which times the trajectory points will be drawn from

	stim_on = settings['stim_on']
	stim_dur = settings['stim_dur']
	delay=settings['delay']
	T = settings['T']   
	
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



def plot_energy_landscape(RNN_model_file, settings,
                          models_dir = '/scratch/spikeRNN/models/DMS_OSF/',
                          xVec_lim=[-25,25], yVec_lim=[-15,40], res=100,
                          suptitle=[]):
    '''
    '''
    model_results_dir = f'{models_dir}{RNN_model_file[:-4]}/'
    rnn_path = os.path.join(models_dir, RNN_model_file) 

    if not os.path.exists(f'{model_results_dir}delay{settings["delay"]}_synX.npy'):
        print('Generating synX')
        xx_trials, trial_stim_labels = tu.generate_synX(rnn_path, settings, 
                                save=1, model_results_dir = model_results_dir)
    else:
        print('Loading synX')
        xx_trials = np.load(f'{model_results_dir}delay{settings["delay"]}_synX.npy')
        trial_stim_labels = np.load(f'{model_results_dir}trial_stim_labels.npy')
    
    rnn_data = scipy.io.loadmat(models_dir + RNN_model_file)
    # Loading model parameters
    taus_gaus = torch.tensor(rnn_data['taus_gaus'])
    taus = torch.tensor(rnn_data['taus'][0])
    w = torch.tensor(rnn_data['w'])
    m = torch.tensor(rnn_data['m'])
    w_in = torch.tensor(rnn_data['w_in'])
    taus_sig = torch.sigmoid(taus_gaus)*(taus[1] - taus[0]) + taus[0] # Neural time constant
    ww = torch.matmul(w, m) # Recurrent weight matrix
    
    stim_condns = [1, -1]
    stim_names = ['+1','-1']

    fig, axs = plt.subplots(1, len(stim_condns),sharey=True, figsize=(10,4))
    for plotI, stim1 in enumerate(stim_condns):
        # Creating a 2d grid over PC space to plot different qs
        nComponents = 2
        pca_period = get_period_pca(xx_trials[trial_stim_labels[:,0] == stim1,:,:],'delay',settings, nComponents)

        # PC space grid 
        res = 100

        xVec = np.linspace(xVec_lim[0],xVec_lim[1],res)
        yVec = np.linspace(yVec_lim[0],yVec_lim[1],res)
    
        u = torch.tensor([0,0], dtype=torch.float32) # current stim is always 0 bc delay
        qMatrix = np.zeros((res,res))            
        for i in np.arange(res):
            for j in np.arange(res):
                x_real = torch.tensor(pca_period.inverse_transform([xVec[i],yVec[j]]))
                qMatrix[i,j] = tu.q_fun(x_real, taus_sig, ww, w_in, u)
        # Getting minimum energy point    
        min_idx = np.unravel_index(np.argmin(qMatrix),qMatrix.shape)
        im = axs[plotI].imshow(np.log(qMatrix).T, extent=[xVec[0],xVec[-1],yVec[-1],yVec[0]], vmin=-4, vmax=4)
        axs[plotI].plot(xVec[min_idx[0]],yVec[min_idx[1]],'rx')
        axs[plotI].set_title(f'{stim_names[plotI]}')

    divider = make_axes_locatable(axs[-1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    cbar=plt.colorbar(im, cax=cax)
    cbar.set_label('log(Energy)')

    # common axis labels
    fig.supxlabel('PC1')
    fig.supylabel('PC2')
    if len(suptitle) > 0:
        fig.suptitle(suptitle)
    else:
        fig.suptitle(f'Energy landscape for {RNN_model_file[-4]}')

    # plt.subplots_adjust(right=0.9)
    plt.show()


def plot_energy_landscape_over_null(RNN_model_file, settings,
                          models_dir = '/scratch/spikeRNN/models/DMS_OSF/',
                          xVec_lim=[-25,25], yVec_lim=[-15,40], res=100,
                          plot=1, title=[]):
    '''
    '''
    model_results_dir = f'{models_dir}{RNN_model_file[:-4]}/'
    rnn_path = os.path.join(models_dir, RNN_model_file) 

    if not os.path.exists(f'{model_results_dir}delay{settings["delay"]}_synX.npy'):
        print('Generating synX')
        xx_trials, trial_stim_labels = generate_synX(rnn_path, settings, 
                                save=1, model_results_dir = model_results_dir)
    else:
        # print('Loading synX')
        xx_trials = np.load(f'{model_results_dir}delay{settings["delay"]}_synX.npy')
        trial_stim_labels = np.load(f'{model_results_dir}trial_stim_labels.npy')
    
    # if not os.path.exists(f'{model_results_dir}delay{settings["delay"]}_synX_null.npy'):
    #     print('Generating synX')
    #     xx_trials_null, _ = tu.generate_synX(rnn_path, settings, 
    #                             save=1, model_results_dir = model_results_dir)
    # else:
    #     print('Loading synX')
    #     xx_trials_null = np.load(f'{model_results_dir}delay{settings["delay"]}_synX_null.npy')
    #     trial_stim_labels = np.load(f'{model_results_dir}trial_stim_labels.npy')

    rnn_data = scipy.io.loadmat(models_dir + RNN_model_file)
    # Loading model parameters
    taus_gaus = torch.tensor(rnn_data['taus_gaus'])
    taus = torch.tensor(rnn_data['taus'][0])
    w = torch.tensor(rnn_data['w'])
    m = torch.tensor(rnn_data['m'])
    w_in = torch.tensor(rnn_data['w_in'])
    taus_sig = torch.sigmoid(taus_gaus)*(taus[1] - taus[0]) + taus[0] # Neural time constant
    ww = torch.matmul(w, m) # Recurrent weight matrix
    
    stim_condns = [0, 1, -1]
    stim_names = ['both', '+1','-1']
    markers = ['rx','mx']

    n_minima = len(stim_condns)-1
    minima = np.zeros((n_minima,2))
    # fig, axs = plt.subplots(1, len(stim_condns),sharey=True, figsize=(10,4))
    for plotI, stim1 in enumerate(stim_condns):
        # Creating a 2d grid over PC space to plot different qs
        nComponents = 2
        if stim1 == 0: # get both +1 and -1 trials
            pca_period = get_period_pca(xx_trials[trial_stim_labels[:,0] != stim1,:,:],'delay',settings, nComponents)
        else:
            pca_period = get_period_pca(xx_trials[trial_stim_labels[:,0] == stim1,:,:],'delay',settings, nComponents)

        # PC space grid 
        res = 100

        xVec = np.linspace(xVec_lim[0],xVec_lim[1],res)
        yVec = np.linspace(yVec_lim[0],yVec_lim[1],res)
    
        u = torch.tensor([0,0], dtype=torch.float32) # current stim is always 0 bc delay
        qMatrix = np.zeros((res,res))            
        for i in np.arange(res):
            for j in np.arange(res):
                x_real = torch.tensor(pca_period.inverse_transform([xVec[i],yVec[j]]))
                qMatrix[i,j] = tu.q_fun(x_real, taus_sig, ww, w_in, u)
        # Getting minimum energy point    
        min_idx = np.unravel_index(np.argmin(qMatrix),qMatrix.shape)
        minima[plotI-1,:] = [xVec[min_idx[0]],yVec[min_idx[1]]]
        if plot:
            if stim1 == 0:
                im = plt.imshow(np.log(qMatrix).T, extent=[xVec[0],xVec[-1],yVec[-1],yVec[0]], vmin=-4, vmax=4)
            else:
                plt.plot(xVec[min_idx[0]],yVec[min_idx[1]],markers[plotI-1], label=f'{stim_names[plotI]}')
    
    dist = np.linalg.norm(minima[0]-minima[1])
    if plot:       
        cbar=plt.colorbar(im)
        cbar.set_label('log(Energy)')
        plt.legend()
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        if len(title) > 0:
            title_dist = f'{title}, distance: {dist:.2f}'
            plt.title(title_dist)
        else:
            plt.title(f'Energy landscape for {RNN_model_file[-4]}')
        plt.show()

    return minima, dist


def plot_energy_landscapes(RNN_model_files, settings,
                          models_dir = '/scratch/spikeRNN/models/DMS_OSF/',
                          xVec_lim=[-25,25], yVec_lim=[-15,40], res=100,
                          suptitle=[]):
    '''
    this is more of a rough plotting function just for visualization...
    '''

    for i_model, RNN_model_file in enumerate(RNN_model_files):
        if i_model%3 == 0:
            fig, axs = plt.subplots(3, 2,sharey=True, figsize=(10,12))
        model_results_dir = f'{models_dir}{RNN_model_file[:-4]}/'
        rnn_path = os.path.join(models_dir, RNN_model_file) 

        if not os.path.exists(f'{model_results_dir}delay{settings["delay"]}_synX.npy'):
            print('Generating synX')
            xx_trials, trial_stim_labels = tu.generate_synX(rnn_path, settings, 
                                    save=1, model_results_dir = model_results_dir)
            
        else:
            print('Loading synX')
            xx_trials = np.load(f'{model_results_dir}delay{settings["delay"]}_synX.npy')
            trial_stim_labels = np.load(f'{model_results_dir}trial_stim_labels.npy')
            
        rnn_data = scipy.io.loadmat(models_dir + RNN_model_file)
        # Loading model parameters
        taus_gaus = torch.tensor(rnn_data['taus_gaus'])
        taus = torch.tensor(rnn_data['taus'][0])
        w = torch.tensor(rnn_data['w'])
        m = torch.tensor(rnn_data['m'])
        w_in = torch.tensor(rnn_data['w_in'])
        taus_sig = torch.sigmoid(taus_gaus)*(taus[1] - taus[0]) + taus[0] # Neural time constant
        ww = torch.matmul(w, m) # Recurrent weight matrix

        stim_condns = [1, -1]
        stim_names = ['+1','-1']

        for plotI, stim1 in enumerate(stim_condns):
            # Creating a 2d grid over PC space to plot different qs
            nComponents = 2
            pca_period = get_period_pca(xx_trials[trial_stim_labels[:,0] == stim1,:,:],'delay',settings, nComponents)

            # PC space grid 
            res = 100

            xVec = np.linspace(xVec_lim[0],xVec_lim[1],res)
            yVec = np.linspace(yVec_lim[0],yVec_lim[1],res)
        
            u = torch.tensor([0,0], dtype=torch.float32) # current stim is always 0 bc delay
            qMatrix = np.zeros((res,res))            
            for i in np.arange(res):
                for j in np.arange(res):
                    x_real = torch.tensor(pca_period.inverse_transform([xVec[i],yVec[j]]))
                    qMatrix[i,j] = tu.q_fun(x_real, taus_sig, ww, w_in, u)
            # Getting minimum energy point    
            min_idx = np.unravel_index(np.argmin(qMatrix),qMatrix.shape)
            im = axs[i_model%3, plotI].imshow(np.log(qMatrix).T, extent=[xVec[0],xVec[-1],yVec[-1],yVec[0]], vmin=-4, vmax=4)
            axs[i_model%3, plotI].plot(xVec[min_idx[0]],yVec[min_idx[1]],'rx')
            axs[i_model%3, plotI].set_title(f'{stim_names[plotI]}')

        divider = make_axes_locatable(axs[i_model%3, -1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        cbar=plt.colorbar(im, cax=cax)
        cbar.set_label('log(Energy)')

        # common axis labels
        fig.supxlabel('PC1')
        fig.supylabel('PC2')
        # if len(suptitle) > 0:
        #     fig.suptitle(suptitle)
        # else:
        #     fig.suptitle(f'Energy landscape for {RNN_model_file[-4]}')

        # plt.subplots_adjust(right=0.9)
        if i_model%3 == 2:
            fig.suptitle(suptitle)
            plt.show()
        
    # plt.show()
