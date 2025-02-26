import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch
import os
import h5py
from sklearn.decomposition import PCA
import scipy.io as si
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pdb
from matplotlib.colors import ListedColormap
from scipy.interpolate import interp1d

def get_period_pca(xx_trials,period,nComponents):
    # Which times the trajectory points will be drawn from
    if period == 'instruction':
        times = np.arange(29,79) # Instruction
    elif period == 'stim1':
        times = np.arange(79,129) # Stim1
    elif period == 'stim2':
        times = np.arange(179,229) # Stim2        
    # Getting period time PCA
    period_avg = np.mean(np.squeeze(xx_trials[:,times,:]),axis=1)
    pca = PCA(n_components=nComponents)
    pca.fit(period_avg)
    pca_period = pca
    return pca_period

# Which non-linearity is used (currently sigmoid)
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

#modality_type = 'instr'
#modality_type = 'instr_two_layers'
modality_type = 'instr2'

base_folder = '/home/cfxuser/Documents/Neurips/models/' + modality_type


# Custom colormap
hex_colors =['#1a2650', '#7185cf','#ff4242', '#6b0000'];  # light green, dark green, light purple, dark purple
minima_colors =['#ffffff','#5a2b79', '#1f6136','#85b89f', '#b583ba']; 
vec = [100, 50, 25, 0]

raw_colors = np.array([tuple(int(hex_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)) for hex_color in hex_colors]) / 255

N = 256
interp_values = np.linspace(100, 0, N)
interp_colors = np.array([interp1d(vec, raw_colors[:, i], axis=0)(interp_values) for i in range(3)]).T

cmap = ListedColormap(interp_colors)


if modality_type == 'instr':
    model_name = 'Task_instr_N_1000_Taus_4.0_25.0_Act_sigmoid_2023_04_24_013958'    
    modalities = [1]
    # External input (stim1/stim2/instruction)
    stim_task = np.array([0,0,1]).T.astype('float32')  
    stim_anti_task = np.array([0,0,-1]).T.astype('float32')
    stim_null = np.array([0,0,0]).T.astype('float32')
    title_text = '1-modality/1-layer'
elif modality_type == 'instr2':
    model_name = 'Task_instr2_N_1000_Taus_4.0_25.0_Act_sigmoid_2023_04_25_061252'
    modalities = [1,-1]
    # External input (stim1/stim2/instruction)
    stim_task_mod1 = np.array([0,0,0,0,1,-1]).T.astype('float32')  
    stim_anti_task_mod1 = np.array([0,0,0,0,-1,-1]).T.astype('float32')
    stim_task_mod2 = np.array([0,0,0,0,1,1]).T.astype('float32')  
    stim_anti_task_mod2 = np.array([0,0,0,0,-1,1]).T.astype('float32')
    stim_null = np.array([0,0,0,0,0,0]).T.astype('float32')
    title_text = '2-modality/1-layer'
elif modality_type == 'instr_two_layers':
    model_name = 'Two_layer_feedback_Task_instr_N_1000_Taus_4.0_25.0_Act_sigmoid_2023_11_01_223000'    
    modalities = [1]
    # External input (stim1/stim2/instruction)
    stim_task = np.array([0,0,1]).T.astype('float32')  
    stim_anti_task = np.array([0,0,-1]).T.astype('float32')
    stim_null = np.array([0,0,0]).T.astype('float32')
    title_text = '1-modality/2-layers'

model_folder = os.path.join(base_folder,model_name)

# Getting RNN activity
synX_filename = r'synX.mat'
synX_path = os.path.join(model_folder,synX_filename)

# Get matrix of simulated nTrials x nTimes x nNeurons trajectories
with h5py.File(synX_path, "r") as f:
    print(f.keys())
    xx_trials = f['xx_trials'][()]
xx_trials = np.transpose(xx_trials, (2, 1, 0)) # nTrials x nTimes x nNeurons trajectories

RNN_model_file = model_name + '.mat'
rnn_data = si.loadmat(os.path.join(base_folder, RNN_model_file))

# Loading model parameters
taus_gaus = torch.tensor(rnn_data['taus_gaus'])
taus = torch.tensor(rnn_data['taus'][0])
w = torch.tensor(rnn_data['w'])
m = torch.tensor(rnn_data['m'])
w_in = torch.tensor(rnn_data['w_in'])
taus_sig = torch.sigmoid(taus_gaus)*(taus[1] - taus[0]) + taus[0] # Neural time constant
ww = torch.matmul(w, m) # Recurrent weight matrix


# Plot energy landscape figures
# Creating a 2d grid over PC space to plot different qs
nComponents = 2
pca_period = get_period_pca(xx_trials,'instruction',nComponents)


# PC space grid 
res = 100
if modality_type == 'instr':
    xVec = np.linspace(-70,50,res)
    yVec = np.linspace(-50,20,res)
elif modality_type == 'instr2':
    xVec = np.linspace(-100,200,res)
    yVec = np.linspace(-100,150,res)
elif modality_type == 'instr_two_layers':
    xVec = np.linspace(-100,50,res)
    yVec = np.linspace(-70,50,res)

if modality_type == 'instr' or modality_type == 'instr_two_layers':
    stim_cases = [stim_null,stim_task,stim_anti_task]
    stim_names = ['null','task','anti_task']
    vmin,vmax = 1.5,7
else:
    stim_cases = [stim_null,stim_task_mod1,stim_anti_task_mod1,stim_task_mod2,stim_anti_task_mod2]
    stim_names = ['null','task_mod1','anti_task_mod1','task_mod2','anti_task_mod2']
    vmin,vmax = 2,8

# Plot first energy landscape as background, others only as minima
fig = plt.figure()
for plotI, u in enumerate(stim_cases):    
    u = torch.tensor(u)
    qMatrix = np.zeros((res,res))            
    qMatrix = np.zeros((res,res))            
    for i in np.arange(res):
        for j in np.arange(res):
            x_real = torch.tensor(pca_period.inverse_transform([xVec[i],yVec[j]]))
            qMatrix[i,j] = q_fun(x_real, taus_sig, ww, w_in, u)
    # Getting minimum energy point    
    min_idx = np.unravel_index(np.argmin(qMatrix),qMatrix.shape)
    #plt.imshow(np.log(qMatrix), extent=[xVec[0],xVec[-1],yVec[0],yVec[-1]])
    if plotI == 0:
        im = plt.imshow(np.log(qMatrix).T, extent=[xVec[0],xVec[-1],yVec[-1],yVec[0]],cmap=cmap, vmin=vmin, vmax=vmax)
    plt.plot(xVec[min_idx[0]],yVec[min_idx[1]],'x',c=minima_colors[plotI])
plt.title(title_text)


#fig.colorbar(im, ax=axs.ravel().tolist())

# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
#divider = make_axes_locatable(axs[-1])
#cax = divider.append_axes("right", size="5%", pad=0.05)
   
cbar=plt.colorbar()
cbar.set_label('log(Energy)')

# common axis labels
plt.xlabel('PC1')
plt.ylabel('PC2')

if modality_type == 'instr':
    modality_name = '1-mod/1-layer'
elif modality_type == 'instr_two_layers':
    modality_name = '1-mod/2-layers'
elif modality_type == 'instr2':
    modality_name = '2-mod/1-layer'

#fig.suptitle('Instruction energy landscape: '+modality_name)
#plt.title('Energy: u=task / PC period = instruction')
#cbar = plt.colorbar()
#cbar.set_label('log Energy')
plt.savefig('/home/cfxuser/Documents/Neurips/Figures/'+"local_figure_energy_landscape_over_null.png")
plt.savefig('/home/cfxuser/Documents/Neurips/Figures/'+modality_type+"_energy_landscape_over_null.pdf")
plt.show()