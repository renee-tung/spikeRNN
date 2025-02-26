import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import h5py
from sklearn.decomposition import PCA
import pdb

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


#modality_type = 'instr'
#modality_type = 'instr2'
modality_type = 'instr_two_layers'

base_folder = '/home/cfxuser/Documents/Neurips/models/' + modality_type

if modality_type == 'instr':
    model_folder = base_folder+'/Task_instr_N_1000_Taus_4.0_25.0_Act_sigmoid_2023_04_24_013958/'
    modalities = [1]
elif modality_type == 'instr2':
    model_folder = base_folder+'/Task_instr2_N_1000_Taus_4.0_25.0_Act_sigmoid_2023_04_25_061252/'
    #model_folder = base_folder+'/Task_instr2_N_1000_Taus_4.0_25.0_Act_sigmoid_2023_04_27_051801/'    
    modalities = [1,-1]
elif modality_type == 'instr_two_layers':
    model_folder = base_folder+'/Two_layer_feedback_Task_instr_N_1000_Taus_4.0_25.0_Act_sigmoid_2023_11_01_223000/'
    modalities = [1]


x_star_pro = np.empty([0, 1000])
x_star_anti = np.empty([0, 1000])
x_star_null = np.empty([0, 1000])
is_stable_pro = np.empty([0])
is_stable_null = np.empty([0])
is_stable_anti = np.empty([0])

for modality_value in modalities:
    task_type = 'pro'
    pickle_name = base_folder+'/fixed_points_'+task_type+'_'+ str(modality_value) +'.pickle'
    file = open(pickle_name, 'rb')
    loaded_dict = pickle.load(file)
    file.close()
    x_star_pro = np.concatenate((x_star_pro,loaded_dict['xstar']),axis=0)
    is_stable_pro = np.concatenate((is_stable_pro,loaded_dict['is_stable']),axis=0)    

    task_type = 'null'
    pickle_name = base_folder+'/fixed_points_'+task_type+'_'+ str(modality_value) +'.pickle'
    file = open(pickle_name, 'rb')
    loaded_dict = pickle.load(file)
    file.close()
    x_star_null = np.concatenate((x_star_null,loaded_dict['xstar']),axis=0)    
    is_stable_null = np.concatenate((is_stable_null,loaded_dict['is_stable']),axis=0)        

    task_type = 'anti'
    pickle_name = base_folder+'/fixed_points_'+task_type+'_'+ str(modality_value) +'.pickle'
    file = open(pickle_name, 'rb')
    loaded_dict = pickle.load(file)
    file.close()
    x_star_anti = np.concatenate((x_star_anti,loaded_dict['xstar']),axis=0)    
    is_stable_anti = np.concatenate((is_stable_anti,loaded_dict['is_stable']),axis=0)    

# Converting to boolean
is_stable_pro = is_stable_pro != 0
is_stable_null = is_stable_null != 0
is_stable_anti = is_stable_anti != 0

# Getting RNN activity
synX_filename = r'synX.mat'
synX_path = os.path.join(model_folder,synX_filename)

# Get matrix of simulated nTrials x nTimes x nNeurons trajectories
with h5py.File(synX_path, "r") as f:
    print(f.keys())
    xx_trials = f['xx_trials'][()]

# Getting trial information 
trial_filename = r'trials.mat'
trial_path = os.path.join(model_folder,trial_filename)
# Get matrix of simulated nTrials x nTimes x nNeurons trajectories
with h5py.File(trial_path, "r") as f:    
    a_group_key = list(f.keys())[1]        
    instr_amp_trials = f[a_group_key]['instr_amp_trials'][()]
    instr_t_trials = f[a_group_key]['instr_t_trials'][()]
    #stim_id_trials = f[a_group_key]['stim_id_trials'][()]
    #stim_lab_trials = f[a_group_key]['stim_lab_trials'][()]
    #u_trials = f[a_group_key]['u_trials'][()]

# Which times the trajectory points will be drawn from
instr_times = np.arange(29,79) # Instruction
stim1_times = np.arange(79,129) # Stim1
instr2_times = np.arange(129,179) # Instruction 2
stim2_times = np.arange(179,229) # Stim2

xx_trials = np.transpose(xx_trials, (2, 1, 0)) # nTrials x nTimes x nNeurons trajectories
print(xx_trials.shape)

# Getting trajectories over time for each trial
nTrials = 1000
nComponents = 3
pca_period = get_period_pca(xx_trials,'instruction',nComponents)
instr_trajectories = np.zeros((nTrials, len(instr_times), nComponents))
for timeI, timeValue in enumerate(instr_times):
    instr_trajectories[:,timeI,:] = pca_period.transform(np.squeeze(xx_trials[:,timeValue,:]))


#fp_task = pca_period.transform(data_pro[0].reshape(1,-1))[0]
#fp_anti_task = pca_period.transform(data_anti[8].reshape(1,-1))[0]
fp_null = pca_period.transform(x_star_null)
fp_pro = pca_period.transform(x_star_pro)
fp_anti = pca_period.transform(x_star_anti)


# Plotting 3d trajectories for some trials
ax = plt.figure().add_subplot(projection='3d')
line1, = ax.plot(0, 0, 0, color='blue',label='pro')
line2, = ax.plot(0, 0, 0, color='red',label='anti')
line3, = ax.plot(0, 0, 0, color='black',label='late')
ax.legend(handles=[line1,line2,line3])

'''
ax.scatter(fp_null[is_stable_null,0],fp_null[is_stable_null,1],fp_null[is_stable_null,2],marker='o',color='black',edgecolor='yellow', s=40, alpha=1)
ax.scatter(fp_null[~is_stable_null,0],fp_null[~is_stable_null,1],fp_null[~is_stable_null,2],marker='x',color='black', s=20, alpha=1)

ax.scatter(fp_pro[is_stable_pro,0],fp_pro[is_stable_pro,1],fp_pro[is_stable_pro,2],marker='o',color='blue',edgecolor='yellow', s=40, alpha=1)
ax.scatter(fp_pro[~is_stable_pro,0],fp_pro[~is_stable_pro,1],fp_pro[~is_stable_pro,2],marker='x',color='blue', s=20, alpha=1)

ax.scatter(fp_anti[is_stable_anti,0],fp_anti[is_stable_anti,1],fp_anti[is_stable_anti,2],marker='o',color='red',edgecolor='yellow', s=40, alpha=1)
ax.scatter(fp_anti[~is_stable_anti,0],fp_anti[~is_stable_anti,1],fp_anti[~is_stable_anti,2],marker='x',color='red', s=20, alpha=1)
'''

ax.scatter(fp_null[:,0],fp_null[:,1],fp_null[:,2],marker='o',color='black',edgecolor='yellow', s=40, alpha=1)
ax.scatter(fp_pro[:,0],fp_pro[:,1],fp_pro[:,2],marker='o',color='blue',edgecolor='yellow', s=40, alpha=1)
ax.scatter(fp_anti[:,0],fp_anti[:,1],fp_anti[:,2],marker='o',color='red',edgecolor='yellow', s=40, alpha=1)


ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_zlabel("PC 3")
ax.set_title("State space at instruction")

nPlottedTrials = 50
for trialI in np.arange(nPlottedTrials):
    x = np.squeeze(instr_trajectories[trialI,:,0])
    y = np.squeeze(instr_trajectories[trialI,:,1])
    z = np.squeeze(instr_trajectories[trialI,:,2])
    if instr_amp_trials[0,trialI] == 1 and instr_t_trials[0,trialI] == -1:
        color='blue'
    elif instr_amp_trials[0,trialI] == -1 and instr_t_trials[0,trialI] == -1:
        color='red'          
    if instr_t_trials[0,trialI] == 1:
        color='black'           
    ax.plot(x, y, z, color=color)
ax.view_init(elev=20, azim=60, roll=0)

plt.savefig("local_figure.png")
plt.savefig(modality_type+"_state_space.pdf")

plt.show()



'''
nPlottedTrials = 50
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

fp_task = pca_period.transform(data_pro[0].reshape(1,-1))[0]
fp_anti_task = pca_period.transform(data_anti[8].reshape(1,-1))[0]
fp_null = pca_period.transform(data_null[5].reshape(1,-1))[0]

# Plotting 3d trajectories for some trials
ax = plt.figure().add_subplot(projection='3d')
line1, = ax.plot(0, 0, 0, color='blue',label='pro')
line2, = ax.plot(0, 0, 0, color='red',label='anti')
line3, = ax.plot(0, 0, 0, color='black',label='unknown')



ax.legend(handles=[line1,line2,line3])
for trialI in np.arange(nPlottedTrials):
    x = np.squeeze(instr_trajectories[trialI,:,0])
    y = np.squeeze(instr_trajectories[trialI,:,1])
    z = np.squeeze(instr_trajectories[trialI,:,2])
    
    if instr_amp_trials[0,trialI] == 1 and instr_t_trials[0,trialI] == -1:
        color='blue'
    elif instr_amp_trials[0,trialI] == -1 and instr_t_trials[0,trialI] == -1:
        color='red'          
    if instr_t_trials[0,trialI] == 1:
        color='black'           
    ax.plot(x, y, z, color=color)

    
    # Plot fixed points
    #ax.plot(fixed_points_pca[:,0], fixed_points_pca[:,1], fixed_points_pca[:,2], marker='s', color='green', markersize=8, linestyle="None")    
    
    # Plotting arrows
    arrow_x, arrow_y, arrow_z = x[-1],y[-1],z[-1]
    arrow_u, arrow_v, arrow_w = x[-1]-x[0],y[-1]-y[0],z[-1]-z[0]
    ax.quiver(arrow_x, arrow_y, arrow_z, arrow_u, arrow_v, arrow_w, length=2, normalize=True, color=color, lw=4)
    

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    ax.set_title("State space at instruction")

# Plotting fixed points
#plt.plot(fp_task[0],fp_task[1],fp_task[2],'bo',marker=(5, 2), markersize=10)
#plt.plot(fp_anti_task[0],fp_anti_task[1],fp_anti_task[2],'ro',marker=(5, 2), markersize=10)
#plt.plot(fp_null[0],fp_null[1],fp_null[2],'ko',marker=(5, 2), markersize=10)
ax.scatter(fp_task[0],fp_task[1],fp_task[2],marker='o',color='blue',edgecolor='yellow', s=40)
ax.scatter(fp_anti_task[0],fp_anti_task[1],fp_anti_task[2],marker='o',color='red',edgecolor='yellow', s=40)
ax.scatter(fp_null[0],fp_null[1],fp_null[2],marker='o',color='black',edgecolor='yellow', s=40)
    
    
plt.show()

'''