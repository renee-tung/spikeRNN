% Name: Robert Kim
% Date: October 11, 2019
% Email: rkim@salk.edu
% eval_go_nogo.m


%% get models of interest

clear; clc;

current_path = pwd;

% First, load one trained rate RNN
% Make sure lambda_grid_search.m was performed on the model.

% Update model_path to point where the trained model is

task_dir = '/scratch/spikeRNN/models/DMS_OSF';
models_type = 'good_models';

cd(task_dir)
if ~exist(models_type, 'dir')
    mkdir(models_type)
end
model_group_dir_path = strcat(task_dir,'/',models_type);

cd(current_path)

% Use Robert func to get good models
task_type = 'xor';
wcard = '*Taus_*';
max_tr = 5999; % 7999 for multi-XOR
perf_threshold = 0.95;
% perf_threshold = [0.60 0.80];
disp(['PERFORMANCE THRESHOLD SET TO ' num2str(perf_threshold)]);
stable_mods = return_stable(task_dir, wcard, perf_threshold, task_type, max_tr);

save([model_group_dir_path,'/',models_type,'_list.mat'], 'stable_mods')

%% get bad models

clear; clc;

current_path = pwd;

% First, load one trained rate RNN
% Make sure lambda_grid_search.m was performed on the model.

% Update model_path to point where the trained model is

task_dir = '/home/nuttidalab/Documents/spikeRNN/models/DMS_OSF';
models_type = 'bad_models';

cd(task_dir)
if ~exist(models_type, 'dir')
    mkdir(models_type)
end
model_group_dir_path = strcat(task_dir,'/',models_type);

cd(current_path)

% Use Robert func to get bad models
task_type = 'xor';
wcard = '*Taus_*';
max_tr = 5999; % 7999 for multi-XOR
% perf_threshold = 0.95;
perf_threshold = [0.60 0.80];
disp(['PERFORMANCE THRESHOLD SET TO ' num2str(perf_threshold)]);
stable_mods = return_stable(task_dir, wcard, perf_threshold, task_type, max_tr);

save([model_group_dir_path,'/',models_type,'_list.mat'], 'stable_mods')

%% Description: script to get IPSCs avg'd over 50 trials of +1 first or -1 first

normalize_ipscs = 1;
lesion_connections = 0; % if not lesioning put 0, else 'ii' etc
longer_delay = 250; % '' if standard delay (150), else a number (eg 200, 250)

models_types = {'good_models';'bad_models'};
% models_types = {'bad_models'};

if normalize_ipscs
    norm_name = '';
else
    norm_name = '_raw';
end
if ischar(lesion_connections)
    lesion_name = ['_lesion',lesion_connections];
else
    lesion_name = '';
end

for n_type = 1:length(models_types)
    models_type = models_types{n_type}; % CHANGE WHICH MODEL HERE

    clearvars -except models_types n_type models_type normalize_ipscs lesion_connections norm_name lesion_name longer_delay;
    
    current_path = pwd;
    
    task_dir = '/home/nuttidalab/Documents/spikeRNN/models/DMS_OSF';
    
    % Load previously saved models in the group of interest
    stable_mods = load([task_dir,'/',models_type, '/', models_type,'_list']).stable_mods;
    
    for n_model = 1:length(stable_mods)
        fprintf('\n');
        disp([models_types{n_type}, ' number ',num2str(n_model), ' of ', num2str(length(stable_mods))])
    
        % get the model
        model_name = stable_mods{n_model};
    
        % make a folder for the model
        cd(task_dir)
        if ~exist(model_name(1:(end-4)), 'dir')
            mkdir(model_name(1:(end-4)))
        end
        model_dir_path = strcat(task_dir,'/',model_name(1:(end-4)));
        cd(current_path)

        % save name for the output
        save_name = [model_dir_path,'/','IPSCs_50travg',norm_name,lesion_name,num2str(longer_delay),'.mat'];

        % check if there are IPSCs calculated already
        if exist(save_name, 'file') > 0
            disp('already calculated, moving to next model...')
            continue
        end

        file_path = fullfile(task_dir, model_name);
        load(file_path);
    
        % dif params for model function
        use_initial_weights = false;
        scaling_factor = opt_scaling_factor;
        down_sample = 1;
        stims = struct(); 
        stims.mode = 'none'; % For LIF simulation, no stims
    
        % trial params
        % T = 411;
        stim_on = 31;
        stim_dur = 50;
        if isempty(longer_delay)
            delay = 150; % standard delay we're using
        else
            delay = longer_delay; 
        end
        T = 261+delay;
        
        fs_rate = 200;
        fs_spk = 20000;
        
        stim1_onset = (stim_on)/fs_rate*fs_spk;
        stim1_offset = (stim_on + stim_dur)/fs_rate*fs_spk;
        stim2_onset = (stim_on + stim_dur + delay)/fs_rate*fs_spk;
        stim2_offset = (stim_on + 2*stim_dur + delay)/fs_rate*fs_spk;
        baseline_onset = round(stim1_onset/2);
        
        n_trials = 50;
        ds = 1;
        
        disp('Generating IPSCs from trials...')
        % +1/+1
        ipscs_samepos = zeros(N,T*100);
        u = zeros(2, T+1); % input stim
        u(1, stim_on:stim_on+stim_dur) = 1; % first stim is +1
        u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = 1; % second stim is +1
        for i=1:n_trials
            if ischar(lesion_connections)
                if i==1
                    disp(['lesioning ',lesion_connections, ' connections...'])
                end
                [~, ~, ~, ~, ~, ~, params] = LIF_network_lesion_fnc(file_path, scaling_factor,...
                    u, stims, ds, lesion_connections);
            else
                if i==1
                    disp('not lesioning')
                end
                [~, ~, ~, ~, ~, ~, params] = LIF_network_fnc(file_path, scaling_factor,...
                    u, stims, ds, use_initial_weights);

            end
            ipscs_temp = params.IPSCs;
            if normalize_ipscs
                if i==1
                    disp(['normalizing IPSCs...'])
                end
                ipscs_mean = mean(ipscs_temp(:,baseline_onset:stim1_onset),2); 
                ipscs_std = std(ipscs_temp(:,baseline_onset:stim1_onset),0, 2);
                if ipscs_std == 0
                    disp('STD = 0')
                end
                ipscs_temp = (ipscs_temp-ipscs_mean)./ipscs_std; % zscore by baseline period
            end
            ipscs_samepos = ipscs_samepos + ipscs_temp;
            clear params ipscs_temp
        end
        ipscs_samepos = ipscs_samepos / n_trials;

        % -1/-1
        ipscs_sameneg = zeros(N,T*100);
        u = zeros(2, T+1); % input stim
        u(1, stim_on:stim_on+stim_dur) = -1; % first stim is -1
        u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = -1; % second stim is -1
        for i=1:n_trials
            [~, ~, ~, ~, ~, ~, params] = LIF_network_fnc(file_path, scaling_factor,...
                u, stims, ds, use_initial_weights);
            ipscs_temp = params.IPSCs;
            if normalize_ipscs
                ipscs_mean = mean(ipscs_temp(:,baseline_onset:stim1_onset),2);
                ipscs_std = std(ipscs_temp(:,baseline_onset:stim1_onset),0, 2);
                ipscs_temp = (ipscs_temp-ipscs_mean)./ipscs_std; % zscore by baseline period
            end
            ipscs_sameneg = ipscs_sameneg + ipscs_temp;
            clear params ipscs_temp
        end
        ipscs_sameneg = ipscs_sameneg / n_trials;
        
        
        save(save_name, 'ipscs_samepos', 'ipscs_sameneg', ...
            'T', 'stim_on', 'stim_dur', 'delay', 'stim1_onset', 'stim2_onset')
        
    end
end


%% get 30 trials of each type and save (for PACs)

normalize_ipscs = 1;
lesion_connections = 'ii'; % if not lesioning put 0, else 'ii' etc
longer_delay = ''; % '' if standard delay (150), else a number (eg 200, 250)

models_types = {'good_models';'bad_models'};
% models_types = {'bad_models'};

if normalize_ipscs
    norm_name = '';
else
    norm_name = '_raw';
end
if ischar(lesion_connections)
    lesion_name = ['_lesion',lesion_connections];
else
    lesion_name = '';
end

for n_type = 1:length(models_types)
    models_type = models_types{n_type}; % CHANGE WHICH MODEL HERE

    clearvars -except models_types n_type models_type normalize_ipscs lesion_connections norm_name lesion_name longer_delay;
    
    current_path = pwd;
    
    task_dir = '/home/nuttidalab/Documents/spikeRNN/models/DMS_OSF';
    
    % Load previously saved models in the group of interest
    stable_mods = load([task_dir,'/',models_type, '/', models_type,'_list']).stable_mods;
    
    for n_model = 1:length(stable_mods)
        fprintf('\n');
        disp([models_types{n_type}, ' number ',num2str(n_model), ' of ', num2str(length(stable_mods))])
    
        % get the model
        model_name = stable_mods{n_model};
    
        % make a folder for the model
        cd(task_dir)
        if ~exist(model_name(1:(end-4)), 'dir')
            mkdir(model_name(1:(end-4)))
        end
        model_dir_path = strcat(task_dir,'/',model_name(1:(end-4)));
        cd(current_path)

        % save name for the output
        save_name = [model_dir_path,'/','IPSCs_30trall',norm_name,lesion_name,num2str(longer_delay),'.mat'];

        % check if there are IPSCs calculated already
        if exist(save_name, 'file') > 0
            disp('already calculated, moving to next model...')
            continue
        end

        file_path = fullfile(task_dir, model_name);
        load(file_path);
    
        % dif params for model function
        use_initial_weights = false;
        scaling_factor = opt_scaling_factor;
        down_sample = 1;
        stims = struct(); 
        stims.mode = 'none'; % For LIF simulation, no stims
    
        % trial params
        % T = 411;
        stim_on = 31;
        stim_dur = 50;
        if isempty(longer_delay)
            delay = 150; % standard delay we're using
        else
            delay = longer_delay; 
        end
        T = 261+delay;
        
        fs_rate = 200;
        fs_spk = 20000;
        
        stim1_onset = (stim_on)/fs_rate*fs_spk;
        stim1_offset = (stim_on + stim_dur)/fs_rate*fs_spk;
        stim2_onset = (stim_on + stim_dur + delay)/fs_rate*fs_spk;
        stim2_offset = (stim_on + 2*stim_dur + delay)/fs_rate*fs_spk;
        baseline_onset = round(stim1_onset/2);

        n_trials = 30;
        ds = 1;

        disp('Generating IPSCs from trials...')

        % +1/+1
        ipscs_samepos = zeros(N,T*100,n_trials);
        for i = 1:n_trials
            u = zeros(2, T+1); % input stim
            u(1, stim_on:stim_on+stim_dur) = 1; % first stim is +1
            u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = 1; % second stim is +1
            if ischar(lesion_connections)
                if i==1
                    disp(['lesioning ',lesion_connections, ' connections...'])
                end
                [~, ~, ~, ~, ~, ~, params] = LIF_network_lesion_fnc(file_path, scaling_factor,...
                    u, stims, ds, lesion_connections);
            else
                if i==1
                    disp('not lesioning')
                end
                [~, ~, ~, ~, ~, ~, params] = LIF_network_fnc(file_path, scaling_factor,...
                    u, stims, ds, use_initial_weights);
            end
            
            ipscs_temp = params.IPSCs;
            if normalize_ipscs
                if i==1
                    disp('normalizing IPSCs...')
                end
                ipscs_mean = mean(ipscs_temp(:,baseline_onset:stim1_onset),2);
                ipscs_std = std(ipscs_temp(:,baseline_onset:stim1_onset),0, 2);
                if ipscs_std ~= 0
                    ipscs_temp = (ipscs_temp-ipscs_mean)./ipscs_std; % zscore by baseline period
                end
            end
            if any(isnan(ipscs_temp))
                disp('there are nans')
            end
            ipscs_samepos(:,:,i) = ipscs_temp;
            
            clear params
        end

        % -1/-1
        ipscs_sameneg = zeros(N,T*100,n_trials);
        for i = 1:n_trials
            u = zeros(2, T+1); % input stim
            u(1, stim_on:stim_on+stim_dur) = -1; % first stim is -1
            u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = -1; % second stim is -1
            if ischar(lesion_connections)
                if i==1
                    disp(['lesioning ',lesion_connections, ' connections...'])
                end
                [~, ~, ~, ~, ~, ~, params] = LIF_network_lesion_fnc(file_path, scaling_factor,...
                    u, stims, ds, lesion_connections);
            else
                if i==1
                    disp('not lesioning')
                end
                [~, ~, ~, ~, ~, ~, params] = LIF_network_fnc(file_path, scaling_factor,...
                    u, stims, ds, use_initial_weights);
            end

            ipscs_temp = params.IPSCs;
            if normalize_ipscs
                if i==1
                    disp('normalizing IPSCs...')
                end
                ipscs_mean = mean(ipscs_temp(:,baseline_onset:stim1_onset),2);
                ipscs_std = std(ipscs_temp(:,baseline_onset:stim1_onset),0, 2);
                if ipscs_std ~= 0
                    ipscs_temp = (ipscs_temp-ipscs_mean)./ipscs_std; % zscore by baseline period
                end
            end
            if any(isnan(ipscs_temp))
                disp('there are nans')
            end
            ipscs_sameneg(:,:,i) = ipscs_temp;

            clear params
        end

        save(save_name, 'ipscs_samepos','ipscs_sameneg', ...
            'T', 'stim_on', 'stim_dur', 'delay')
        
    end
end


%% trial examples (old)


% --------------------------------------------------------------
% Same +1/+1 trial example
% --------------------------------------------------------------

T = 350;
stim_on = 50;
stim_dur = 50;
delay = 150; % longer delay

u = zeros(2, T+1); % input stim
u(1, stim_on:stim_on+stim_dur) = 1; % first stim is +1
u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = 1; % second stim is +1


% Run the LIF simulation 
stims = struct();
stims.mode = 'none';
[W, REC, spk, rs, all_fr, out, params] = LIF_network_fnc(file_path, scaling_factor,...
u, stims, ds, use_initial_weights);
dt = params.dt;
T = params.T;
t = dt:dt:T;

same_out = out;   % LIF network output
same_rs = rs;     % firing rates
same_spk = spk;   % spikes
same_IPSCs_pos = params.IPSCs;  % IPSCs

plot(same_out)


% --------------------------------------------------------------
% Same -1/-1 trial example
% --------------------------------------------------------------

T = 350;
stim_on = 50;
stim_dur = 50;
delay = 150; % longer delay

u = zeros(2, T+1); % input stim
u(1, stim_on:stim_on+stim_dur) = -1; % first stim is -1
u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = -1; % second stim is -1

plot(transpose(u))


% Run the LIF simulation 
stims = struct();
stims.mode = 'none';
[W, REC, spk, rs, all_fr, out, params] = LIF_network_fnc(file_path, scaling_factor,...
u, stims, ds, use_initial_weights);
dt = params.dt;
T = params.T;
t = dt:dt:T;

same_out = out;   % LIF network output
same_rs = rs;     % firing rates
same_spk = spk;   % spikes
same_IPSCs_neg = params.IPSCs;  % IPSCs

% --------------------------------------------------------------
% Diff +1/-1 trial example
% --------------------------------------------------------------

T = 350;
stim_on = 50;
stim_dur = 50;
delay = 150; % longer delay

u = zeros(2, T+1); % input stim
u(1, stim_on:stim_on+stim_dur) = 1; % first stim is +1
u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = -1; % second stim is -1

plot(transpose(u))

% Run the LIF simulation
stims = struct();
stims.mode = 'none';
[W, REC, spk, rs, all_fr, out, params] = LIF_network_fnc(file_path, scaling_factor,...
    u, stims, ds, use_initial_weights);
dt = params.dt;
T = params.T;
t = dt:dt:T;

% diff_out = out;   % LIF network output
% diff_rs = rs;     % firing rates
% diff_spk = spk;   % spikes
diff_IPSCs_pos = params.IPSCs;  % IPSCs

% --------------------------------------------------------------
% Diff -1/+1 trial example
% --------------------------------------------------------------

T = 350;
stim_on = 50;
stim_dur = 50;
delay = 150; % longer delay

u = zeros(2, T+1); % input stim
u(1, stim_on:stim_on+stim_dur) = -1; % first stim is -1
u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = 1; % second stim is +1

plot(transpose(u))

% Run the LIF simulation
stims = struct();
stims.mode = 'none';
[W, REC, spk, rs, all_fr, out, params] = LIF_network_fnc(file_path, scaling_factor,...
    u, stims, ds, use_initial_weights);
dt = params.dt;
T = params.T;
t = dt:dt:T;

diff_out = out;   % LIF network output
diff_rs = rs;     % firing rates
diff_spk = spk;   % spikes
diff_IPSCs_neg = params.IPSCs;  % IPSCs


% --------------------------------------------------------------
% Plot the network output
% --------------------------------------------------------------
figure; axis tight; hold on;
plot(t, same_out, 'm', 'linewidth', 2);
plot(t, diff_out, 'g', 'linewidth', 2);


% --------------------------------------------------------------
% Plot the spike raster
% --------------------------------------------------------------
% NoGo spike raster
figure('Units', 'Normalized', 'Outerposition', [0 0 0.22 0.20]);
hold on; axis tight;
inh_ind = find(inh);
exc_ind = find(exc);
all_ind = [exc_ind; inh_ind];
all_ind = 1:N;
for i = 1:length(all_ind)
  curr_spk = same_spk(all_ind(i), 10:end);
  if exc(all_ind(i)) == 1
    plot(t(find(curr_spk)), ones(1, length(find(curr_spk)))*i, 'r.', 'markers', 8);
  else
    plot(t(find(curr_spk)), ones(1, length(find(curr_spk)))*i, 'b.', 'markers', 8);
  end
end
xlim([0, 1]);
ylim([-5, 205]);

% Go spike raster
figure('Units', 'Normalized', 'Outerposition', [0 0 0.22 0.20]);
hold on; axis tight;
inh_ind = find(inh);
exc_ind = find(exc);
all_ind = [exc_ind; inh_ind];
all_ind = 1:N;
for i = 1:length(all_ind)
  curr_spk = diff_spk(all_ind(i), 10:end);
  if exc(all_ind(i)) == 1
    plot(t(find(curr_spk)), ones(1, length(find(curr_spk)))*i, 'r.', 'markers', 8);
  else
    plot(t(find(curr_spk)), ones(1, length(find(curr_spk)))*i, 'b.', 'markers', 8);
  end
end
xlim([0, 1]);
ylim([-5, 205]);


% --------------------------------------------------------------
% Plot the IPSCs
% --------------------------------------------------------------

figure; hold on; 
plot(transpose(diff_IPSCs_pos))
title('Diff trial IPSCs +1/-1')

figure; hold on; 
plot(transpose(same_IPSCs_pos))
title('Same trial IPSCs +1/+1')

figure; hold on; 
plot(transpose(diff_IPSCs_neg))
title('Diff trial IPSCs -1/+1')

figure; hold on; 
plot(transpose(same_IPSCs_neg))
title('Same trial IPSCs -1/-1')


% --------------------------------------------------------------
% Plot spectrograms
% --------------------------------------------------------------

window = 100;
noverlap = round(window/1.5);
nfft = window * 2;
fs = size(diff_spk,2);
[s, f, t] = spectrogram(diff_IPSCs(1,:), window, noverlap, nfft, fs);

figure; hold on;
imagesc(10*log(abs(real(s))))
% imagesc(real(s))
colorbar
set(gca, 'YDir','normal')


% --------------------------------------------------------------
% Save IPSCs
% --------------------------------------------------------------

save([model_dir_path,'/','IPSCs.mat'], 'diff_IPSCs_pos','same_IPSCs_pos', 'diff_IPSCs_neg','same_IPSCs_neg')




%% PACs calculation

% starting parameters, inspired by Daume et al 2024
% here: https://github.com/rutishauserlab/SBCAT-release-NWB/blob/main/NWB_SBCAT_analysis/helpers/internal/PAC/cfc_tort_comodulogram.m

% Input
% datMat: 2d matrix containing time samples of interest per trial (samples x trials)
% srate: sampling rate
% n_surrogates: number of surrogates for obtaining z-scored comodulogram (0 = no surrogates computed; default: 0)
% n_bins: number of bins to compute modulation index (default: 18 bins)
% LF_steps: center frequencies for phase signal in Hz (default: 2:2:14)
% LF_bw: Bandwidth for phase signal in Hz (default: 2)
% HF_steps: center frequencies for amplitude signal in Hz (default: 30:5:150; bandwidth is determined by phase signal frequency)
% tcutEdge: time to cut off at the edges of each trial to prevent filter artifacts in s (full time will be cutoff at the beginning and end of trial; default: 0 (no cutoff)) 
%
% Output
% comdlgrm: raw MI comodulugram; LF_steps x HF_steps
% comdlgrm_z: z-scored MI comodulugram; if n_surrogates > 0; LF_steps x HF_steps

% clear; clc
% load('Task_xor_N_200_Taus_4.0_25.0_Act_sigmoid_2019_09_06_152659_IPSCs.mat', 'diff_IPSCs','same_IPSCs');

%% get 15 trials

T = 300;
stim_on = 50;
stim_dur = 50;
delay = 10;

u = zeros(2, T+1); % input stim
u(1, stim_on:stim_on+stim_dur) = 1; % first stim is +1
u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = -1; % second stim is -1

n_trials = 15;

% Run the LIF simulation on n_trials trials
diff_IPSCs = zeros([200,30000, n_trials]); % 200 neurons, 30k samples, n_trials
for i=1:n_trials

    stims = struct();
    stims.mode = 'none';
    [W, REC, spk, rs, all_fr, out, params] = LIF_network_fnc(file_path, scaling_factor,...
        u, stims, ds, use_initial_weights);
    dt = params.dt;
    T = params.T;
    t = dt:dt:T;

    diff_IPSCs(:,:,i) = params.IPSCs;  % IPSCs

end

clearvars -except diff_IPSCs model_dir_path

%% organize data for function

datMat = squeeze(diff_IPSCs(1,:,:)); % just take the first neuron; samples x trials 
srate = 20000; % sampling rate
n_surrogates = 200; % 200 were used for z-scored MI in Daume et al
n_bins = 18; % go by default
LF_steps = 2:1:30;
LF_bw = 2;
HF_steps = 30:2:150;
tcutEdge = 0;

% run function and save variables
[comdlgrm, comdlgrm_z, phase2power] = cfc_tort_comodulogram(datMat,srate,n_surrogates,n_bins,LF_steps,LF_bw,HF_steps,tcutEdge);
save([model_dir_path, '/', 'comodulogram_diffIPSCs15.mat'],'comdlgrm','comdlgrm_z','phase2power')
load([model_dir_path, '/', 'comodulogram_diffIPSCs15.mat'],'comdlgrm','comdlgrm_z','phase2power')

% plot
figure; hold on; axis image
x0=10; y0=10; width=800; height=1500;
set(gcf,'position',[x0,y0,width,height])
imagesc(transpose(comdlgrm_z));
yticks(1:length(HF_steps)); yticklabels(HF_steps); ylabel('frequency for amplitude (Hz)', 'FontSize',16);
xticks(1:length(LF_steps)); xticklabels(LF_steps); xlabel('frequency for phase (Hz)','FontSize',16)
cb = colorbar(); ylabel(cb, 'Modulation Index, z-scored', 'FontSize',16,'Rotation',270)
saveas(gcf, 'comdlgrmz_diffIPSCs15.png')


figure; hold on; axis image
x0=10; y0=10; width=800; height=1500;
set(gcf,'position',[x0,y0,width,height])
imagesc(transpose(comdlgrm));
yticks(1:length(HF_steps)); yticklabels(HF_steps); ylabel('frequency for amplitude (Hz)', 'FontSize',16);
xticks(1:length(LF_steps)); xticklabels(LF_steps); xlabel('frequency for phase (Hz)','FontSize',16)
cb = colorbar(); ylabel(cb, 'Modulation Index', 'FontSize',16,'Rotation',270)
saveas(gcf, 'comdlgrm_diffIPSCs15.png')


[n_phase, n_amplitude, n_bins] = size(phase2power);
for i_phase = 1:n_phase
    figure; hold on; 
    for i_amplitude = 1:n_amplitude
        plot(squeeze(phase2power(i_phase, i_amplitude, :)))
    end
    title(['phase ',num2str(LF_steps(i_phase))])
end

%% plot dynamics of 25 avg'd trials of each type

model_name = stable_mods{1};
    
% make a folder for the model
cd(task_dir)
if ~exist(model_name(1:(end-4)), 'dir')
    mkdir(model_name(1:(end-4)))
end
model_dir_path = strcat(task_dir,'/',model_name(1:(end-4)));
cd(current_path)

file_path = fullfile(task_dir, model_name);
load(file_path);
scaling_factor = opt_scaling_factor;
stims = struct();
stims.mode = 'none';
down_sample=1;
use_initial_weights = false;

T = 411;
stim_on = 31;
stim_dur = 50;
delay = 150; % longer delay

fs_rate = 200;
fs_spk = 20000;

stim1_onset = (stim_on)/fs_rate*fs_spk;
stim1_offset = (stim_on + stim_dur)/fs_rate*fs_spk;
stim2_onset = (stim_on + stim_dur + delay)/fs_rate*fs_spk;
stim2_offset = (stim_on + 2*stim_dur + delay)/fs_rate*fs_spk;

n_trials = 25;
n_components = 3;
time_cutoff = 501;

pcs_all = zeros(T*fs_spk/fs_rate - time_cutoff + 1, 4, n_components);

% +1/+1
u = zeros(2, T+1); % input stim
u(1, stim_on:stim_on+stim_dur) = 1; % first stim is +1
u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = 1; % second stim is +1
for i=1:n_trials
    [~, ~, ~, rs, ~, ~, ~] = LIF_network_fnc(file_path, scaling_factor,...
        u, stims, down_sample, use_initial_weights);
    rs = rs(:,time_cutoff:end);
    pcs = rs' * pca(rs');
    pcs_all(:,1,:) = pcs_all(:,1,:) + reshape(pcs(:,1:n_components),length(rs),1,n_components);
end
pcs_all(:,1,:) = pcs_all(:,1,:) / n_trials;

% +1/-1
u = zeros(2, T+1); % input stim
u(1, stim_on:stim_on+stim_dur) = 1; % first stim is +1
u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = -1; % second stim is -1
for i=1:n_trials
    [~, ~, ~, rs, ~, ~, ~] = LIF_network_fnc(file_path, scaling_factor,...
        u, stims, down_sample, use_initial_weights);
    rs = rs(:,time_cutoff:end);
    pcs = rs' * pca(rs');
    pcs_all(:,2,:) = pcs_all(:,2,:) + reshape(pcs(:,1:n_components),length(rs),1,n_components);
end
pcs_all(:,2,:) = pcs_all(:,2,:) / n_trials;

% -1/-1
u = zeros(2, T+1); % input stim
u(1, stim_on:stim_on+stim_dur) = -1; % first stim is -1
u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = -1; % second stim is -1
for i=1:n_trials
    [~, ~, ~, rs, ~, ~, ~] = LIF_network_fnc(file_path, scaling_factor,...
        u, stims, down_sample, use_initial_weights);
    rs = rs(:,time_cutoff:end);
    pcs = rs' * pca(rs');
    pcs_all(:,3,:) = pcs_all(:,3,:) + reshape(pcs(:,1:n_components),length(rs),1,n_components);
end
pcs_all(:,3,:) = pcs_all(:,3,:) / n_trials;

% -1/+1
u = zeros(2, T+1); % input stim
u(1, stim_on:stim_on+stim_dur) = -1; % first stim is -1
u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = -1; % second stim is -1
for i=1:n_trials
    [~, ~, ~, rs, ~, ~, ~] = LIF_network_fnc(file_path, scaling_factor,...
        u, stims, down_sample, use_initial_weights);
    rs = rs(:,time_cutoff:end);
    pcs = rs' * pca(rs');
    pcs_all(:,4,:) = pcs_all(:,4,:) + reshape(pcs(:,1:n_components),length(rs),1,n_components);
end
pcs_all(:,4,:) = pcs_all(:,4,:) / n_trials;


colors = ["red","blue","cyan","magenta"];
labels = ["+1/+1","+1/-1","-1/+1","-1/-1"];

figure; hold on; view(3);
for i = 1:4
    h = plot3(pcs_all(:,i,1)', pcs_all(:,i,2)', pcs_all(:,i,3)','color',colors(i));
    set(h,{'DisplayName'},{labels(i)})
    h = scatter3(pcs_all(stim1_onset,i,1), pcs_all(stim1_onset,i,2), pcs_all(stim1_onset,i,3), 100, 'filled', colors(i),'<');
    set(h,{'DisplayName'},{'stim 1 onset'})
    h = scatter3(pcs_all(stim2_onset,i,1), pcs_all(stim2_onset,i,2), pcs_all(stim2_onset,i,3), 100, 'filled', colors(i));
    set(h,{'DisplayName'},{'stim 2 onset'})
    h = scatter3(pcs_all(end,i,1), pcs_all(end,i,2), pcs_all(end,i,3), 100, 'filled', colors(i),'pentagram');
    set(h,{'DisplayName'},{'end'})
end
legend show
saveas(gcf,[model_dir_path '/trajectories_delay150.png'])

% downsample
ds = 100;
pcs_all_ds = downsample(pcs_all,ds);
figure; hold on; view(3);
for i = 1:4
    h = plot3(pcs_all_ds(:,i,1)', pcs_all_ds(:,i,2)', pcs_all_ds(:,i,3)','color',colors(i));
    set(h,{'DisplayName'},{labels(i)})
    h = scatter3(pcs_all_ds(stim1_onset/ds,i,1), pcs_all_ds(stim1_onset/ds,i,2), pcs_all_ds(stim1_onset/ds,i,3), 100, 'filled', colors(i),'<');
    set(h,{'DisplayName'},{'stim 1 onset'})
    h = scatter3(pcs_all_ds(stim2_onset/ds,i,1), pcs_all_ds(stim2_onset/ds,i,2), pcs_all_ds(stim2_onset/ds,i,3), 100, 'filled', colors(i));
    set(h,{'DisplayName'},{'stim 2 onset'})
    h = scatter3(pcs_all_ds(end,i,1), pcs_all_ds(end,i,2), pcs_all_ds(end,i,3), 100, 'filled', colors(i),'pentagram');
    set(h,{'DisplayName'},{'end'})
end
legend show
saveas(gcf,[model_dir_path '/trajectories_delay150_downsample.png'])


% just stim 1
figure; hold on; view(3);
for i = 1:4
    h = plot3(pcs_all_ds(stim1_onset/ds:stim1_offset/ds,i,1)', pcs_all_ds(stim1_onset/ds:stim1_offset/ds,i,2)', pcs_all_ds(stim1_onset/ds:stim1_offset/ds,i,3)','color',colors(i));
    set(h,{'DisplayName'},{labels(i)})
    h = scatter3(pcs_all_ds(stim1_onset/ds,i,1), pcs_all_ds(stim1_onset/ds,i,2), pcs_all_ds(stim1_onset/ds,i,3), 100, 'filled', colors(i),'<');
    set(h,{'DisplayName'},{'stim 1 onset'})
end
legend show
title('Stim 1 duration')
saveas(gcf,[model_dir_path '/trajectories_delay150_downsample_stim1.png'])


% just delay period
figure; hold on; view(3);
for i = 1:4
    h = plot3(pcs_all_ds(stim1_offset/ds:stim2_onset/ds,i,1)', pcs_all_ds(stim1_offset/ds:stim2_onset/ds,i,2)', pcs_all_ds(stim1_offset/ds:stim2_onset/ds,i,3)','color',colors(i));
    set(h,{'DisplayName'},{labels(i)})
    h = scatter3(pcs_all_ds(stim2_onset/ds,i,1), pcs_all_ds(stim2_onset/ds,i,2), pcs_all_ds(stim2_onset/ds,i,3), 100, 'filled', colors(i));
    set(h,{'DisplayName'},{'stim 2 onset'})
end
legend show
title('delay period')
saveas(gcf,[model_dir_path '/trajectories_delay150_downsample_delay.png'])

% just stim 2
figure; hold on; view(3);
for i = 1:4
    h = plot3(pcs_all_ds(stim2_onset/ds:stim2_offset/ds,i,1)', pcs_all_ds(stim2_onset/ds:stim2_offset/ds,i,2)', pcs_all_ds(stim2_onset/ds:stim2_offset/ds,i,3)','color',colors(i));
    set(h,{'DisplayName'},{labels(i)})
    h = scatter3(pcs_all_ds(stim2_onset/ds,i,1), pcs_all_ds(stim2_onset/ds,i,2), pcs_all_ds(stim2_onset/ds,i,3), 100, 'filled', colors(i));
    set(h,{'DisplayName'},{'stim 2 onset'})
end
legend show
title('stim 2 onset & duration')
saveas(gcf,[model_dir_path '/trajectories_delay150_downsample_stim2.png'])

% just response period
figure; hold on; view(3);
for i = 1:4
    h = plot3(pcs_all_ds(stim2_offset/ds:end,i,1)', pcs_all_ds(stim2_offset/ds:end,i,2)', pcs_all_ds(stim2_offset/ds:end,i,3)','color',colors(i));
    set(h,{'DisplayName'},{labels(i)})
    h = scatter3(pcs_all_ds(end,i,1), pcs_all_ds(end,i,2), pcs_all_ds(end,i,3), 100, 'filled', colors(i),'pentagram');
    set(h,{'DisplayName'},{'end'})
end
legend show
title('response period')
saveas(gcf,[model_dir_path '/trajectories_delay150_downsample_response.png'])



%% try to quantify differences btwn trajectories

pcs_samepos_ds % +1/+1
pcs_diffpos_ds % +1/-1
pcs_diffneg_ds % -1/+1
pcs_sameneg_ds % -1/-1


difference = zeros(411, 4);
difference(:,1) = mean(pcs_samepos_ds-pcs_diffpos_ds, 2); % +/+ and +/-
difference(:,2) = mean(pcs_samepos_ds-pcs_sameneg_ds, 2); % +/+ and -/-
difference(:,3) = mean(pcs_sameneg_ds-pcs_diffneg_ds, 2); % -/- and -/+
difference(:,4) = mean(pcs_diffneg_ds-pcs_diffpos_ds, 2); % -/+ and +/-

figure; hold on;
plot(difference)
xline(stim1_onset/ds, '--k')
xline(stim1_offset/ds, '--k')
xline(stim2_onset/ds, '--k')
xline(stim2_offset/ds, '--k')
yline(0)
legend(['+/+ and +/-'; ...
    '+/+ and -/-'; ...
    '-/- and -/+'; ...
    '-/+ and +/-'])

saveas(gcf,[model_dir_path '/trajectories_delay150_downsample_differences.png'])



% n = 1000; % average every n values
% b = arrayfun(@(i) mean(a(i:i+n-1)),1:n:length(a)-n+1)'; % the averaged vector