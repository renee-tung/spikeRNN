% Name: Robert Kim
% Date: October 11, 2019
% Email: rkim@salk.edu
% eval_go_nogo.m
% Description: Script to evaluate a trained LIF RNN model constructed
% to perform the Go-NoGo task

clear; clc;

current_path = pwd;

% First, load one trained rate RNN
% Make sure lambda_grid_search.m was performed on the model.

% Update model_path to point where the trained model is
% model_path = '/Users/Renee/Downloads/spikeRNN/models/go-nogo/P_rec_0.2_Taus_4.0_20.0';

task_dir = '/scratch/spikeRNN/models/DMS_OSF';

% Use Robert func to get good models
task_type = 'xor';
wcard = '*Taus_*';
max_tr = 5999; % 7999 for multi-XOR
perf_threshold = 0.95;
% perf_threshold = [0.60 0.80];
disp(['PERFORMANCE THRESHOLD SET TO ' num2str(perf_threshold)]);
stable_mods = return_stable(task_dir, wcard, perf_threshold, task_type, max_tr);

% CHANGE WHICH MODEL HERE
model_name = stable_mods{5};

% model_path = '/home/nuttidalab/Documents/spikeRNN/models/xor/P_rec_0.2_Taus_4.0_25.0';
% mat_file = dir(fullfile(model_path, '*.mat'));
% model_name = mat_file(3).name; % to change which model

% make a folder for the model
cd(task_dir)
if ~exist(model_name(1:(end-4)), 'dir')
    mkdir(model_name(1:(end-4)))
end
model_dir_path = strcat(task_dir,'/',model_name(1:(end-4)));

cd(current_path)

file_path = fullfile(task_dir, model_name);
load(file_path);

% model_path = '/home/nuttidalab/Documents/spikeRNN/models/xor/P_rec_0.2_Taus_4.0_25.0';
% mat_file = dir(fullfile(model_path, '*.mat'));
% model_name = mat_file(1).name;
% file_path = fullfile(model_path, model_name);

use_initial_weights = false;
scaling_factor = opt_scaling_factor;
down_sample = 1;

% For LIF simulation, no stims
stims = struct();
stims.mode = 'none';

%% Plot 100 random trial outputs on longer delay period (evaluate model)

% some model / cycling parameters
n_trials = 100;
eval_amp_threshold = 0.7;
fs_rate = 200;
fs_spk = 20000;

T = 411;
stim_on = 31;
stim_dur = 50;
delay = 150; % longer delay

stim1_onset = (stim_on)/fs_rate*fs_spk;
stim1_offset = (stim_on + stim_dur)/fs_rate*fs_spk;
stim2_onset = (stim_on + stim_dur + delay)/fs_rate*fs_spk;
stim2_offset = (stim_on + 2*stim_dur + delay)/fs_rate*fs_spk;

% model eval on 100 random trials
eval_perf = zeros(n_trials,1);

figure(); hold on;
for i = 1:n_trials+1
    [eval_u, eval_label] = generate_input_stim_xor(T, stim_on, stim_dur, delay);
    [~, ~, ~, ~, ~, eval_o, ~] = LIF_network_fnc(file_path, scaling_factor,...
        eval_u, stims, down_sample, use_initial_weights);
    if eval_label == "same"
        if sum(unique(eval_u(1,:))) > 0
            plot(transpose(eval_o),'r') % +1/+1 red
        else
            plot(transpose(eval_o),'m') % -1/-1 pink
        end
        if max(eval_o(stim2_offset:end)) > eval_amp_threshold
            eval_perf(i) = 1;
        end
    else
        if sum(unique(eval_u(1,:))) > 0
            plot(transpose(eval_o), 'b')
        else
            plot(transpose(eval_o), 'c')
        end
        if min(eval_o(stim2_offset:end)) < -eval_amp_threshold
            eval_perf(i) = 1;
        end
    end
end
xline(stim1_onset, 'k--')
xline(stim1_offset, 'k--')
xline(stim2_onset, 'k--')
xline(stim2_offset, 'k--')

eval_perf_mean = mean(eval_perf);
title([model_name(end-20:end-4) ' Performance: ' num2str(eval_perf_mean,3)])
saveas(gcf,[model_dir_path '/output_delay150.png'])


%% Plot dynamics with biiiiig matrix (adapted from Robert code)
% old dynamics plots at bottom of this file

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

n_trials = 25; %size(outs, 2);

all_rs = zeros(n_trials*4, N, T*fs_spk/fs_rate);

stim1s = [-1, 1];
stim2s = [-1, 1];
counter = 0;
full_stims = zeros(4,2);
for ii = stim1s
    for jj = stim2s
        u = zeros(2, T+1); % input stim
        u(1, stim_on:stim_on+stim_dur) = ii; % first stim
        u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = jj; % second stim
        for i=1:n_trials
            [~, ~, ~, rs, ~, ~, ~] = LIF_network_fnc(file_path, scaling_factor,...
                u, stims, down_sample, use_initial_weights);
            all_rs(i+(counter*n_trials),:,:) = rs;
        end
        counter = counter + 1;
        full_stims(counter,:) = [ii, jj];
    end
end


% Make big matrix & PCA
time_cutoff = 51;
% time_cutoff = 201;
ax = squeeze(mean(all_rs(1:n_trials,   :, time_cutoff:end)));   % -1 -1 (ORANGE) NOW (MAGENTA)
ay = squeeze(mean(all_rs(n_trials+1:n_trials*2,  :, time_cutoff:end)));  % -1 1 (RED) NOW (CYAN)
bx = squeeze(mean(all_rs(n_trials*2+1:n_trials*3, :, time_cutoff:end)));  % 1 -1 (PURPLE) NOW (BLUE)
by = squeeze(mean(all_rs(n_trials*3+1:end, :, time_cutoff:end)));  % 1 1 (BLUE) NOW (RED)

%ax = ax(exc, :);
%ay = ay(exc, :);
%bx = bx(exc, :);
%by = by(exc, :);

trial_dur = size(ax, 2);

combined_data = [ax, ay, bx, by]; % neurons x time
%combined_data = [ax, ax_pv_reduced, bx, bx_pv_reduced];

W = pca(combined_data'); % neurons x 200 components

Z = combined_data'*W; % timexneurons * neuronxcomponents = time x 200 components
comps = [1, 2, 3];

pcs = Z(:,comps);
pcs = reshape(pcs, [],4,length(comps));

colors = ["magenta","cyan","blue","red"];
labels = ["-1/-1","-1/+1","+1/-1","+1/+1"];

% Plotting
figure; hold on; view(3);
for i = 1:4

    h = plot3(pcs(:,i,1)', pcs(:,i,2)', pcs(:,i,3)','color',colors(i), 'LineWidth',2); 
    set(h,{'DisplayName'},{labels(i)})
    h = scatter3(pcs(stim1_onset-time_cutoff,i,1), pcs(stim1_onset-time_cutoff,i,2), pcs(stim1_onset-time_cutoff,i,3), ...
        100, 'filled', colors(i),'<', 'SizeData',500);
    set(h,{'DisplayName'},{'stim 1 onset'})
    h = scatter3(pcs(stim2_onset-time_cutoff,i,1), pcs(stim2_onset-time_cutoff,i,2), pcs(stim2_onset-time_cutoff,i,3), ...
        100, 'filled', colors(i), 'SizeData',300);
    set(h,{'DisplayName'},{'stim 2 onset'})
    h = scatter3(pcs(end,i,1), pcs(end,i,2), pcs(end,i,3), ...
        100, 'filled', colors(i),'pentagram', 'SizeData',500);
    set(h,{'DisplayName'},{'end'})

end

legend show
% saveas(gcf,[model_dir_path '/trajectories_delay150_combinedpca.png'])



%% get IPSCs avg'd over 25 trials of each type

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

ds = 1;

% +1/+1
ipscs_samepos = zeros(N,T*100);
u = zeros(2, T+1); % input stim
u(1, stim_on:stim_on+stim_dur) = 1; % first stim is +1
u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = 1; % second stim is +1
for i=1:n_trials
    [~, ~, ~, ~, ~, ~, params] = LIF_network_fnc(file_path, scaling_factor,...
        u, stims, ds, use_initial_weights);
    ipscs_samepos = ipscs_samepos + params.IPSCs;
    clear params
end
ipscs_samepos = ipscs_samepos / n_trials;

% +1/-1
ipscs_diffpos = zeros(N,T*100);
u = zeros(2, T+1); % input stim
u(1, stim_on:stim_on+stim_dur) = 1; % first stim is +1
u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = -1; % second stim is -1
for i=1:n_trials
    [~, ~, ~, ~, ~, ~, params] = LIF_network_fnc(file_path, scaling_factor,...
        u, stims, ds, use_initial_weights);
    ipscs_diffpos = ipscs_diffpos + params.IPSCs;
    clear params
end
ipscs_diffpos = ipscs_diffpos / n_trials;

% -1/-1
ipscs_sameneg = zeros(N,T*100);
u = zeros(2, T+1); % input stim
u(1, stim_on:stim_on+stim_dur) = -1; % first stim is -1
u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = -1; % second stim is -1
for i=1:n_trials
    [~, ~, ~, ~, ~, ~, params] = LIF_network_fnc(file_path, scaling_factor,...
        u, stims, ds, use_initial_weights);
    ipscs_sameneg = ipscs_sameneg + params.IPSCs;
    clear params
end
ipscs_sameneg = ipscs_sameneg / n_trials;


% -1/+1
ipscs_diffneg = zeros(N, T*100);
u = zeros(2, T+1); % input stim
u(1, stim_on:stim_on+stim_dur) = -1; % first stim is -1
u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = -1; % second stim is -1
for i=1:n_trials
    [~, ~, ~, ~, ~, ~, params] = LIF_network_fnc(file_path, scaling_factor,...
        u, stims, ds, use_initial_weights);
    ipscs_diffneg = ipscs_diffneg + params.IPSCs;
    clear params
end
ipscs_diffneg = ipscs_diffneg / n_trials;


stim1_time = (stim_on)/200*20000;
stim2_time = (stim_on + stim_dur + delay)/200*20000;


save([model_dir_path,'/','IPSCs_25travg.mat'], 'ipscs_samepos','ipscs_diffpos', 'ipscs_sameneg','ipscs_diffneg', ...
    'T', 'stim_on', 'stim_dur', 'delay', 'stim1_time', 'stim2_time')


%% get 15 trials of each type and save

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

n_trials = 15;


% Run the LIF simulation on n_trials trials
diff_IPSCs = zeros([200,30000, n_trials]); % 200 neurons, 30k samples, n_trials


% +1/+1
ipscs_samepos = zeros(N,T*100,n_trials); % 200 neurons, 35k samples, n_trials
u = zeros(2, T+1); % input stim
u(1, stim_on:stim_on+stim_dur) = 1; % first stim is +1
u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = 1; % second stim is +1
for i=1:n_trials
    [~, ~, ~, ~, ~, ~, params] = LIF_network_fnc(file_path, scaling_factor,...
        u, stims, ds, use_initial_weights);
    ipscs_samepos(:,:,i) = params.IPSCs;
    clear params
end

% +1/-1
ipscs_diffpos = zeros(N,T*100,n_trials);
u = zeros(2, T+1); % input stim
u(1, stim_on:stim_on+stim_dur) = 1; % first stim is +1
u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = -1; % second stim is -1
for i=1:n_trials
    [~, ~, ~, ~, ~, ~, params] = LIF_network_fnc(file_path, scaling_factor,...
        u, stims, ds, use_initial_weights);
    ipscs_diffpos(:,:,i) = params.IPSCs;
    clear params
end

% -1/-1
ipscs_sameneg = zeros(N,T*100,n_trials);
u = zeros(2, T+1); % input stim
u(1, stim_on:stim_on+stim_dur) = -1; % first stim is -1
u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = -1; % second stim is -1
for i=1:n_trials
    [~, ~, ~, ~, ~, ~, params] = LIF_network_fnc(file_path, scaling_factor,...
        u, stims, ds, use_initial_weights);
    ipscs_sameneg(:,:,i) = params.IPSCs;
    clear params
end

% -1/+1
ipscs_diffneg = zeros(N,T*100,n_trials);
u = zeros(2, T+1); % input stim
u(1, stim_on:stim_on+stim_dur) = -1; % first stim is -1
u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = -1; % second stim is -1
for i=1:n_trials
    [~, ~, ~, ~, ~, ~, params] = LIF_network_fnc(file_path, scaling_factor,...
        u, stims, ds, use_initial_weights);
    ipscs_diffneg(:,:,i) = params.IPSCs;
    clear params
end


save([model_dir_path,'/','IPSCs_15trall.mat'], 'ipscs_samepos','ipscs_diffpos', 'ipscs_sameneg','ipscs_diffneg', ...
    'T', 'stim_on', 'stim_dur', 'delay')


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