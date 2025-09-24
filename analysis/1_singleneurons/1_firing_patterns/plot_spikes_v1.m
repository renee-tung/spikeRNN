%% plot_spikes

% This script is for plotting neuron spikes for correct versus incorrect trials. 

% v1 is for a single model

clear; clc;

% add path
addpath('/home/nuttidalab/Documents/spikeRNN/spiking/')

%% Get the model we want to look at

models_types = {'good_models';'bad_models'};
lesion_connections = {0}; %{0;'ii';'ee';'ei';'ie'};
n_lesion_types = length(lesion_connections);
longer_delay = ''; % '' if standard delay (150), else a number (eg 200, 250)

% for n_type = 1:length(models_types) % good and bad models
n_type = 2; % bad models
models_type = models_types{n_type};

clearvars -except models_types n_type models_type lesion_connections n_lesion_types longer_delay;

current_path = pwd;
task_dir = '/scratch/spikeRNN/models/DMS_OSF';
stable_mods = load([task_dir,'/',models_type, '/', models_type,'_list']).stable_mods; % Load previously saved models in the group of interest

n_model = 1;
fprintf('\n');
disp([models_types{n_type}, ' number ', num2str(n_model), ' of ', num2str(length(stable_mods))]);

% Get the model
model_name = stable_mods{n_model};
model_dir_path = fullfile(task_dir, model_name(1:end-4)); % Folder for model results

file_path = fullfile(task_dir, model_name);
load(file_path);


%% Set the params for LIF function

% Different params for model function
use_initial_weights = false;
scaling_factor = opt_scaling_factor;
down_sample = 1;
stims = struct();
stims.mode = 'none'; % For LIF simulation, no stims

%% Timing info for the trials we want to generate

% Some model / cycling parameters
n_trials = 100;
eval_amp_threshold = 0.7;
fs_rate = 200;
fs_spk = 20000;

% Time settings
stim_on = 31;
stim_dur = 50;
if isempty(longer_delay)
    delay = 150; % Standard delay
else
    delay = longer_delay;
end
T = 261 + delay;

stim1_onset = (stim_on) / fs_rate * fs_spk;
stim1_offset = (stim_on + stim_dur) / fs_rate * fs_spk;
stim2_onset = (stim_on + stim_dur + delay) / fs_rate * fs_spk;
stim2_offset = (stim_on + 2 * stim_dur + delay) / fs_rate * fs_spk;


%% Generate trials, parfor (not used here)

% Model eval on 100 random trials for each lesion type
eval_perf = zeros(n_trials,1); % Store models' performance
o_array = zeros(n_trials, 41100);
label_array = zeros(n_trials,1);
parfor i = 1:n_trials % For 100 trials
    [u, label] = generate_input_stim_xor(T, stim_on, stim_dur, delay); % Random stimulus set
    % [eval_u, eval_label] = generate_specific_input_stim_xor(T, stim1, stim2, stim_on, stim_dur, delay);
    
    [~, ~, spk_train, rates, ~, outputs, params] = LIF_network_fnc(file_path, scaling_factor,...
        u, stims, down_sample, use_initial_weights); % [W, REC, spk, rs, all_fr, out, params]

    % Initialize trial performance
    trial_perf = 0;
    if label == "same" % Check model performance
        if max(outputs(stim2_offset:end)) > eval_amp_threshold
            trial_perf = 1;
        end
    else
        if min(outputs(stim2_offset:end)) < -eval_amp_threshold
            trial_perf = 1;
        end
    end

    % Store the performance result
    eval_perf(i) = trial_perf;
    label_array(i) = label == "same";
    o_array(i,:) = outputs;
end






%% Generate some individual trials and plot example trials

inh_ind = find(inh);
exc_ind = find(exc);
all_ind = [exc_ind; inh_ind];
dt = params.dt;
t = dt:dt:T;

n_trials = 10;

for i = 1:n_trials % For 100 trials
    [u, label] = generate_input_stim_xor(T, stim_on, stim_dur, delay); % Random stimulus set
    % [eval_u, eval_label] = generate_specific_input_stim_xor(T, stim1, stim2, stim_on, stim_dur, delay);
    
    [~, ~, spk_train, rates, ~, outputs, params] = LIF_network_fnc(file_path, scaling_factor,...
        u, stims, down_sample, use_initial_weights); % [W, REC, spk, rs, all_fr, out, params]

    % Initialize trial performance
    trial_perf = 0;
    if label == "same" % Check model performance
        if max(outputs(stim2_offset:end)) > eval_amp_threshold
            trial_perf = 1;
        end
    else
        if min(outputs(stim2_offset:end)) < -eval_amp_threshold
            trial_perf = 1;
        end
    end

    figure('Units', 'Normalized', 'Outerposition', [0 0 0.22 0.20]);
    hold on; axis tight;
    
    % all_ind = 1:N;
    for j = 1:length(all_ind)
        curr_spk = spk_train(all_ind(j), 10:end);
        if exc(all_ind(j)) == 1
            plot(t(find(curr_spk)), ones(1, length(find(curr_spk)))*j, 'r.', 'markers', 8);
        else
            plot(t(find(curr_spk)), ones(1, length(find(curr_spk)))*j, 'b.', 'markers', 8);
        end
    end
    
    for k = 1:size(u,1)
        stim_identity = unique(u(k,:));
        stim_identity = stim_identity(stim_identity~=0);
        if stim_identity == 1
            stimcolor = 'r';
        elseif stim_identity == -1
            stimcolor= 'b';
        end
        if k == 1
            xregion(stim1_onset/fs_spk, stim1_offset/fs_spk, FaceColor=stimcolor, FaceAlpha=0.3)
        elseif k==2
            xregion(stim2_onset/fs_spk, stim2_offset/fs_spk, FaceColor=stimcolor, FaceAlpha=0.3)
        end
    end
    
    xlim([0, 2.1]);
    ylim([-5, 205]);

    title(['Performance: ',num2str(trial_perf)])

end


%% check if spikes and rates match up -- they do

% Plot some example rates
t = 0:1/fs_spk*fs_rate:T;
t = t(10:end-1);
for j = 150:160
    figure('Units', 'Normalized', 'Outerposition', [0 0 0.22 0.20]);
    hold on; axis tight;
    curr_rs = rates(all_ind(j), 10:end);
    if exc(all_ind(j)) == 1
        plot(t, curr_rs/max(curr_rs), 'r');
    else
        plot(t, curr_rs/max(curr_rs), 'b');
    end
end

% Plot the spikes
figure('Units', 'Normalized', 'Outerposition', [0 0 0.22 0.20]);
hold on; axis tight;

for j = 1:length(all_ind)
    curr_spk = spk_train(all_ind(j), 10:end);
    if exc(all_ind(j)) == 1
        plot(t(find(curr_spk)), ones(1, length(find(curr_spk)))*j, 'r.', 'markers', 8);
    else
        plot(t(find(curr_spk)), ones(1, length(find(curr_spk)))*j, 'b.', 'markers', 8);
    end
end

for k = 1:size(u,1)
    stim_identity = unique(u(k,:));
    stim_identity = stim_identity(stim_identity~=0);
    if stim_identity == 1
        stimcolor = 'r';
    elseif stim_identity == -1
        stimcolor= 'b';
    end
    if k == 1
        xregion(stim1_onset/fs_spk, stim1_offset/fs_spk, FaceColor=stimcolor, FaceAlpha=0.3)
    elseif k==2
        xregion(stim2_onset/fs_spk, stim2_offset/fs_spk, FaceColor=stimcolor, FaceAlpha=0.3)
    end
end

xlim([0, 2.1]);
ylim([-5, 205]);

title(['Performance: ',num2str(trial_perf)])


%% Calculate average firing rates for each neuron

% Model eval on 100 random trials for each lesion type
eval_perf = zeros(n_trials,1); % Store models' performance
rate_array = zeros(n_trials, N, 41100);
label_array = zeros(n_trials,1);
parfor i = 1:n_trials % For 100 trials
    [u, label] = generate_input_stim_xor(T, stim_on, stim_dur, delay); % Random stimulus set
    % [eval_u, eval_label] = generate_specific_input_stim_xor(T, stim1, stim2, stim_on, stim_dur, delay);
    
    [~, ~, spk_train, rates, ~, outputs, params] = LIF_network_fnc(file_path, scaling_factor,...
        u, stims, down_sample, use_initial_weights); % [W, REC, spk, rs, all_fr, out, params]

    % Initialize trial performance
    trial_perf = 0;
    if label == "same" % Check model performance
        if max(outputs(stim2_offset:end)) > eval_amp_threshold
            trial_perf = 1;
        end
    else
        if min(outputs(stim2_offset:end)) < -eval_amp_threshold
            trial_perf = 1;
        end
    end

    % Store the performance result
    eval_perf(i) = trial_perf;
    label_array(i) = label == "same";
    rate_array(i,:,:) = rates;
end


%% Plot average firing rates for each neuron

for i = 1:5
    figure(); hold on; axis tight;
    for j = 1:40
        subplot(5,8,j)
        idx = (i-1)*40+j;
        if exc(all_ind(idx)) == 1
            plotcolor='r';
        else
            plotcolor='b';
        end
        plot(rate_array(all_ind(idx),:), plotcolor)
        set(gca,'XTick',[])
        set(gca,'YTick',[])

        for k = 1:size(u,1)
            stim_identity = unique(u(k,:));
            stim_identity = stim_identity(stim_identity~=0);
            if stim_identity == 1
                stimcolor = 'r';
            elseif stim_identity == -1
                stimcolor= 'b';
            end
            if k == 1
                xregion(stim1_onset, stim1_offset, FaceColor=stimcolor, FaceAlpha=0.3)
            elseif k==2
                xregion(stim2_onset, stim2_offset, FaceColor=stimcolor, FaceAlpha=0.3)
            end
        end

    end
end






%%

if n_type == 1 % 1 is good
    plotcolor1 = '#49BEA3';
    plotcolor2 = '#236975';

    % plotcolor1 = [0,0.478,0.478,0.5];%"#007a7a";
    % plotcolor2 = [0,0.278, 0.278,0.5];%"#004747";
elseif n_type == 2 % 2 is bad
    plotcolor1 = '#6E439A';
    plotcolor2 = '#2B1644';

    % plotcolor1 = [0.478,0,0.478,0.5];%"#7a007a";
    % plotcolor2 = [0.278,0,0.278,0.5]; %"#470047";
end

fs_ds = 200;
o_array = downsample_signal(fs_spk, fs_ds, o_array);

figure('Position',[10 10 900 600]); hold on;
plot(o_array(label_array == 1,:)', 'Color', plotcolor1);
plot(o_array(label_array == 0,:)', 'Color', plotcolor2);
xline(stim1_onset/fs_spk*fs_ds, 'Color','k','LineWidth',2)
xline(stim1_offset/fs_spk*fs_ds, 'Color','k','LineWidth',2)
xline(stim2_onset/fs_spk*fs_ds, 'Color','k','LineWidth',2)
xline(stim2_offset/fs_spk*fs_ds, 'Color','k','LineWidth',2)
title(['High-performing model, mean performance: ', num2str(mean(eval_perf))])
xlabel('Time (ms)')
x = 0:fs_ds/2:length(o_array);
xticks(x)
xticklabels(x/fs_ds*1000)
ylabel('Output')
fontsize(16,"points")
ylim([-3 3])


% eval_perf_mean = eval_perf_mean_all;
% save(save_name, 'eval_perf_mean')
