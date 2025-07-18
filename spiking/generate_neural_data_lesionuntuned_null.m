%% GENERATE DATA

% Script to generate and save trials of DMS data to computer for further
% analysis


% Will include:

% spike times
% LFP
% trial labels
% trial performance
% trial timing


clear; clc

%% some params for this data

normalize_ipscs = 0;
lesion_connections = 'untunedpost'; % if not lesioning put 0, else 'ii' etc
delay = 150; % 150 is the standard "testing" delay duration
n_trials_per_condn = 250;


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


%% add path
addpath('/home/nuttidalab/Documents/renee/spikeRNN/spiking/')
% addpath('/home/nuttidalab/Documents/spikeRNN/spiking/')


%% directory info

% all_model_path = '/scratch/all_DMS_models/';
all_model_path = '/home/nuttidalab/Documents/renee/all_DMS_models/';
cd(all_model_path)
model_list = dir('*.mat');


%% curate the model list

% here, i want to "lesion" untuned neurons in good and bad models
good_model_list = {};
bad_model_list = {};
for n_model = 1:length(model_list)
    
    model_name = model_list(n_model).name(1:end-4);
    this_model_dir = fullfile(all_model_path,model_name);
    bhv_save_name = [this_model_dir,'/','bhvdata',norm_name,'','_delay',num2str(delay),'.mat']; % load unlesioned performance
    load(bhv_save_name)
    mean_perf = mean(all_trial_perfs);

    if mean_perf > 0.95
        good_model_list{end+1} = model_name;
    elseif mean_perf >=0.60 && mean_perf <= 0.80
        bad_model_list{end+1} = model_name;
    end

end



%% Timing info for the trials we want to generate

% Some model / cycling parameters
eval_amp_threshold = 0.7;
fs_rate = 200;
fs_spk = 20000;
fs_ds = 1000; % downsample frequency for LFP data

% Time settings
stim_on = 31;
stim_dur = 50;
% delay was defined earlier
T = 261 + delay;

stim1_onset = (stim_on) / fs_rate * fs_spk;
stim1_offset = (stim_on + stim_dur) / fs_rate * fs_spk;
stim2_onset = (stim_on + stim_dur + delay) / fs_rate * fs_spk;
stim2_offset = (stim_on + 2 * stim_dur + delay) / fs_rate * fs_spk;


%% Set the params for LIF function

% Different params for model function
use_initial_weights = false;
down_sample = 1;
stims = struct();
stims.mode = 'none'; % For LIF simulation, no stims

%% Trial stimuli

stim1s = [-1,1];
stim2s = [-1,1];

n_trials_total = length(stim1s) * length(stim2s) * n_trials_per_condn;

% counter = 0;
% full_stims = zeros(4,2);
% for ii = stim1s
%     for jj = stim2s
%         counter = counter + 1;
%         full_stims(counter,:) = [ii, jj];
%     end
% end


%% cycle through models and generate data

model_lists = {good_model_list; bad_model_list};
for i_list = 1:2
    model_list = model_lists{i_list};
for n_model = 1:length(model_list)
    fprintf('\n');
    disp(['model number ',num2str(n_model), ' of ', num2str(length(model_list))])

    % get model name and load model
    model_name = model_list{n_model};
    disp(model_name)
    model_path = fullfile(all_model_path, [model_name,'.mat']);
    load(model_path)
    model_path = fullfile(all_model_path, [model_name,'.mat']); % rewrite bc was overwritten
    disp(['mean stable performance ', num2str(mean(stable_perfs))])

    % make directory for this model if one doesn't exist
    if ~exist(model_name, 'dir')
        mkdir(model_name)
    end
    this_model_dir = fullfile(all_model_path,model_name);

    % save name for the output
    neural_save_name = [this_model_dir,'/','neuraldata',norm_name,lesion_name,'_delay',num2str(delay),'.mat'];
    bhv_save_name = [this_model_dir,'/','bhvdata',norm_name,lesion_name,'_delay',num2str(delay),'.mat'];
    timing_save_name = [this_model_dir,'/','timingdata',norm_name,lesion_name,'_delay',num2str(delay),'.mat'];

    % check if there is data calculated already
    if exist(neural_save_name, 'file') > 0
        disp('already calculated, moving to next model...')
        continue
    end

    % get scaling param for this model
    scaling_factor = opt_scaling_factor;

    % get untuned neuron idxs for this model
    load([this_model_dir,'/','tuning_delay_',num2str(delay),'.mat']);
    untuned_idx = untuned_idx + 1; % change from python to matlab indexing
    if isempty(untuned_idx)
        disp('no untuned neurons, skipping this model')
        continue
    end

    % generate trials
    counter = 0;
    
    matObj = matfile(neural_save_name, 'Writable',true);

    all_spk_times(n_trials_total) = struct('spk_times', []);
    matObj.all_rates = zeros(n_trials_total, N, T/fs_rate*fs_ds); % trials x neurons x time
    matObj.all_lfp = zeros(n_trials_total, N, T/fs_rate*fs_ds); % trials x neurons x time
    % all_rates = zeros(n_trials_total, N, T/fs_rate*fs_ds); % trials x neurons x time
    % all_lfp = zeros(n_trials_total, N, T/fs_rate*fs_ds); % trials x neurons x time
    all_trial_labels = zeros(n_trials_total,2);
    all_trial_perfs = zeros(n_trials_total,1);
    
    for ii = stim1s
        for jj = stim2s
            [u, label] = generate_specific_input_stim_xor(T, ii, jj, stim_on, stim_dur, delay);
            this_spk_times(n_trials_per_condn) = struct();
            this_lfp = zeros(n_trials_per_condn, N, T/fs_rate*fs_ds);
            this_rates = zeros(n_trials_per_condn, N, T/fs_rate*fs_ds);
            this_trial_perfs = zeros(n_trials_per_condn,1);
            parfor n_trial=1:n_trials_per_condn
                % [~, ~, spk_train, rates, ~, outputs, params] = LIF_network_fnc(model_path, scaling_factor,...
                %     u, stims, down_sample, use_initial_weights);
                % [~, ~, spk_train, rates, ~, outputs, params] = LIF_network_lesion_neurons_fnc(model_path, scaling_factor,...
                %     u, stims, down_sample, untuned_idx, "presynaptic");
                [~, ~, spk_train, rates, ~, outputs, params] = LIF_network_lesion_neurons_fnc(model_path, scaling_factor,...
                    u, stims, down_sample, untuned_idx, "postsynaptic");
                
                % get spike times, convert to 0-indexing, and store
                spk_times = arrayfun(@(i) find(spk_train(i,:) ~= 0)-1, (1:size(spk_train,1))', 'UniformOutput', false);
                this_spk_times(n_trial).spk_times = spk_times;

                % get rates, downsample, and store
                this_rates(n_trial,:,:) = downsample_signal(fs_spk, fs_ds, rates);

                % get LFP, downsample, and store
                this_lfp(n_trial,:,:) = downsample_signal(fs_spk, fs_ds, params.IPSCs);
                
                % get performance on this trial
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
                this_trial_perfs(n_trial) = trial_perf;

            end

            % store data in the larger "all" variables
            start_idx = counter*n_trials_per_condn+1;
            end_idx = counter*n_trials_per_condn+n_trials_per_condn;
            all_spk_times(start_idx:end_idx) = this_spk_times;
            matObj.all_rates(start_idx:end_idx,:,:) = single(this_rates);
            matObj.all_lfp(start_idx:end_idx,:,:) = single(this_lfp);
            % all_rates(start_idx:end_idx,:,:) = this_rates;
            % all_lfp(start_idx:end_idx,:,:) = this_lfp;
            all_trial_labels(counter*n_trials_per_condn+1:counter*n_trials_per_condn+n_trials_per_condn,:) = repmat([ii,jj],n_trials_per_condn,1);
            all_trial_perfs(start_idx:end_idx) = this_trial_perfs;

            clear this_spk_times this_rates this_lfp this_trial_perfs params

            counter = counter + 1;
            
        end
    end

    % save the data
    save(neural_save_name, '-append', 'all_spk_times', '-v7.3')
    % save(neural_save_name, 'all_spk_times','all_lfp','-v7.3') % neural
    save(bhv_save_name, 'all_trial_labels','all_trial_perfs') % behavioral
    save(timing_save_name, 'stim_on','stim_dur','delay','T','fs_spk','fs_rate','fs_ds') % timing


    % clear the variables
    clear all_spk_times all_lfp all_trial_labels all_trial_perfs




end

end