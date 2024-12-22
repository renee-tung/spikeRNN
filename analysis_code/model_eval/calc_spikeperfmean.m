%% Calculate spike_perf_mean for each model

% for the letters task, calculate the spiking model performance on the
% training task (delay=10)

clear; clc;

% Directory containing all the trained rate RNN model .mat files
task_load = 2;
model_dir = ['/home/nuttidalab/Documents/spikeRNN/models/letters/load_', num2str(task_load), '/'];

mat_files = dir(fullfile(model_dir, '*.mat'));

% Whether to use the initial random connectivity weights
% This should be set to false unless you want to compare
% the effects of pre-trained vs post-trained weights
use_initial_weights = false; 

% Number of trials to use to evaluate the LIF RNN
n_trials = 100;

% Set the delay for the model eval
delay = 10;

% Grid search
for i = 1:length(mat_files)
    curr_fname = mat_files(i).name;
    curr_full = fullfile(mat_files(i).folder, curr_fname);
    disp(['Analyzing ' curr_fname]);

    % Get the task name
    if ~isempty(findstr(curr_full, 'go-nogo'))
        task_name = 'go-nogo';
    elseif ~isempty(findstr(curr_full, 'mante'))
        task_name = 'mante';
    elseif ~isempty(findstr(curr_full, 'xor'))
        task_name = 'xor';
    elseif ~isempty(findstr(curr_full, 'letters'))
        task_name = 'letters';
    end

    % Load the model
    load(curr_full);

    % skip if already calculated
    if exist('spike_perf_mean_smooth')
        clearvars -except model_dir mat_files n_trials use_initial_weights task_load delay
        continue;
    end

    % letters task
    if strcmpi(task_name, 'letters')
        down_sample = 1;

        trials = zeros(n_trials, 1);
        stable_perfs = zeros(n_trials, 1);

        stim_on = 51;
        stim_dur = 100;
        stim2_dur = 130;
        response_time = stim_on + stim_dur + delay + 10;
        T = response_time + stim2_dur;

        n_input_chans = task_load*2;

        parfor j = 1:n_trials
            u = zeros(n_input_chans, T);
            u_lab = zeros(1, 2);

            letters = 1:n_input_chans; % load*2 letter choices
            stim_letters = randperm(n_input_chans, task_load); % load letter choices
            probe_letter = randperm(n_input_chans, 1); % 1 letter choice

            u(stim_letters, stim_on:stim_on+stim_dur) = 1; % stimulus presentation
            u(probe_letter, stim_on+stim_dur+delay:end) = 1; % probe presentation

            label = 2*(ismember(probe_letter, stim_letters)) - 1;
            trials(j) = label;

            stims = struct();
            stims.mode = 'none';
            use_smoothing=true;
            [W, REC, spk, rs, all_fr, out, params] = LIF_network_fnc(curr_full, opt_scaling_factor,...
                u, stims, down_sample, use_initial_weights, use_smoothing);

            if label == 1
                if max(out(response_time*100:end)) > 0.7
                    stable_perfs(j) = 1;
                end
            elseif label == -1
                if min(out(response_time*100:end)) < -0.7
                    stable_perfs(j) = 1;
                end
            end
        end % parfor end

        % Save the perfs
        disp(mean(stable_perfs))
        spike_perf_mean_smooth = mean(stable_perfs);
        save(curr_full, 'spike_perf_mean_smooth', '-append');
        % save(curr_full, 'stable_perfs', '-append');   
        clearvars -except model_dir mat_files n_trials use_initial_weights task_load delay
    else
        disp('task not implemented');
    end
end


