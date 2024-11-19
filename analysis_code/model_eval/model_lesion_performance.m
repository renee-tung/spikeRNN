%% Plot behavior performance for I-I lesioned models

% here we differentiate performance depending on the trial types
% ie are models better at certain trial types compared to others

clear; clc;

% models_types = {'good_models';'bad_models'};
models_types = {'good_models'};
% lesion_connections = {0;'ii';'ee';'ei';'ie'};
lesion_connections = {0; 'ii'};
n_lesion_types = length(lesion_connections);

longer_delay = ''; % '' if standard delay (150), else a number (eg 200, 250)

for n_models_type = 1:length(models_types) % good and bad models
    models_type = models_types{n_models_type};

    clearvars -except models_types n_type models_type lesion_connections n_lesion_types longer_delay;

    current_path = pwd;

    task_dir = '/scratch/spikeRNN/models/DMS_OSF';

    save_name = [task_dir,'/',models_type, '/', models_type,'_trialtype_performance',num2str(longer_delay),'.mat'];

    % Load previously saved models in the group of interest
    stable_mods = load([task_dir,'/',models_type, '/', models_type,'_list']).stable_mods;

    eval_perf_mean_plus = zeros(length(stable_mods), n_lesion_types);
    eval_perf_mean_minus = zeros(length(stable_mods), n_lesion_types);
    eval_perf_mean_plus_all = zeros(length(stable_mods), n_lesion_types);
    eval_perf_mean_minus_all = zeros(length(stable_mods), n_lesion_types);
    for n_model = 1:length(stable_mods) % for each model
        fprintf('\n');
        disp([models_types{i}, ' number ', num2str(n_model), ' of ', num2str(length(stable_mods))]);

        % Get the model
        model_name = stable_mods{n_model};
        model_dir_path = fullfile(task_dir, model_name(1:end-4)); % Folder for model results

        file_path = fullfile(task_dir, model_name);
        load(file_path);

        % Different params for model function
        use_initial_weights = false;
        scaling_factor = opt_scaling_factor;
        down_sample = 1;
        stims = struct();
        stims.mode = 'none'; % For LIF simulation, no stims

        % Some model / cycling parameters
        n_trials = 200;
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
        stim2_offset = int64(stim2_offset);

        % Model eval on 200 random trials for each lesion type
        eval_perf = zeros(length(lesion_connections), n_trials); % Store models' performance
        trial_type = zeros(n_trials,1);

        for n_lesion = 1:n_lesion_types % For each lesion type
            parfor i = 1:n_trials % For n trials
                [eval_u, eval_label] = generate_input_stim_xor(T, stim_on, stim_dur, delay); % Random stimulus set
                first_stim = unique(eval_u(1,eval_u(1,:) ~= 0));
                trial_type(i) = first_stim;
                if ischar(lesion_connections{n_lesion}) % If one of the lesion trials
                    [~, ~, ~, ~, ~, eval_o, ~] = LIF_network_lesion_fnc(file_path, scaling_factor,...
                        eval_u, stims, down_sample, lesion_connections{n_lesion});
                else
                    [~, ~, ~, ~, ~, eval_o, ~] = LIF_network_fnc(file_path, scaling_factor,...
                        eval_u, stims, down_sample, use_initial_weights);
                end

                % Initialize trial performance
                trial_perf = 0;
                if eval_label == "same" % Check model performance
                    if max(eval_o(stim2_offset:end)) > eval_amp_threshold
                        trial_perf = 1;
                    end
                else
                    if min(eval_o(stim2_offset:end)) < -eval_amp_threshold
                        trial_perf = 1;
                    end
                end

                % Store the performance result
                eval_perf(n_lesion, i) = trial_perf;
            end

            % Display performance for the current lesion type
            mean_perf_plus = mean(eval_perf(n_lesion, trial_type > 0));
            mean_perf_minus = mean(eval_perf(n_lesion, trial_type < 0));
            disp([num2str(lesion_connections{n_lesion}), ' performance +1: ', num2str(mean_perf_plus)]);
            disp([num2str(lesion_connections{n_lesion}), ' performance -1: ', num2str(mean_perf_minus)]);
            eval_perf_mean_plus(n_model, n_lesion) = mean_perf_plus; % Save mean performance
            eval_perf_mean_minus(n_model, n_lesion) = mean_perf_minus; % Save mean performance
        end
    eval_perf_mean_plus_all(n_model, :) = eval_perf_mean_plus(n_model,:);
    eval_perf_mean_minus_all(n_model, :) = eval_perf_mean_minus(n_model,:);
    end
    eval_perf_mean_plus = eval_perf_mean_plus_all;
    eval_perf_mean_minus = eval_perf_mean_minus_all;
    save(save_name, 'eval_perf_mean_plus', 'eval_perf_mean_minus')
end


%% Plot behavior performance for I-I lesioned models - determine non-preferred stim

% here we differentiate performance depending on the trial types
% ie are models better at certain trial types compared to others

clear; clc;

models_type = 'good_models';
lesion_connections = {'ii'};
n_lesion_types = length(lesion_connections);

longer_delay = ''; % '' if standard delay (150), else a number (eg 200, 250)

n_repeats = 5;

task_dir = '/scratch/spikeRNN/models/DMS_OSF';
stable_mods = load([task_dir,'/',models_type, '/', models_type,'_list']).stable_mods;
% save_name = [task_dir,'/',models_type, '/', models_type,'_trialtype_performance',num2str(longer_delay),'.mat'];

pref_stim = ones(length(stable_mods),n_repeats);
for j = 1:n_repeats

    eval_perf_mean_plus = zeros(length(stable_mods), n_lesion_types);
    eval_perf_mean_minus = zeros(length(stable_mods), n_lesion_types);
    eval_perf_mean_plus_all = zeros(length(stable_mods), n_lesion_types);
    eval_perf_mean_minus_all = zeros(length(stable_mods), n_lesion_types);
    for n_model = 1:length(stable_mods) % for each model
        fprintf('\n');
        disp([models_type, ' number ', num2str(n_model), ' of ', num2str(length(stable_mods))]);

        % Get the model
        model_name = stable_mods{n_model};
        model_dir_path = fullfile(task_dir, model_name(1:end-4)); % Folder for model results

        file_path = fullfile(task_dir, model_name);
        load(file_path);

        % Different params for model function
        use_initial_weights = false;
        scaling_factor = opt_scaling_factor;
        down_sample = 1;
        stims = struct();
        stims.mode = 'none'; % For LIF simulation, no stims

        % Some model / cycling parameters
        n_trials = 200;
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
        stim2_offset = int64(stim2_offset);

        % Model eval on 200 random trials for each lesion type
        eval_perf = zeros(length(lesion_connections), n_trials); % Store models' performance
        trial_type = zeros(n_trials,1);

        for n_lesion = 1:n_lesion_types % For each lesion type
            parfor i = 1:n_trials % For n trials
                [eval_u, eval_label] = generate_input_stim_xor(T, stim_on, stim_dur, delay); % Random stimulus set
                first_stim = unique(eval_u(1,eval_u(1,:) ~= 0));
                trial_type(i) = first_stim;
                if ischar(lesion_connections{n_lesion}) % If one of the lesion trials
                    [~, ~, ~, ~, ~, eval_o, ~] = LIF_network_lesion_fnc(file_path, scaling_factor,...
                        eval_u, stims, down_sample, lesion_connections{n_lesion});
                else
                    [~, ~, ~, ~, ~, eval_o, ~] = LIF_network_fnc(file_path, scaling_factor,...
                        eval_u, stims, down_sample, use_initial_weights);
                end

                % Initialize trial performance
                trial_perf = 0;
                if eval_label == "same" % Check model performance
                    if max(eval_o(stim2_offset:end)) > eval_amp_threshold
                        trial_perf = 1;
                    end
                else
                    if min(eval_o(stim2_offset:end)) < -eval_amp_threshold
                        trial_perf = 1;
                    end
                end

                % Store the performance result
                eval_perf(n_lesion, i) = trial_perf;
            end

            % Display performance for the current lesion type
            mean_perf_plus = mean(eval_perf(n_lesion, trial_type > 0));
            mean_perf_minus = mean(eval_perf(n_lesion, trial_type < 0));
            disp([num2str(lesion_connections{n_lesion}), ' performance +1: ', num2str(mean_perf_plus)]);
            disp([num2str(lesion_connections{n_lesion}), ' performance -1: ', num2str(mean_perf_minus)]);
            eval_perf_mean_plus(n_model, n_lesion) = mean_perf_plus; % Save mean performance
            eval_perf_mean_minus(n_model, n_lesion) = mean_perf_minus; % Save mean performance
        end
        eval_perf_mean_plus_all(n_model, :) = eval_perf_mean_plus(n_model,:);
        eval_perf_mean_minus_all(n_model, :) = eval_perf_mean_minus(n_model,:);
    end
    eval_perf_mean_plus = eval_perf_mean_plus_all;
    eval_perf_mean_minus = eval_perf_mean_minus_all;
    pref_stim(eval_perf_mean_minus > eval_perf_mean_plus,j) = -1;
    % save(save_name, 'eval_perf_mean_plus', 'eval_perf_mean_minus')
end

all_pref_stim = pref_stim;

% save the stim preference to the model
clearvars -except all_pref_stim stable_mods task_dir
for n_model = 1:length(stable_mods) % for each model
    % Get the model
    model_name = stable_mods{n_model};

    file_path = fullfile(task_dir, model_name);
    load(file_path);

    if exist('pref_stim')
        clearvars -except task_dir n_model stable_mods all_pref_stim
        continue;
    else
        pref_stim = mode(all_pref_stim(n_model,:));
        save(file_path, 'pref_stim', '-append');
    end
end