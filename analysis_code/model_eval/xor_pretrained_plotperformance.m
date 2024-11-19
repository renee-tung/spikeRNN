%% Plot behavior performance for pretrained models on xor task

clear; clc;

models_types = {'good_models';'bad_models'};
lesion_connections = {0;'ii';'ee';'ei';'ie'};
n_lesion_types = length(lesion_connections);

longer_delay = 400; % '' if standard delay (150), else a number (eg 200, 250)

for n_type = 1:length(models_types) % good and bad models
    models_type = models_types{n_type};

    clearvars -except models_types n_type models_type lesion_connections n_lesion_types longer_delay;

    current_path = pwd;

    task_dir = '/home/nuttidalab/Documents/spikeRNN/models/DMS_OSF';

    save_name = [task_dir,'/',models_type, '/', models_type,'_performance',num2str(longer_delay),'.mat'];

    % Load previously saved models in the group of interest
    stable_mods = load([task_dir,'/',models_type, '/', models_type,'_list']).stable_mods;

    eval_perf_mean = zeros(length(stable_mods), n_lesion_types);
    eval_perf_mean_all = zeros(length(stable_mods), n_lesion_types);
    for n_model = 1:length(stable_mods) % for each model
        fprintf('\n');
        disp([models_types{n_type}, ' number ', num2str(n_model), ' of ', num2str(length(stable_mods))]);

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
        stim2_offset = int64(stim2_offset);

        % Model eval on 100 random trials for each lesion type
        eval_perf = zeros(length(lesion_connections), n_trials); % Store models' performance

        for n_lesion = 1:n_lesion_types % For each lesion type
            parfor i = 1:n_trials % For 100 trials
                [eval_u, eval_label] = generate_input_stim_xor(T, stim_on, stim_dur, delay); % Random stimulus set
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
            mean_perf = mean(eval_perf(n_lesion, :));
            disp([num2str(lesion_connections{n_lesion}), ' performance: ', num2str(mean_perf)]);
            eval_perf_mean(n_model, n_lesion) = mean_perf; % Save mean performance
        end
    eval_perf_mean_all(n_model, :) = eval_perf_mean(n_model,:);
    end
    eval_perf_mean = eval_perf_mean_all;
    save(save_name, 'eval_perf_mean')
end

