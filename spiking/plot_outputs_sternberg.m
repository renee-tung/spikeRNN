clear; clc;

% Directory containing all the trained rate RNN model .mat files

model_dir = '/home/nuttidalab/Documents/renee/sternberg';
mat_files = dir(fullfile(model_dir, '*.mat'));

% model_dir = '/home/nuttidalab/Documents/renee/sternberg/singleload_1/';
% mat_files = dir(fullfile(model_dir, '*.mat'));

fs_rate = 200;
fs_spk = 20000;

% Whether to use the initial random connectivity weights
% This should be set to false unless you want to compare
% the effects of pre-trained vs post-trained weights
use_initial_weights = false;

% Number of trials to use to evaluate the LIF RNN
n_trials = 50;

T = 250;
stim_on = 50;
stim_dur = 25;
delay = 10;
match = 0; % random 50% chance in/out

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
    elseif ~isempty(findstr(curr_full, 'sternberg'))
        task_name = 'sternberg';
    end

    % Load the model
    load(curr_full);
    clear load

    if strcmpi(task_name, 'sternberg')
        disp(eval_perfs)
        if tr == 39999
            disp('Low training performance, skipping')
            continue
        end
    end


    % Sternberg task
    if strcmpi(task_name, 'sternberg')
        down_sample = 1;

        loads = [1, 3];
        % loads = [1];
        
        scaling_factor = opt_scaling_factor;
        disp(scaling_factor)
        for i_load = 1:length(loads)
            wm_load = loads(i_load);
            target_eval = int32((double(stim_on) + double(stim_dur)*double(wm_load) + ...
                double(delay) + double(stim_dur) + 10.0) / 200 * 20000);

            outs = zeros(n_trials, T*100);
            trials = zeros(n_trials, 1);
            perfs = zeros(n_trials, 1);

            parfor j = 1:n_trials
                [u, label] = generate_input_stim_sternberg(T, stim_on, stim_dur, delay, wm_load, match);

                trials(j) = label;

                stims = struct();
                stims.mode = 'none';
                [W, REC, spk, rs, all_fr, out, params] = LIF_network_fnc(curr_full, scaling_factor,...
                    u, stims, down_sample, use_initial_weights);
                outs(j, :) = out;
                if label == 1
                    if max(out(target_eval:end)) > 0.7
                        perfs(j) = 1;
                    end
                elseif label == -1
                    if min(out(target_eval:end)) < -0.7
                        perfs(j) = 1;
                    end
                end
            end % parfor end

            figure; hold on;
            for j = 1:n_trials
                if trials(j) == 1
                    plot(outs(j,:), 'r');
                else
                    plot(outs(j,:), 'b');
                end
            end
            title([mat_files(i).name(end-9:end-4), ' perf ', num2str(mean(perfs)), ' lambda ', num2str(opt_scaling_factor)]);
            xline(stim_on*100)
            for i_load = 1:wm_load
                xline((stim_on + stim_dur*i_load)*100);
            end
            xline((stim_on + stim_dur*i_load + delay)*100) % probe on
            xline((stim_on + stim_dur*i_load + delay + stim_dur)*100)

        end % load loop end

    end
end

        