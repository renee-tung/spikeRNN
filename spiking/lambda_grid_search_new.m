%% newer version of lambda_grid_search
% adding a constraint that the task output cannot be both greater than 0.7
% and less than -0.7 to be correct.


clear; clc;

% Directory containing all the trained rate RNN model .mat files
% model_dir = '/home/nuttidalab/Documents/renee/all_DMS_models'; 
model_dir = '/home/nuttidalab/Documents/renee/jitter_models/models/xor/P_rec_0.2_Taus_4.0_25.0/';

% mat_files = dir(fullfile(model_dir, '*.mat'));
mat_files = dir(fullfile(model_dir, '*Jitter_10_10*.mat'));


% Whether to use the initial random connectivity weights
% This should be set to false unless you want to compare
% the effects of pre-trained vs post-trained weights
use_initial_weights = false; 

% Number of trials to use to evaluate the LIF RNN
n_trials = 100;

% Scaling factor values to try for grid search
% The more values it has, the longer the search
scaling_factors = [20:5:75];
% scaling_factors = [1:5:30];

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
  end

  % Load the model
  load(curr_full);

  disp(['prev opt_scaling_factor: ', num2str(opt_scaling_factor), ', perf: ', num2str(max(all_perfs))])
  % Skip if the file was run before
  % if exist('opt_scaling_factor_new') && ~isnan(opt_scaling_factor_new)
  %   clearvars -except model_dir mat_files n_trials scaling_factors use_initial_weights input_freq
  %   continue;
  % else
  %   opt_scaling_factor_new = NaN;
  %   save(curr_full, 'opt_scaling_factor_new', '-append');
  % end

  figure;
  % Go-NoGo task
  if strcmpi(task_name, 'go-nogo')
    down_sample = 1;
    all_perfs_new = zeros(length(scaling_factors), 1);

    for k = 1:length(scaling_factors)
      outs = zeros(n_trials, 20000);
      trials = zeros(n_trials, 1);
      perfs = zeros(n_trials, 1);

      scaling_factor = scaling_factors(k);
      disp(scaling_factor)

      parfor j = 1:n_trials
        u = zeros(1, 201);
        if rand >= 0.50
          u(51:75) = 1.0;
          trials(j) = 1;
        end
        stims = struct();
        stims.mode = 'none';
        [W, REC, spk, rs, all_fr, out, params] = LIF_network_fnc(curr_full, scaling_factor,...
        u, stims, down_sample, use_initial_weights);
        outs(j, :) = out;
        if max(out(10000:end)) > 0.7 & trials(j) == 1
          perfs(j) = 1;
        elseif max(out(10000:end)) < 0.3 & trials(j) == 0
          perfs(j) = 1;
        end
      end
      all_perfs(k) = mean(perfs);
    
    % subplot(3, 4, k); hold on;
    %   plot(outs(trials == 1,:)', 'Color', [1, 0, 0, 0.5]);
    %   plot(outs(trials == -1,:)', 'Color', [0, 0, 1, 0.5]);
    %   title(['scaling factor ', num2str(scaling_factors(k))])
    % 
    end
    [v, ind] = max(all_perfs);
    [v, scaling_factors(ind)]


    % sgtitle(['task load 3, model with 400 neurons, optimal scaling factor ', ...
    %     'old: ', num2str(opt_scaling_factor), ', new: ', num2str(scaling_factors(ind))])


    % Save the optimal scaling factor
    opt_scaling_factor_new = scaling_factors(ind);
    save(curr_full, 'opt_scaling_factor_new', 'all_perfs_new', '-append');
    clear opt_scaling_factor_new;

  % XOR task
  elseif strcmpi(task_name, 'xor')
    down_sample = 1;
    all_perfs_new = zeros(length(scaling_factors), 1);

    for k = 1:length(scaling_factors)
      outs = zeros(n_trials, 30000);
      trials = zeros(n_trials, 1);
      perfs = zeros(n_trials, 1);

      scaling_factor = scaling_factors(k);
      % disp(scaling_factor)
      parfor j = 1:n_trials
        u = zeros(2, 301);
        u_lab = zeros(1, 2);

        % Stim 1
        if rand >= 0.50
          u(1, 51:100) = 1;
          u_lab(1) = 1;
        else
          u(1, 51:100) = -1;
          u_lab(1) = -1;
        end

        % Stim 2
        if rand >= 0.50
          u(2, 151:200) = 1;
          u_lab(2) = 1;
        else
          u(2, 151:200) = -1;
          u_lab(2) = -1;
        end
        label = prod(u_lab);
        trials(j) = label;


        stims = struct();
        stims.mode = 'none';
        [W, REC, spk, rs, all_fr, out, params] = LIF_network_fnc(curr_full, scaling_factor,...
            u, stims, down_sample, use_initial_weights);
        outs(j, :) = out;
        if label == 1
          if max(out(20000:end)) > 0.7 && min(out(20000:end)) > -0.7
            perfs(j) = 1;
          end
        elseif label == -1
          if min(out(20000:end)) < -0.7 && max(out(20000:end)) < 0.7
            perfs(j) = 1;
          end
        end
      end % parfor end
      all_perfs_new(k) = mean(perfs);

      subplot(3, 4, k); hold on;
      plot(outs(trials == 1,:)', 'Color', [1, 0, 0, 0.5]);
      plot(outs(trials == -1,:)', 'Color', [0, 0, 1, 0.5]);
      title(['scaling factor ', num2str(scaling_factors(k))])
    
    end
    [v, ind] = max(all_perfs_new);
    [v, scaling_factors(ind)]

    sgtitle([curr_fname(end-10:end-4), ', perf ', num2str(v), ', lambda ', ...
       num2str(scaling_factors(ind))])

    % sgtitle([curr_fname, ', perf ', num2str(v), ', optimal scaling factor ', ...
    %     'old: ', num2str(opt_scaling_factor), ', new: ', num2str(scaling_factors(ind))])

    % Save the optimal scaling factor
    opt_scaling_factor_new = scaling_factors(ind);
    if i==3
        break
    end
    % save(curr_full, 'opt_scaling_factor_new', 'all_perfs_new', '-append');
    clearvars -except model_dir mat_files n_trials scaling_factors use_initial_weights input_freq
  end
end