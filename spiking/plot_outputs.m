clear; clc;

% Directory containing all the trained rate RNN model .mat files
% model_dir = '/home/nuttidalab/Documents/spikeRNN/models/xor/P_rec_0.2_Taus_4.0_25.0'; 
% model_dir = '/home/nuttidalab/Documents/renee/lfp_models/models/xor/P_rec_0.2_Taus_4.0_25.0';
model_dir = '/home/nuttidalab/Documents/renee/lfp_input_models_wtrain/models/xor/phase/P_rec_0.2_Taus_4.0_25.0';

mat_files = dir(fullfile(model_dir, '*8_25*.mat'));

% Whether to use the initial random connectivity weights
% This should be set to false unless you want to compare
% the effects of pre-trained vs post-trained weights
use_initial_weights = false; 

% Number of trials to use to evaluate the LIF RNN
n_trials = 100;

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


  % XOR task
  if strcmpi(task_name, 'xor')
    down_sample = 1;

    scaling_factor = opt_scaling_factor;
      outs = zeros(n_trials, 30000);
      trials = zeros(n_trials, 1);
      perfs = zeros(n_trials, 1);

      
      disp(scaling_factor)
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
          u(2, 111:160) = 1;
          u_lab(2) = 1;
        else
          u(2, 111:160) = -1;
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
          if max(out(20000:end)) > 0.7
            perfs(j) = 1;
          end
        elseif label == -1
          if min(out(20000:end)) < -0.7
            perfs(j) = 1;
          end
        end
      end % parfor end
      
      figure; plot(outs');title([mat_files(i).name(1:end-4), ' perf ',num2str(mean(perfs))])
      

  end
end