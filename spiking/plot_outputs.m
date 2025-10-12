clear; clc;

% Directory containing all the trained rate RNN model .mat files
% model_dir = '/home/nuttidalab/Documents/spikeRNN/models/xor/P_rec_0.2_Taus_4.0_25.0'; 
% model_dir = '/home/nuttidalab/Documents/renee/lfp_models/models/xor/P_rec_0.2_Taus_4.0_25.0';
model_dir = '/home/nuttidalab/Documents/renee/lfp_input_models_wtrain/models/xor/phase/P_rec_0.2_Taus_4.0_25.0';
cd(model_dir)
mat_files = dir(fullfile(model_dir, '*4.0*.mat'));

stim_on = 51;
stim_dur = 50;
delay = 400;
T = 251 + delay;

fs_rate = 200;
fs_spk = 20000;

stim1_onset = (stim_on) / fs_rate * fs_spk;
stim1_offset = (stim_on + stim_dur) / fs_rate * fs_spk;
stim2_onset = (stim_on + stim_dur + delay) / fs_rate * fs_spk;
stim2_offset = (stim_on + 2 * stim_dur + delay) / fs_rate * fs_spk;
response_end = min((stim_on + 2 * stim_dur + delay + 10 + 100) / fs_rate * fs_spk, T/fs_rate*fs_spk);


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

  if i == 8
      break
  end

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

  % get input freq
    split1 = strsplit(curr_fname, '_2025');
    split2 = strsplit(split1{1}, '_');
    input_freq = str2num(split2{end});

    disp([num2str(input_freq), 'Hz model, max stable performance ', num2str(max(all_perfs))])

  if max(all_perfs) < 0.95
      disp('performance to low, moving to next model...')
      continue
  end


  % XOR task
  if strcmpi(task_name, 'xor')
    down_sample = 1;

    scaling_factor = opt_scaling_factor;
      outs = zeros(n_trials, T/fs_rate*fs_spk);
      trials = zeros(n_trials, 1);
      perfs = zeros(n_trials, 1);

      
      disp(scaling_factor)
      parfor j = 1:n_trials
        u = zeros(2, T+1);
        u_lab = zeros(1, 2);

        % Stim 1
        if rand >= 0.50
          u(1, stim_on:stim_on + stim_dur) = 1;
          u_lab(1) = 1;
        else
          u(1, stim_on:stim_on+stim_dur) = -1;
          u_lab(1) = -1;
        end

        % Stim 2
        if rand >= 0.50
          u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = 1;
          u_lab(2) = 1;
        else
          u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = -1;
          u_lab(2) = -1;
        end
        label = prod(u_lab);
        trials(j) = label;

        wave = sin(2*pi*input_freq/fs_rate*(1:T));
        wave = wave * u_lab(1); % flip based on stim1
        lfp_input = zeros(1,T+1); % python code: 2 * np.pi * f / fs * time[period[0]:period[1]]
        lfp_input(stim_on+stim_dur:stim_on+stim_dur+delay) = wave(1:delay+1);

        stims = struct();
        stims.mode = 'none';
        [W, REC, spk, rs, all_fr, out, params] = LIF_network_fnc(curr_full, scaling_factor,...
            u, stims, lfp_input, down_sample, use_initial_weights);
        outs(j, :) = out;
        if label == 1
          if (max(out(stim2_offset:response_end)) > 0.7) && (min(out(stim2_offset:response_end)) > -0.7)
            perfs(j) = 1;
          end
        elseif label == -1
          if (min(out(stim2_offset:response_end)) < -0.7) && (max(out(stim2_offset:response_end)) < 0.7)
            perfs(j) = 1;
          end
        end
      end % parfor end
      
      figure; plot(outs');title([mat_files(i).name(1:end-4), ' perf ',num2str(mean(perfs))])
      

  end
end