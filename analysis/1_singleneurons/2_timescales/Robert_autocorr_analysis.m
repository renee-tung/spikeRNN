% Name: Robert Kim
% Date: 07-13-2019
% Email: rkim@salk.edu
% autocorr_analysis.m
% Description: Script to compute autocorrelation values for the trained models

clear; clc;
addpath('/cnl/chaos/ROBERT/wm_intrinsic_timescales/code/matlab');
addpath('/cnl/chaos/ROBERT/wm_intrinsic_timescales/code/matlab/sim');

% Get all the trained models
task_path = '/nadata/cnl/chaos/ROBERT/wm_intrinsic_timescales/models/xor/FIX_P_rec_0.2_Taus_4.0_25.0/from_inh_to_exc_0.60_0.80_1.3';
task_type = 'xor';
wcard = '*Taus_*';
max_tr = 5999; % 7999 for multi-XOR
%perf_threshold = 0.95;
perf_threshold = [0.60 0.80];
disp(['PERFORMANCE THRESHOLD SET TO ' num2str(perf_threshold)]);
stable_mods = return_stable(task_path, wcard, perf_threshold, task_type, max_tr);

% Number of trials to use for computing autocorr
auto_num_trials = 50;

%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
shuffle_weights = false;
if shuffle_weights == true
  disp('Shuffling weights!!')
end
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

for i = 1:length(stable_mods)
  clear mean_decay;
  curr_mod = stable_mods{i};
  model_path = fullfile(task_path, curr_mod);
  model_path
  load(model_path);

  if exist('mean_decay')
    clearvars -except task_* stable_mods auto_num_trials shuffle_weights;
    continue;
  else
    mean_decay = NaN;
    save(model_path, 'mean_decay', '-append');
  end

  % only perform the analysis if the model is stable
  down_sample = 1;
  scaling_factor = opt_scaling_factor; 

  trial_spks = zeros(N*auto_num_trials, 21);
  frs = [];
  outs = zeros(auto_num_trials, 2100);

  % Run it first to get the weights
  stims = struct();
  stims.mode = 'none';
  [W, REC, spk, rs, all_fr, out, params] = LIF_network_fnc(model_path, scaling_factor,...
  u, stims, down_sample, shuffle_weights);
  dt = params.dt;
  T = params.T;
  t = dt:dt:T;

  % Freeze the weights!
  use_this_W = W*scaling_factor;

  for ii = 1:auto_num_trials
    if ~isempty(strfind(curr_mod, 'xor'))
      u = zeros(2, 211);
    elseif ~isempty(strfind(curr_mod, 'whisker'))
      u = zeros(1, 211);
    elseif ~isempty(strfind(curr_mod, '2afc'))
      u = zeros(1, 211);
    elseif ~isempty(strfind(curr_mod, 'mante'))
      u = zeros(4, 211);
    elseif ~isempty(strfind(curr_mod, 'xor_hard'))
      u = zeros(4, 211);
    end

    % Run the LIF simulation 
    stims = struct();
    stims.mode = 'none';
    % [W, REC, spk, rs, all_fr, out, params] = LIF_network_fnc(model_path, scaling_factor,...
    % u, stims, down_sample, shuffle_weights, use_this_W);
    [W, REC, spk, rs, all_fr, out, params] = LIF_network_fnc(model_path, scaling_factor,...
    u, stims, down_sample, shuffle_weights);
    dt = params.dt;
    T = params.T;
    t = dt:dt:T;
    outs(ii, :) = out(1:10:end);

    srate = 1/dt;
    bin_size = 50; % bin-size in ms
    npts = srate/1000*bin_size; % bin-size in pts
    nbins = length(t)/npts;

    sum_spks = [];
    for bn = 1:nbins
      curr_bin_st = (bn-1)*npts+1;
      curr_bin_end = (bn)*npts;
      sum_spks = [sum_spks, sum(spk(:, curr_bin_st:curr_bin_end), 2)];
    end
    trial_spks((ii-1)*N+1:ii*N, :) = sum_spks;
    frs = [frs, sum(spk(:, 100:end), 2)/T];
  end

  % Remove the first bin (optional)
  trial_spks = trial_spks(:, 2:end);

  % Synaptic decay time constants
  if length(taus) == 1
    syn_decay = ones(N, 1)*params.td*1000;
  else
    syn_decay = params.td*1000;
  end

  % Compute autocorrelation
  auto_c = zeros(N, 14); % number of lag values (14 => 700 ms)
  for k = 1:N
    ex_neu = trial_spks(k:N:end, :);
    for jjj = 1:size(auto_c, 2)
      curr_c = [];
      for iii = 1:size(trial_spks, 2)-jjj
        curr_corr = corr(ex_neu(:, iii), ex_neu(:, iii+jjj));
        curr_c = [curr_c, curr_corr];
      end
      auto_c(k, jjj) = nanmean(curr_c);
    end
  end

  % Remove any outlier units (i.e. units that don't fire)
  mfr = mean(frs, 2);
  low_fr = find(mfr < 3);

  auto_c(low_fr, :) = [];

  taus_decay = [];
  taus_amp = [];
  tds = [];
  Ns = [];

  new_auto_c = nan(size(auto_c));

  for ai = 1:size(auto_c, 1)
    xdata = 1:size(auto_c, 2); % skip the first one
    ydata = auto_c(ai, :);

    mm = 2;
    delta = 0;
    while delta >= 0 & mm <= length(ydata)
      delta = ydata(mm) - ydata(mm-1);
      mm = mm + 1;
    end
    mm = mm -2; % when to start the fitting

    if length(find(isnan(ydata))) == 0 & mm < 4
      xdata = 1:size(auto_c, 2)-(mm-1);
      ydata = auto_c(ai, mm:end);

      new_auto_c(ai, xdata) = ydata;

      fun = @(x, xdata)x(1)*(exp(x(2)*xdata)+x(3));
      x0 = [0, 0, 0];
      options = optimoptions('lsqcurvefit','Algorithm','levenberg-marquardt');
      x = lsqcurvefit(fun, x0, xdata, ydata, [], [], options);
      Ns = [Ns, ai];

      xdata2 = linspace(xdata(1), xdata(end), length(xdata));
      taus_decay = [taus_decay, 1/-x(2)];
      taus_amp= [taus_amp, x(1)];

      if length(taus) > 1
        tds = [tds, params.td(ai)*1000];
      else
        tds = [tds, params.td*1000];
      end
    else
      taus_decay = [taus_decay, NaN];
      taus_amp = [taus_amp, NaN];
      if length(taus) > 1
        tds = [tds, params.td(ai)*1000];
      else
        tds = [tds, params.td*1000];
      end
      continue;
    end
  end

  % Compute the autocorr decay constants
  taus_decay_ms = taus_decay*bin_size;

  % Remove long or negative decays
  outliers = find(taus_decay_ms > 500 | taus_decay_ms < 0 | taus_amp < 0);
  taus_decay_ms(outliers) = [];
  auto_c(outliers, :) = [];
  new_auto_c(outliers, :) = [];

  mean_decay = nanmean(taus_decay_ms);

  % Get the neuron indices
  auto_N = 1:N; auto_N(low_fr) = []; auto_N(outliers) = [];

  % Get the firing rates
  auto_N_fr = mfr(auto_N);

  mean_decay

  save(model_path, 'mean_decay', 'taus_decay_ms', 'auto_c', 'auto_N', ...
  'syn_decay', 'new_auto_c', 'auto_N_fr', 'outs', 'trial_spks', 'use_this_W', '-append');
  
  clearvars -except task_* stable_mods auto_num_trials shuffle_weights 
end


%{    
% Get all the trained models
task_path = '/nadata/cnl/chaos/ROBERT/spiking_working_memory/models/xor/P_rec_0.2';
stable_mods = return_stable(task_path);


full_auto = [];
all_syns = [];
all_taus = [];
for i = 1:length(stable_mods)
  load(fullfile(task_path, stable_mods{i}));
  if length(auto_N) ~= length(taus_decay_ms)
    continue;
  end
  full_auto = [full_auto; auto_c];
  all_syns = [all_syns, syn_decay(auto_N)'];
  all_taus = [all_taus, taus_decay_ms];
end

figure; axis tight; hold on;
plot(full_auto(:, 2:end)', 'b');
plot(nanmean(full_auto(:, 2:end)), 'ro-', 'linewidth', 2);

%}



