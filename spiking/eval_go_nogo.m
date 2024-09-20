% Name: Robert Kim
% Date: October 11, 2019
% Email: rkim@salk.edu
% eval_go_nogo.m
% Description: Script to evaluate a trained LIF RNN model constructed
% to perform the Go-NoGo task

clear; clc;

% First, load one trained rate RNN
% Make sure lambda_grid_search.m was performed on the model.
% Update model_path to point where the trained model is
% model_path = '/Users/Renee/Downloads/spikeRNN/models/go-nogo/P_rec_0.2_Taus_4.0_20.0'; 
model_path = '/home/nuttidalab/Documents/spikeRNN/models/DMS_OSF';
% model_path = '/home/nuttidalab/Documents/spikeRNN/models/xor';
mat_file = dir(fullfile(model_path, '*.mat'));
model_name = mat_file(1).name;
model_path = fullfile(model_path, model_name);
load(model_path);

use_initial_weights = false;
scaling_factor = opt_scaling_factor;
down_sample = 1;

% --------------------------------------------------------------
% NoGo trial example
% --------------------------------------------------------------
u = zeros (1, 201); % input stim

% Run the LIF simulation 
stims = struct();
stims.mode = 'none';
[W, REC, spk, rs, all_fr, out, params] = LIF_network_fnc(model_path, scaling_factor,...
u, stims, down_sample, use_initial_weights);
dt = params.dt;
T = params.T;
t = dt:dt:T;

nogo_out = out;   % LIF network output
nogo_rs = rs;     % firing rates
nogo_spk = spk;   % spikes
nogo_IPSCs = params.IPSCs;  % IPSCs


% --------------------------------------------------------------
% Go trial example
% --------------------------------------------------------------
u = zeros (1, 201); % input stim
u(31:50) = 1;

% Run the LIF simulation 
stims = struct();
stims.mode = 'none';
[W, REC, spk, rs, all_fr, out, params] = LIF_network_fnc(model_path, scaling_factor,...
u, stims, down_sample, use_initial_weights);
dt = params.dt;
T = params.T;
t = dt:dt:T;

go_out = out;   % LIF network output
go_rs = rs;     % firing rates
go_spk = spk;   % spikes
go_IPSCs = params.IPSCs;  % IPSCs

% --------------------------------------------------------------
% Plot the network output
% --------------------------------------------------------------
figure; axis tight; hold on;
plot(t, nogo_out, 'm', 'linewidth', 2);
plot(t, go_out, 'g', 'linewidth', 2);


% --------------------------------------------------------------
% Plot the spike raster
% --------------------------------------------------------------
% NoGo spike raster
figure('Units', 'Normalized', 'Outerposition', [0 0 0.22 0.20]);
hold on; axis tight;
inh_ind = find(inh);
exc_ind = find(exc);
all_ind = [exc_ind; inh_ind];
all_ind = 1:N;
for i = 1:length(all_ind)
  curr_spk = nogo_spk(all_ind(i), 10:end);
  if exc(all_ind(i)) == 1
    plot(t(find(curr_spk)), ones(1, length(find(curr_spk)))*i, 'r.', 'markers', 8);
  else
    plot(t(find(curr_spk)), ones(1, length(find(curr_spk)))*i, 'b.', 'markers', 8);
  end
end
xlim([0, 1]);
ylim([-5, 205]);

% Go spike raster
figure('Units', 'Normalized', 'Outerposition', [0 0 0.22 0.20]);
hold on; axis tight;
inh_ind = find(inh);
exc_ind = find(exc);
all_ind = [exc_ind; inh_ind];
all_ind = 1:N;
for i = 1:length(all_ind)
  curr_spk = go_spk(all_ind(i), 10:end);
  if exc(all_ind(i)) == 1
    plot(t(find(curr_spk)), ones(1, length(find(curr_spk)))*i, 'r.', 'markers', 8);
  else
    plot(t(find(curr_spk)), ones(1, length(find(curr_spk)))*i, 'b.', 'markers', 8);
  end
end
xlim([0, 1]);
ylim([-5, 205]);


% --------------------------------------------------------------
% Plot the IPSCs
% --------------------------------------------------------------

figure; hold on; 
plot(transpose(go_IPSCs))
title('Go trial IPSCs')

figure; hold on; 
plot(transpose(nogo_IPSCs))
title('No-Go trial IPSCs')


% --------------------------------------------------------------
% Plot spectrograms
% --------------------------------------------------------------

window = 100;
noverlap = round(window/1.5);
nfft = window * 2;
fs = size(go_spk,2);
[s, f, t] = spectrogram(go_IPSCs(1,:), window, noverlap, nfft, fs);

figure; hold on;
imagesc(10*log(abs(real(s))))
% imagesc(real(s))
colorbar
set(gca, 'YDir','normal')


% --------------------------------------------------------------
% Save IPSCs
% --------------------------------------------------------------
save('IPSCs.mat', 'go_IPSCs','nogo_IPSCs')