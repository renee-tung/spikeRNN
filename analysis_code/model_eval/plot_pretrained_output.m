
%% Plot model outputs


clear; clc;
% addpath('/cnl/chaos/ROBERT/wm_intrinsic_timescales/code/matlab');
% addpath('/cnl/chaos/ROBERT/wm_intrinsic_timescales/code/matlab/sim');

% Get the trained XOR model
task_type = 'xor';
task_path = '/home/nuttidalab/Documents/spikeRNN/models/DMS_OSF';
mat_files = return_stable(task_path, '*Taus*', 0.95, task_type);

% choose one example model (1 - 41)
model_num = 5;
model_path = fullfile(task_path, mat_files{model_num});

load(model_path);
model_path = fullfile(task_path, mat_files{model_num});

%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
shuffle_weights = false;
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

scaling_factor = opt_scaling_factor;
down_sample = 1;
num_trials = 10;
outs = zeros(4*num_trials, 41000);
xor_labs = [[1, 1];, [1, -1]; [-1, 1]; [-1, -1]];

counter = 1;
% for each trial type (1/1, 1/-1, -1/1, -1/-1)
for i = 1:4
  i
  u = zeros(2, 411);
  u(1, 31:80) = xor_labs(i, 1);
  u(2, 231:280) = xor_labs(i, 2);
  for ii = 1:num_trials
    stims = struct();
    stims.mode = 'none';
    [W, REC, spk, rs, all_fr, out, params] = LIF_network_fnc(model_path, scaling_factor,...
    u, stims, down_sample, shuffle_weights);
    dt = params.dt;
    T = params.T;
    t = dt:dt:T;
    outs(counter, :) = out;
    counter = counter + 1;
  end
end

figure; hold on;
plot(t, mean(outs(1:10, :)), 'r');
plot(t, mean(outs(11:20, :)), 'b');
plot(t, mean(outs(21:30, :)), 'c');
plot(t, mean(outs(31:40, :)), 'm');
axis tight;
plot([t(3100), t(3100)], ylim, 'k--')
plot([t(8000), t(8000)], ylim, 'k--')
plot([t(23100), t(23100)], ylim, 'k--')
plot([t(28000), t(28000)], ylim, 'k--')