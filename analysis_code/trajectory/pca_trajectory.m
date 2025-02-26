% Name: Robert Kim
% Date: 08-10-2019
% Email: rkim@salk.edu
% pca_trajectory.m
% Description: Script to plot state trajectories in PC space

clear; clc
% addpath('/cnl/chaos/ROBERT/wm_intrinsic_timescales/code/matlab');
% addpath('/cnl/chaos/ROBERT/wm_intrinsic_timescales/code/matlab/sim');
% addpath('/home/rkim/Documents/MATLAB/matlab-cmu');

clear; clc;

current_path = pwd;

task_path = '/scratch/spikeRNN/models/DMS_OSF';
mat_files = return_stable(task_path, '*Taus*', 0.95, 'xor');
model_name = mat_files{3};
model_path = fullfile(task_path, model_name);

% make a folder for the model
cd(task_path)
if ~exist(model_name(1:(end-4)), 'dir')
    mkdir(model_name(1:(end-4)))
end
model_dir_path = strcat(task_path,'/',model_name(1:(end-4)));

cd(current_path)

% Where output images will be saved
out_dir = model_dir_path; %'/nadata/cnl/chaos/ROBERT/spiking_working_memory/code/RK_TF_RNN/xor';

load(model_path);
model_path = fullfile(task_path, model_name);

% Scaling factor and sampling rate
scaling_factor = opt_scaling_factor;
down_sample = 1;

%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
shuffle_weights = false;
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

% Evaluate performance across all possible input combinations
first_stim = [-1, 1];
second_stim = [-1, 1];
full_stims = [];
outs = zeros(4, 50, 41000);
all_rs = zeros(size(outs, 1)*size(outs, 2), N, size(outs, 3));
counter = 1;
counter2 = 1;
for ii = 1:length(first_stim)
  ii
  for jj = 1:length(second_stim)
    u = zeros(2, 411);
    u(1, 31:80) = first_stim(ii);
    u(2, 231:280) = second_stim(jj);

    for i = 1:size(outs, 2)
      stims = struct();
      stims.mode = 'none';

      [W, REC, spk, rs, all_fr, out, params] = LIF_network_fnc(model_path, scaling_factor,...
      u, stims, down_sample, shuffle_weights);
      outs(counter, i, :) = out(1, 1:end);
      all_rs(counter2, :, :) = rs;
      counter2 = counter2 + 1;
    end
    counter = counter + 1;
    full_stims = [full_stims; [first_stim(ii), second_stim(jj)]];
  end
end
mean_outs = squeeze(mean(outs, 2));

figure; axis tight; hold on;
plot(mean_outs(1, :), 'linewidth', 2, 'Color', 'r');
plot(mean_outs(2, :), 'linewidth', 2, 'Color', 'b');
plot(mean_outs(3, :), 'linewidth', 2, 'Color', 'c');
plot(mean_outs(4, :), 'linewidth', 2, 'Color', 'm');

N_tr = size(outs, 2);
ax = squeeze(mean(all_rs(1:N_tr,   :, 51:end)));   % -1 -1   
ay = squeeze(mean(all_rs(N_tr+1:N_tr*2,  :, 51:end)));  % -1 1
bx = squeeze(mean(all_rs(N_tr*2+1:N_tr*3, :, 51:end)));  % 1 -1
by = squeeze(mean(all_rs(N_tr*3+1:end, :, 51:end)));  % 1 1
% ^ originally trials x neurons x time; 51:end to cut off beginning time
% segment; take mean over trials so becomes neurons x time for each condition

trial_dur = size(ax, 2);

combined_data = [ax, ay, bx, by]; % neurons x time (concatenated)

W = pca(combined_data'); % pca on time x neurons; becomes 200 x 200 (neurons x PCs)

Z = combined_data'*W; % timexneurons @ neuronsxPCs -> timexPCs
comps = [1, 2, 3];

figure('Units', 'Normalized', 'Outerposition', [0 0 0.20 0.40]);
axis tight; hold all;
fix_dur = 300*10;
stim_dur = 500*10;
delay_dur = 1500*10;
resp_dur = 1300*10;
step_size = 20;
line_style = '-';
for i = 1:4
  cue_st = (i-1)*trial_dur+1+fix_dur;
  % Cue onset
  %plot3(Z(cue_st-10, comps(1)), Z(cue_st-10, comps(2)), Z(cue_st-10, comps(3)),...
  %'o', 'markers', 12, 'MarkerFace', 'g');

  % Cue period
  if full_stims(i, 1) == 1
    marker_col = 'g';
  else
    marker_col = 'w';
  end
  plot3(Z(cue_st:step_size:cue_st+stim_dur-1, comps(1)),...
  Z(cue_st:step_size:cue_st+stim_dur-1, comps(2)),...
  Z(cue_st:step_size:cue_st+stim_dur-1, comps(3)), 'go', 'linewidth', 1,...
  'markers', 10, 'MarkerFace', marker_col);

  % Delay period
  delay_st = cue_st + stim_dur;
  plot3(Z(delay_st:step_size:delay_st+delay_dur-1, comps(1)),...
  Z(delay_st:step_size:delay_st+delay_dur-1, comps(2)),...
  Z(delay_st:step_size:delay_st+delay_dur-1, comps(3)), 'linewidth', 1,...
  'color', 'c');

  probe_st = delay_st + delay_dur;

  % Probe onset
  %plot3(Z(probe_st-10, comps(1)), Z(probe_st-10, comps(2)), Z(probe_st-10, comps(3)),...
  %'o', 'markers', 12, 'MarkerFace', 'm');

  % Probe period
  if full_stims(i, 2) == 1
    probe_marker_col = 'm';
  else
    probe_marker_col = 'w';
  end

  plot3(Z(probe_st:step_size:probe_st+stim_dur-1, comps(1)),...
  Z(probe_st:step_size:probe_st+stim_dur-1, comps(2)),...
  Z(probe_st:step_size:probe_st+stim_dur-1, comps(3)), 'mo', 'linewidth', 1,...
  'markers', 10, 'MarkerFace', probe_marker_col);
end
xlabel('PC1');
ylabel('PC2');
zlabel('PC3');
view([-85 20]);
%print(fullfile(out_dir, 'pca_trajectory_v2.eps'), '-painters', '-depsc');
saveas(gcf,[out_dir '/trajectories_robert_delay150.png'])

% 2D PCA plots
figure('Units', 'Normalized', 'Outerposition', [0 0 0.20 0.40]);
axis tight; hold all;
fix_dur = (300-90)*1;
stim_dur = 500*1;
delay_dur = 500*1;
resp_dur = 1000*1;
step_size = 20;
line_style = '-';
for i = 1:4
  cue_st = (i-1)*trial_dur+1+fix_dur;

  % Cue period
  if full_stims(i, 1) == 1
    marker_col = 'g';
  else
    marker_col = 'w';
  end
  plot(Z(cue_st:step_size:cue_st+stim_dur-1, comps(3)),...
  Z(cue_st:step_size:cue_st+stim_dur-1, comps(2)), 'go', 'linewidth', 1,...
  'markers', 10, 'MarkerFace', marker_col);

  % Delay period  

  delay_st = cue_st + stim_dur;
  plot(Z(delay_st:step_size:delay_st+delay_dur-1, comps(3)),...
  Z(delay_st:step_size:delay_st+delay_dur-1, comps(2)), 'linewidth', 1,...
  'color', 'c');

  probe_st = delay_st + delay_dur;

  % Probe period
  if full_stims(i, 2) == 1
    probe_marker_col = 'm';
  else
    probe_marker_col = 'w';
  end

  plot(Z(probe_st:step_size:probe_st+stim_dur-1, comps(3)),...
  Z(probe_st:step_size:probe_st+stim_dur-1, comps(2)), 'mo', 'linewidth', 1,...
  'markers', 10, 'MarkerFace', probe_marker_col);
end 
xlabel('PC3');
ylabel('PC2');





%print(fullfile(out_dir, 'severe_pca_trajectory_pc1_pc2.eps'), '-painters', '-depsc');
%print(fullfile(out_dir, 'moderate_pca_trajectory_pc2_pc3.eps'), '-painters', '-depsc');


