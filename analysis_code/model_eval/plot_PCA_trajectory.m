%% Load model

clear; clc;

current_path = pwd;

% First, load one trained rate RNN
% Make sure lambda_grid_search.m was performed on the model.

% Update model_path to point where the trained model is
% model_path = '/Users/Renee/Downloads/spikeRNN/models/go-nogo/P_rec_0.2_Taus_4.0_20.0';

task_dir = '/scratch/spikeRNN/models/DMS_OSF';

% Use Robert func to get good models
task_type = 'xor';
wcard = '*Taus_*';
max_tr = 5999; % 7999 for multi-XOR
perf_threshold = 0.95;
% perf_threshold = [0.60 0.80];
disp(['PERFORMANCE THRESHOLD SET TO ' num2str(perf_threshold)]);
stable_mods = return_stable(task_dir, wcard, perf_threshold, task_type, max_tr);

% or load model lists 
models_type = 'good_models';
model_list_path = [task_dir, '/', models_type, '/', models_type, '_list.mat'];
model_names = load(model_list_path);
stable_mods = model_names.stable_mods;

models_type = 'bad_models';
model_list_path = [task_dir, '/', models_type, '/', models_type, '_list.mat'];
model_names = load(model_list_path);
unstable_mods = model_names.stable_mods;

% CHANGE WHICH MODEL HERE
% model_name = stable_mods{2}; % good model
model_name = unstable_mods{1}; % bad model

% model_path = '/home/nuttidalab/Documents/spikeRNN/models/xor/P_rec_0.2_Taus_4.0_25.0';
% mat_file = dir(fullfile(model_path, '*.mat'));
% model_name = mat_file(3).name; % to change which model

% make a folder for the model
cd(task_dir)
if ~exist(model_name(1:(end-4)), 'dir')
    mkdir(model_name(1:(end-4)))
end
model_dir_path = strcat(task_dir,'/',model_name(1:(end-4)));

cd(current_path)

file_path = fullfile(task_dir, model_name);
load(file_path);

% model_path = '/home/nuttidalab/Documents/spikeRNN/models/xor/P_rec_0.2_Taus_4.0_25.0';
% mat_file = dir(fullfile(model_path, '*.mat'));
% model_name = mat_file(1).name;
% file_path = fullfile(model_path, model_name);

use_initial_weights = false;
scaling_factor = opt_scaling_factor;
down_sample = 1;

% For LIF simulation, no stims
stims = struct();
stims.mode = 'none';


%% plot some example outputs

% some model / cycling parameters
n_trials = 100;
eval_amp_threshold = 0.7;
fs_rate = 200;
fs_spk = 20000;

T = 411;
stim_on = 31;
stim_dur = 50;
delay = 150; % longer delay

stim1_onset = (stim_on)/fs_rate*fs_spk;
stim1_offset = (stim_on + stim_dur)/fs_rate*fs_spk;
stim2_onset = (stim_on + stim_dur + delay)/fs_rate*fs_spk;
stim2_offset = (stim_on + 2*stim_dur + delay)/fs_rate*fs_spk;

% model eval on 100 random trials
eval_perf = zeros(n_trials,1);

figure(); hold on;
for i = 1:n_trials+1
    [eval_u, eval_label] = generate_input_stim_xor(T, stim_on, stim_dur, delay);
    [~, ~, ~, ~, ~, eval_o, ~] = LIF_network_fnc(file_path, scaling_factor,...
        eval_u, stims, down_sample, use_initial_weights);
    if eval_label == "same"
        if sum(unique(eval_u(1,:))) > 0
            plot(transpose(eval_o),'r') % +1/+1 red
        else
            plot(transpose(eval_o),'m') % -1/-1 pink
        end
        if max(eval_o(stim2_offset:end)) > eval_amp_threshold
            eval_perf(i) = 1;
        end
    else
        if sum(unique(eval_u(1,:))) > 0
            plot(transpose(eval_o), 'b')
        else
            plot(transpose(eval_o), 'c')
        end
        if min(eval_o(stim2_offset:end)) < -eval_amp_threshold
            eval_perf(i) = 1;
        end
    end
end
xline(stim1_onset, 'k--')
xline(stim1_offset, 'k--')
xline(stim2_onset, 'k--')
xline(stim2_offset, 'k--')

eval_perf_mean = mean(eval_perf);
title([model_name(end-20:end-4) ' Performance: ' num2str(eval_perf_mean,3)])
% saveas(gcf,[model_dir_path '/output_delay150.png'])




%% Plot dynamics with biiiiig matrix (adapted from Robert code)


T = 411;
stim_on = 31;
stim_dur = 50;
delay = 150; % longer delay

fs_rate = 200;
fs_spk = 20000;

stim1_onset = (stim_on)/fs_rate*fs_spk;
stim1_offset = (stim_on + stim_dur)/fs_rate*fs_spk;
stim2_onset = (stim_on + stim_dur + delay)/fs_rate*fs_spk;
stim2_offset = (stim_on + 2*stim_dur + delay)/fs_rate*fs_spk;

n_trials = 25; %size(outs, 2);

all_rs = zeros(n_trials*4, N, T*fs_spk/fs_rate);

stim1s = [-1, 1];
stim2s = [-1, 1];
counter = 0;
full_stims = zeros(4,2);
for ii = stim1s
    for jj = stim2s
        u = zeros(2, T+1); % input stim
        u(1, stim_on:stim_on+stim_dur) = ii; % first stim
        u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = jj; % second stim
        for i=1:n_trials
            [~, ~, ~, rs, ~, ~, ~] = LIF_network_fnc(file_path, scaling_factor,...
                u, stims, down_sample, use_initial_weights);
            all_rs(i+(counter*n_trials),:,:) = rs;
        end
        counter = counter + 1;
        full_stims(counter,:) = [ii, jj];
    end
end


% Make big matrix & PCA
time_cutoff = 51;
% time_cutoff = 1;
% time_cutoff = stim1_onset-1;

ax = squeeze(mean(all_rs(1:n_trials,   :, time_cutoff:end)));   % -1 -1 (ORANGE) NOW (MAGENTA)
ay = squeeze(mean(all_rs(n_trials+1:n_trials*2,  :, time_cutoff:end)));  % -1 1 (RED) NOW (CYAN)
bx = squeeze(mean(all_rs(n_trials*2+1:n_trials*3, :, time_cutoff:end)));  % 1 -1 (PURPLE) NOW (BLUE)
by = squeeze(mean(all_rs(n_trials*3+1:end, :, time_cutoff:end)));  % 1 1 (BLUE) NOW (RED)

%ax = ax(exc, :);
%ay = ay(exc, :);
%bx = bx(exc, :);
%by = by(exc, :);

trial_dur = size(ax, 2);

combined_data = [ax, ay, bx, by]; % neurons x time
%combined_data = [ax, ax_pv_reduced, bx, bx_pv_reduced];

W = pca(combined_data'); % neurons x 200 components

Z = combined_data'*W; % timexneurons * neuronxcomponents = time x 200 components
comps = [1, 2, 3];

pcs = Z(:,comps);
pcs = reshape(pcs, [],4,length(comps));


%% plot (ugly)

colors = ["magenta","cyan","blue","red"];
labels = ["-1/-1","-1/+1","+1/-1","+1/+1"];

% Plotting
figure; hold on; view(3);
for i = 1:4

    h = plot3(pcs(:,i,1)', pcs(:,i,2)', pcs(:,i,3)','color',colors(i), 'LineWidth',2); 
    set(h,{'DisplayName'},{labels(i)})
    h = scatter3(pcs(stim1_onset-time_cutoff,i,1), pcs(stim1_onset-time_cutoff,i,2), pcs(stim1_onset-time_cutoff,i,3), ...
        100, 'filled', colors(i),'<', 'SizeData',500);
    set(h,{'DisplayName'},{'stim 1 onset'})
    h = scatter3(pcs(stim2_onset-time_cutoff,i,1), pcs(stim2_onset-time_cutoff,i,2), pcs(stim2_onset-time_cutoff,i,3), ...
        100, 'filled', colors(i), 'SizeData',300);
    set(h,{'DisplayName'},{'stim 2 onset'})
    h = scatter3(pcs(end,i,1), pcs(end,i,2), pcs(end,i,3), ...
        100, 'filled', colors(i),'pentagram', 'SizeData',500);
    set(h,{'DisplayName'},{'end'})

end

legend show
% saveas(gcf,[model_dir_path '/trajectories_delay150_combinedpca.png'])




%% PCA plot -- with proper colors

% Color hex codes and labels
colors = ["#9C9C9C","#4C4C4C","#236975", "#49BEA3"];
labels = ["-1/-1","-1/+1","+1/-1","+1/+1"];

% Function to convert hex color to RGB
function rgb = hex2rgb(hex)
    hex = char(hex); % Convert to char in case it’s a string
    hex = hex(2:end); % Remove the '#' symbol
    rgb = reshape(sscanf(hex, '%2x') / 255, 1, 3); % Convert to RGB
end

% Convert hex to RGB
colors_rgb = arrayfun(@(x) hex2rgb(colors(x)), 1:length(colors), 'UniformOutput', false);

% Plotting
figure; hold on; view(3);

for i = 1:4
    % 3D plot for each color
    h = plot3(pcs(:,i,1)', pcs(:,i,2)', pcs(:,i,3)', 'color', colors_rgb{i}, 'LineWidth', 2); 
    set(h, 'DisplayName', labels(i))
    
    % Stimulus 1 onset (scatter)
    h = scatter3(pcs(stim1_onset-time_cutoff, i, 1), pcs(stim1_onset-time_cutoff, i, 2), pcs(stim1_onset-time_cutoff, i, 3), ...
        100, colors_rgb{i}, 'filled', '<', 'SizeData', 500);
    set(h, 'DisplayName', 'stim 1 onset')
    
    % Stimulus 2 onset (scatter)
    h = scatter3(pcs(stim2_onset-time_cutoff, i, 1), pcs(stim2_onset-time_cutoff, i, 2), pcs(stim2_onset-time_cutoff, i, 3), ...
        100, colors_rgb{i}, 'filled', 'SizeData', 300);
    set(h, 'DisplayName', 'stim 2 onset')
    
    % End point (scatter)
    h = scatter3(pcs(end, i, 1), pcs(end, i, 2), pcs(end, i, 3), ...
        100, colors_rgb{i}, 'filled', 'pentagram', 'SizeData', 500);
    set(h, 'DisplayName', 'end')
end

% Show the legend
legend show;


%% PCA plot -- now adding thickness to stims

% Plotting
figure; hold on; view(3);

for i = 1:4
    % 3D plot for each color (using smoothed data)
    h = plot3(pcs(:,i,1), pcs(:,i,2), pcs(:,i,3), 'color', colors_rgb{i}, 'LineWidth', 2); 
    set(h, 'DisplayName', labels(i))
    
    % Stimulus 1 
    plot3(pcs(stim1_onset-time_cutoff:stim1_offset-time_cutoff,i,1), ...
        pcs(stim1_onset-time_cutoff:stim1_offset-time_cutoff,i,2), ...
        pcs(stim1_onset-time_cutoff:stim1_offset-time_cutoff,i,3), ...
        'color', colors_rgb{i}, 'LineWidth', 5)
    % h = scatter3(smoothed_pcs(stim1_onset-time_cutoff, i, 1), smoothed_pcs(stim1_onset-time_cutoff, i, 2), smoothed_pcs(stim1_onset-time_cutoff, i, 3), ...
    %     100, colors_rgb{i}, 'filled', '<', 'SizeData', 500);
    % set(h, 'DisplayName', 'stim 1')
    
    % Stimulus 2 
    plot3(pcs(stim2_onset-time_cutoff:stim2_offset-time_cutoff,i,1), ...
        pcs(stim2_onset-time_cutoff:stim2_offset-time_cutoff,i,2), ...
        pcs(stim2_onset-time_cutoff:stim2_offset-time_cutoff,i,3), ...
        'color', colors_rgb{i}, 'LineWidth', 7)
    % h = scatter3(smoothed_pcs(stim2_onset-time_cutoff, i, 1), smoothed_pcs(stim2_onset-time_cutoff, i, 2), smoothed_pcs(stim2_onset-time_cutoff, i, 3), ...
    %     100, colors_rgb{i}, 'filled', 'SizeData', 300);
    % set(h, 'DisplayName', 'stim 2')
    
    % % End point (scatter)
    % h = scatter3(smoothed_pcs(end, i, 1), smoothed_pcs(end, i, 2), smoothed_pcs(end, i, 3), ...
    %     100, colors_rgb{i}, 'filled', 'pentagram', 'SizeData', 500);
    % set(h, 'DisplayName', 'end')
end

% Show the legend
legend show;




%% PCA plot -- now smoothing

% Apply smoothing to the data
smoothed_pcs = zeros(size(pcs));
for i = 1:4
    smoothed_pcs(:,i,1) = smoothdata(pcs(:,i,1), 'movmean', 500);  % Apply smoothing along x-axis
    smoothed_pcs(:,i,2) = smoothdata(pcs(:,i,2), 'movmean', 500);  % Apply smoothing along y-axis
    smoothed_pcs(:,i,3) = smoothdata(pcs(:,i,3), 'movmean', 500);  % Apply smoothing along z-axis
end

% Plotting
figure; hold on; view(3);

for i = 1:4
    % 3D plot for each color (using smoothed data)
    h = plot3(smoothed_pcs(:,i,1), smoothed_pcs(:,i,2), smoothed_pcs(:,i,3), 'color', colors_rgb{i}, 'LineWidth', 2); 
    set(h, 'DisplayName', labels(i))
    
    % Stimulus 1 onset (scatter)
    h = scatter3(smoothed_pcs(stim1_onset-time_cutoff, i, 1), smoothed_pcs(stim1_onset-time_cutoff, i, 2), smoothed_pcs(stim1_onset-time_cutoff, i, 3), ...
        100, colors_rgb{i}, 'filled', '<', 'SizeData', 500);
    set(h, 'DisplayName', 'stim 1 onset')
    
    % Stimulus 2 onset (scatter)
    h = scatter3(smoothed_pcs(stim2_onset-time_cutoff, i, 1), smoothed_pcs(stim2_onset-time_cutoff, i, 2), smoothed_pcs(stim2_onset-time_cutoff, i, 3), ...
        100, colors_rgb{i}, 'filled', 'SizeData', 300);
    set(h, 'DisplayName', 'stim 2 onset')
    
    % End point (scatter)
    h = scatter3(smoothed_pcs(end, i, 1), smoothed_pcs(end, i, 2), smoothed_pcs(end, i, 3), ...
        100, colors_rgb{i}, 'filled', 'pentagram', 'SizeData', 500);
    set(h, 'DisplayName', 'end')
end

% Show the legend
legend show;


%% PCA plot -- now smoothing and adding thickness to stims

% Apply smoothing to the data
smoothed_pcs = zeros(size(pcs));
for i = 1:4
    smoothed_pcs(:,i,1) = smoothdata(pcs(:,i,1), 'movmean', 500);  % Apply smoothing along x-axis
    smoothed_pcs(:,i,2) = smoothdata(pcs(:,i,2), 'movmean', 500);  % Apply smoothing along y-axis
    smoothed_pcs(:,i,3) = smoothdata(pcs(:,i,3), 'movmean', 500);  % Apply smoothing along z-axis
end

% Plotting
figure; hold on; view(3);

for i = 1:4
    % 3D plot for each color (using smoothed data)
    h = plot3(smoothed_pcs(:,i,1), smoothed_pcs(:,i,2), smoothed_pcs(:,i,3), 'color', colors_rgb{i}, 'LineWidth', 2); 
    set(h, 'DisplayName', labels(i))
    
    % Stimulus 1 
    plot3(smoothed_pcs(stim1_onset-time_cutoff:stim1_offset-time_cutoff,i,1), ...
        smoothed_pcs(stim1_onset-time_cutoff:stim1_offset-time_cutoff,i,2), ...
        smoothed_pcs(stim1_onset-time_cutoff:stim1_offset-time_cutoff,i,3), ...
        'color', colors_rgb{i}, 'LineWidth', 5)
    % h = scatter3(smoothed_pcs(stim1_onset-time_cutoff, i, 1), smoothed_pcs(stim1_onset-time_cutoff, i, 2), smoothed_pcs(stim1_onset-time_cutoff, i, 3), ...
    %     100, colors_rgb{i}, 'filled', '<', 'SizeData', 500);
    % set(h, 'DisplayName', 'stim 1')
    
    % Stimulus 2 
    plot3(smoothed_pcs(stim2_onset-time_cutoff:stim2_offset-time_cutoff,i,1), ...
        smoothed_pcs(stim2_onset-time_cutoff:stim2_offset-time_cutoff,i,2), ...
        smoothed_pcs(stim2_onset-time_cutoff:stim2_offset-time_cutoff,i,3), ...
        'color', colors_rgb{i}, 'LineWidth', 7)
    % h = scatter3(smoothed_pcs(stim2_onset-time_cutoff, i, 1), smoothed_pcs(stim2_onset-time_cutoff, i, 2), smoothed_pcs(stim2_onset-time_cutoff, i, 3), ...
    %     100, colors_rgb{i}, 'filled', 'SizeData', 300);
    % set(h, 'DisplayName', 'stim 2')
    
    % % End point (scatter)
    % h = scatter3(smoothed_pcs(end, i, 1), smoothed_pcs(end, i, 2), smoothed_pcs(end, i, 3), ...
    %     100, colors_rgb{i}, 'filled', 'pentagram', 'SizeData', 500);
    % set(h, 'DisplayName', 'end')
end

% Show the legend
legend show;





%%


clear; clc


T = 411;
stim_on = 31;
stim_dur = 50;
delay = 150; % longer delay

fs_rate = 200;
fs_spk = 20000;

stim1_onset = (stim_on)/fs_rate*fs_spk;
stim1_offset = (stim_on + stim_dur)/fs_rate*fs_spk;
stim2_onset = (stim_on + stim_dur + delay)/fs_rate*fs_spk;
stim2_offset = (stim_on + 2*stim_dur + delay)/fs_rate*fs_spk;

time_cutoff = stim1_onset -1;


% Color hex codes and labels
colors = ["#9C9C9C","#4C4C4C","#236975", "#49BEA3"];
labels = ["-1/-1","-1/+1","+1/-1","+1/+1"];

% % Function to convert hex color to RGB
% function rgb = hex2rgb(hex)
%     hex = char(hex); % Convert to char in case it’s a string
%     hex = hex(2:end); % Remove the '#' symbol
%     rgb = reshape(sscanf(hex, '%2x') / 255, 1, 3); % Convert to RGB
% end

% Convert hex to RGB
colors_rgb = arrayfun(@(x) hex2rgb(colors(x)), 1:length(colors), 'UniformOutput', false);

% Apply smoothing to the data
smoothed_pcs = zeros(size(pcs));
for i = 1:4
    smoothed_pcs(:,i,1) = smoothdata(pcs(:,i,1), 'movmean', 500);  % Apply smoothing along x-axis
    smoothed_pcs(:,i,2) = smoothdata(pcs(:,i,2), 'movmean', 500);  % Apply smoothing along y-axis
    smoothed_pcs(:,i,3) = smoothdata(pcs(:,i,3), 'movmean', 500);  % Apply smoothing along z-axis
end

% Plotting
figure; hold on; view(3);

for i = 1:4
    % 3D plot for each color (using smoothed data)
    h = plot3(smoothed_pcs(:,i,1), smoothed_pcs(:,i,2), smoothed_pcs(:,i,3), 'color', colors_rgb{i}, 'LineWidth', 2); 
    set(h, 'DisplayName', labels(i))
    
    % Stimulus 1 
    plot3(smoothed_pcs(stim1_onset-time_cutoff:stim1_offset-time_cutoff,i,1), ...
        smoothed_pcs(stim1_onset-time_cutoff:stim1_offset-time_cutoff,i,2), ...
        smoothed_pcs(stim1_onset-time_cutoff:stim1_offset-time_cutoff,i,3), ...
        'color', colors_rgb{i}, 'LineWidth', 5)
    % h = scatter3(smoothed_pcs(stim1_onset-time_cutoff, i, 1), smoothed_pcs(stim1_onset-time_cutoff, i, 2), smoothed_pcs(stim1_onset-time_cutoff, i, 3), ...
    %     100, colors_rgb{i}, 'filled', '<', 'SizeData', 500);
    % set(h, 'DisplayName', 'stim 1')
    
    % Stimulus 2 
    plot3(smoothed_pcs(stim2_onset-time_cutoff:stim2_offset-time_cutoff,i,1), ...
        smoothed_pcs(stim2_onset-time_cutoff:stim2_offset-time_cutoff,i,2), ...
        smoothed_pcs(stim2_onset-time_cutoff:stim2_offset-time_cutoff,i,3), ...
        'color', colors_rgb{i}, 'LineWidth', 7)
    % h = scatter3(smoothed_pcs(stim2_onset-time_cutoff, i, 1), smoothed_pcs(stim2_onset-time_cutoff, i, 2), smoothed_pcs(stim2_onset-time_cutoff, i, 3), ...
    %     100, colors_rgb{i}, 'filled', 'SizeData', 300);
    % set(h, 'DisplayName', 'stim 2')
    
    % % End point (scatter)
    % h = scatter3(smoothed_pcs(end, i, 1), smoothed_pcs(end, i, 2), smoothed_pcs(end, i, 3), ...
    %     100, colors_rgb{i}, 'filled', 'pentagram', 'SizeData', 500);
    % set(h, 'DisplayName', 'end')
end

% Show the legend
legend show;

