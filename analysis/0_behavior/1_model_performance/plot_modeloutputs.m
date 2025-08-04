%% Plot behavior performance for pretrained models on xor task

clear; clc;

% models_types = {'good_models';'bad_models'};
% lesion_connections = {0}; %{0;'ii';'ee';'ei';'ie'};
% n_lesion_types = length(lesion_connections);
% 
longer_delay = ''; % '' if standard delay (150), else a number (eg 200, 250)


% for n_type = 1:length(models_types) % good and bad models

current_path = pwd;

task_dir = '/home/nuttidalab/Documents/renee/all_DMS_models';

addpath('/home/nuttidalab/Documents/renee/spikeRNN/spiking')

% save_name = [task_dir,'/',models_type, '/', models_type,'_performance',num2str(longer_delay),'.mat'];

% Load previously saved models in the group of interest
% stable_mods = load([task_dir,'/',models_type, '/', models_type,'_list']).stable_mods;

% eval_perf_mean = zeros(length(stable_mods), n_lesion_types);
% eval_perf_mean_all = zeros(length(stable_mods), n_lesion_types);
% if n_type == 1
%     n_model = 2; % get model 2 for good models
% else
%     n_model = 1; % get model 1 for bad models
% end
% fprintf('\n');
% disp([models_types{n_type}, ' number ', num2str(n_model), ' of ', num2str(length(stable_mods))]);

% Get the model

% model_name = 'Task_xor_N_200_Taus_4.0_25.0_Act_sigmoid_2019_09_07_004449.mat' % bad model, but improving
% model_name = 'Task_xor_N_200_Taus_4.0_25.0_Act_sigmoid_2019_09_07_005938.mat' % bad model, but improving
% model_name = 'Task_xor_N_200_Taus_4.0_25.0_Act_sigmoid_2019_09_06_204500.mat' % bad model, but improving
% model_name = 'Task_xor_N_200_Taus_4.0_25.0_Act_sigmoid_2019_09_06_191340.mat' % bad model, at chance
% model_name = 'Task_xor_N_200_Taus_4.0_25.0_Act_sigmoid_2019_11_06_075450.mat' % bad model, at chance
model_name = 'Task_xor_N_200_Taus_4.0_25.0_Act_sigmoid_2019_09_07_012954.mat' % bad model, below chance
% model_name = 'Task_xor_N_200_Taus_4.0_25.0_Act_sigmoid_2019_09_07_005938.mat'
n_type = 2; % 1 for good, 2 for bad (just for plotting colors)

% model_name = 'Task_xor_N_200_Taus_4.0_25.0_Act_sigmoid_2019_09_06_152659.mat'; % good model
% model_name = 'Task_xor_N_200_Taus_4.0_25.0_Act_sigmoid_2019_09_07_070855.mat'; % good model, worse after lesion
% n_type = 1;

model_dir_path = fullfile(task_dir, model_name(1:end-4)); % Folder for model results


% colorby = "correct"; % "correct" or "stim"
colorby = "stim";

lesion = "none";
% lesion = "untuned";
% lesion = "null";


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
% stim2_offset = int64(stim2_offset);

% get untuned neuron idxs for this model (if interested in this)
model_dir_path = fullfile(task_dir, model_name(1:end-4)); % Folder for model results
load([model_dir_path,'/','tuning_delay_',num2str(delay),'.mat']);
untuned_idx = untuned_idx + 1; % change from python to matlab indexing
n_untuned = length(untuned_idx);

% Model eval on 100 random trials
eval_perf = zeros(n_trials,1); % Store models' performance
o_array = zeros(n_trials, 41100);
label_array = zeros(n_trials,1);
parfor i = 1:n_trials % For 100 trials
    [eval_u, eval_label] = generate_input_stim_xor(T, stim_on, stim_dur, delay); % Random stimulus set
    if lesion == "none"
        [~, ~, ~, ~, ~, eval_o, ~] = LIF_network_fnc(file_path, scaling_factor,...
            eval_u, stims, down_sample, use_initial_weights);
    elseif lesion == "untuned"
        [~, ~, ~, ~, ~, eval_o, ~] = LIF_network_lesion_neurons_fnc(file_path, scaling_factor,...
                        eval_u, stims, down_sample, untuned_idx, "presynaptic");
    elseif lesion == "null"
        random_idx = randsample(N, n_untuned);
        [~, ~, ~, ~, ~, eval_o, ~] = LIF_network_lesion_neurons_fnc(file_path, scaling_factor,...
                        eval_u, stims, down_sample, random_idx, "presynaptic");
    end
    
    if colorby == "stim"
        u_stims = zeros(1,2);
        for j = 1:size(eval_u,1)
            this_stims = unique(eval_u(j,:));
            u_stims(j) = this_stims(this_stims ~= 0);
        end
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
    eval_perf(i) = trial_perf;
    if colorby=="correct"
        label_array(i) = eval_label == "same";
    elseif colorby=="stim"
        if u_stims(1) == 1 && u_stims(2) == 1
            label_array(i) = 1;
        elseif u_stims(1) == 1 && u_stims(2) == -1
            label_array(i) = 2;
        elseif u_stims(1) == -1 && u_stims(2) == 1
            label_array(i) = 3;
        elseif u_stims(1) == -1 && u_stims(2) == -1
            label_array(i) = 4;
        else
            label_array(i) = nan;
        end
    end
    o_array(i,:) = eval_o;
end

plotcolor1 = '#49BEA3';
plotcolor2 = '#236975';
plotcolor3 = '#2B1644';
plotcolor4 = '#6E439A';


fs_ds = 200;
o_array = downsample_signal(fs_spk, fs_ds, o_array);

figure('Position',[10 10 900 600]); hold on;
if colorby=="correct"
    if n_type == 1
        h1=plot(o_array(label_array == 1,:)', 'Color', plotcolor1, 'DisplayName', 'Correct');
        h2=plot(o_array(label_array == 0,:)', 'Color', plotcolor2, 'DisplayName', 'Incorrect');
    elseif n_type == 2
        h1=plot(o_array(label_array == 1,:)', 'Color', plotcolor4, 'DisplayName', 'Correct');
        h2=plot(o_array(label_array == 0,:)', 'Color', plotcolor3, 'DisplayName', 'Incorrect');
    end
    legend([h1(1),h2(1)], {'Correct','Incorrect'});
    legend('AutoUpdate','off'); 
elseif colorby == "stim"
    % h1=plot(o_array(label_array == 1,:)', 'Color', plotcolor1, 'DisplayName', '+1/+1');
    % h2=plot(o_array(label_array == 2,:)', 'Color', plotcolor2, 'DisplayName', '+1/-1');
    % h3=plot(o_array(label_array == 3,:)', 'Color', plotcolor3, 'DisplayName', '-1/+1');
    % h4=plot(o_array(label_array == 4,:)', 'Color', plotcolor4, 'DisplayName', '-1/-1');
    % legend([h1(1); h2(1); h3(1); h4(1)], {'+1/+1';'+1/-1';'-1/+1';'-1/-1'})
    % legend('AutoUpdate','off'); 

    h1=plot(mean(o_array(label_array == 1,:),1)', 'Color', plotcolor1, 'DisplayName', '+1/+1','LineWidth',3);
    h2=plot(mean(o_array(label_array == 2,:),1)', 'Color', plotcolor2, 'DisplayName', '+1/-1','LineWidth',3);
    h3=plot(mean(o_array(label_array == 3,:),1)', 'Color', plotcolor3, 'DisplayName', '-1/+1','LineWidth',3);
    h4=plot(mean(o_array(label_array == 4,:),1)', 'Color', plotcolor4, 'DisplayName', '-1/-1','LineWidth',3);
    legend([h1(1); h2(1); h3(1); h4(1)], {'+1/+1';'+1/-1';'-1/+1';'-1/-1'})
    legend('AutoUpdate','off'); 


end
xline(stim1_onset/fs_spk*fs_ds, 'Color','k','LineWidth',2)
xline(stim1_offset/fs_spk*fs_ds, 'Color','k','LineWidth',2)
xline(stim2_onset/fs_spk*fs_ds, 'Color','k','LineWidth',2)
xline(stim2_offset/fs_spk*fs_ds, 'Color','k','LineWidth',2)
title(['Model ', model_name(end-9:end-4),', mean performance: ', num2str(mean(eval_perf))])
xlabel('Time (ms)')
x = 0:fs_ds/2:length(o_array);
xticks(x)
xticklabels(x/fs_ds*1000)
ylabel('Output')
fontsize(16,"points")
ylim([-1 1])

% saveas(gcf, [models_type,'_output.svg'],'svg')


% eval_perf_mean = eval_perf_mean_all;
% save(save_name, 'eval_perf_mean')
% end


% figure; hold on;
% plot(u(1,:), 'Color','k', 'LineWidth',2)
% plot(u(2,:), 'Color','k', 'LineWidth',2);
% plot(o, 'color','g', 'LineWidth',2);
% ylim([-1.1 1.1]);
% xticklabels(0:250:500/.2)
