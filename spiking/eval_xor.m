% Name: Robert Kim
% Date: October 11, 2019
% Email: rkim@salk.edu
% eval_go_nogo.m
% Description: Script to evaluate a trained LIF RNN model constructed
% to perform the Go-NoGo task

clear; clc;

current_path = pwd;

% First, load one trained rate RNN
% Make sure lambda_grid_search.m was performed on the model.

% Update model_path to point where the trained model is
% model_path = '/Users/Renee/Downloads/spikeRNN/models/go-nogo/P_rec_0.2_Taus_4.0_20.0'; 
model_path = '/home/nuttidalab/Documents/spikeRNN/models/DMS_OSF';
% model_path = '/home/nuttidalab/Documents/spikeRNN/models/xor';
mat_file = dir(fullfile(model_path, '*.mat'));
model_name = mat_file(1).name;

% make a folder for the model
cd(model_path)
if ~exist(model_name(1:(end-4)), 'dir')
    mkdir(model_name)
end

cd(current_path)

model_path = fullfile(model_path, model_name);
load(model_path);

% get the model path again bc was written over
model_path = '/home/nuttidalab/Documents/spikeRNN/models/DMS_OSF';
% model_path = '/home/nuttidalab/Documents/spikeRNN/models/xor';
mat_file = dir(fullfile(model_path, '*.mat'));
model_name = mat_file(1).name;
model_path = fullfile(model_path, model_name);

use_initial_weights = false;
scaling_factor = opt_scaling_factor;
down_sample = 1;

% --------------------------------------------------------------
% Same trial example
% --------------------------------------------------------------

T = 300;
stim_on = 50;
stim_dur = 50;
delay = 10;

u = zeros(2, T+1); % input stim
u(1, stim_on:stim_on+stim_dur) = 1; % first stim is +1
u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = 1; % second stim is +1


% Run the LIF simulation 
stims = struct();
stims.mode = 'none';
[W, REC, spk, rs, all_fr, out, params] = LIF_network_fnc(model_path, scaling_factor,...
u, stims, down_sample, use_initial_weights);
dt = params.dt;
T = params.T;
t = dt:dt:T;

same_out = out;   % LIF network output
same_rs = rs;     % firing rates
same_spk = spk;   % spikes
same_IPSCs = params.IPSCs;  % IPSCs



% --------------------------------------------------------------
% Diff trial example
% --------------------------------------------------------------

T = 300;
stim_on = 50;
stim_dur = 50;
delay = 10;

u = zeros(2, T+1); % input stim
u(1, stim_on:stim_on+stim_dur) = 1; % first stim is +1
u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = -1; % second stim is -1

n_trials = 15;

% Run the LIF simulation on n_trials trials
diff_IPSCs = zeros([200,30000, n_trials]); % 200 neurons, 30k samples, n_trials
for i=1:n_trials

    stims = struct();
    stims.mode = 'none';
    [W, REC, spk, rs, all_fr, out, params] = LIF_network_fnc(model_path, scaling_factor,...
        u, stims, down_sample, use_initial_weights);
    dt = params.dt;
    T = params.T;
    t = dt:dt:T;

    % diff_out = out;   % LIF network output
    % diff_rs = rs;     % firing rates
    % diff_spk = spk;   % spikes
    diff_IPSCs(:,:,i) = params.IPSCs;  % IPSCs

end

% --------------------------------------------------------------
% Plot the network output
% --------------------------------------------------------------
figure; axis tight; hold on;
plot(t, same_out, 'm', 'linewidth', 2);
plot(t, diff_out, 'g', 'linewidth', 2);


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
  curr_spk = same_spk(all_ind(i), 10:end);
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
  curr_spk = diff_spk(all_ind(i), 10:end);
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
plot(transpose(diff_IPSCs))
title('Diff trial IPSCs')

figure; hold on; 
plot(transpose(same_IPSCs))
title('Same trial IPSCs')


% --------------------------------------------------------------
% Plot spectrograms
% --------------------------------------------------------------

window = 100;
noverlap = round(window/1.5);
nfft = window * 2;
fs = size(diff_spk,2);
[s, f, t] = spectrogram(diff_IPSCs(1,:), window, noverlap, nfft, fs);

figure; hold on;
imagesc(10*log(abs(real(s))))
% imagesc(real(s))
colorbar
set(gca, 'YDir','normal')


% --------------------------------------------------------------
% Save IPSCs
% --------------------------------------------------------------

save('Task_xor_N_200_Taus_4.0_25.0_Act_sigmoid_2019_09_06_152659_IPSCs.mat', 'diff_IPSCs','same_IPSCs')

%% PACs calculation

% starting parameters, inspired by Daume et al 2024
% here: https://github.com/rutishauserlab/SBCAT-release-NWB/blob/main/NWB_SBCAT_analysis/helpers/internal/PAC/cfc_tort_comodulogram.m

% Input
% datMat: 2d matrix containing time samples of interest per trial (samples x trials)
% srate: sampling rate
% n_surrogates: number of surrogates for obtaining z-scored comodulogram (0 = no surrogates computed; default: 0)
% n_bins: number of bins to compute modulation index (default: 18 bins)
% LF_steps: center frequencies for phase signal in Hz (default: 2:2:14)
% LF_bw: Bandwidth for phase signal in Hz (default: 2)
% HF_steps: center frequencies for amplitude signal in Hz (default: 30:5:150; bandwidth is determined by phase signal frequency)
% tcutEdge: time to cut off at the edges of each trial to prevent filter artifacts in s (full time will be cutoff at the beginning and end of trial; default: 0 (no cutoff)) 
%
% Output
% comdlgrm: raw MI comodulugram; LF_steps x HF_steps
% comdlgrm_z: z-scored MI comodulugram; if n_surrogates > 0; LF_steps x HF_steps

% clear; clc
% load('Task_xor_N_200_Taus_4.0_25.0_Act_sigmoid_2019_09_06_152659_IPSCs.mat', 'diff_IPSCs','same_IPSCs');

%% get 15 trials

T = 300;
stim_on = 50;
stim_dur = 50;
delay = 10;

u = zeros(2, T+1); % input stim
u(1, stim_on:stim_on+stim_dur) = 1; % first stim is +1
u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = -1; % second stim is -1

n_trials = 15;

% Run the LIF simulation on n_trials trials
diff_IPSCs = zeros([200,30000, n_trials]); % 200 neurons, 30k samples, n_trials
for i=1:n_trials

    stims = struct();
    stims.mode = 'none';
    [W, REC, spk, rs, all_fr, out, params] = LIF_network_fnc(model_path, scaling_factor,...
        u, stims, down_sample, use_initial_weights);
    dt = params.dt;
    T = params.T;
    t = dt:dt:T;

    diff_IPSCs(:,:,i) = params.IPSCs;  % IPSCs

end

clearvars -except diff_IPSCs

%% organize data for function

datMat = squeeze(diff_IPSCs(1,:,:)); % just take the first neuron; samples x trials 
srate = 20000; % sampling rate
n_surrogates = 200; % 200 were used for z-scored MI in Daume et al
n_bins = 18; % go by default
LF_steps = 2:2:14;
LF_bw = 2;
HF_steps = 30:5:150;
tcutEdge = 0;

[comdlgrm, comdlgrm_z, phase2power] = cfc_tort_comodulogram(datMat,srate,n_surrogates,n_bins,LF_steps,LF_bw,HF_steps,tcutEdge);

save('comodulogram_diffIPSCs15.mat','comdlgrm','comdlgrm_z','phase2power')

figure; hold on; axis image
x0=10; y0=10; width=1000; height=800;
set(gcf,'position',[x0,y0,width,height])
imagesc(comdlgrm_z);
xticks(1:length(HF_steps)); xticklabels(HF_steps); xlabel('frequency for amplitude (Hz)', 'FontSize',16);
yticks(1:length(LF_steps)); yticklabels(LF_steps); ylabel('frequency for phase (Hz)','FontSize',16)
cb = colorbar(); ylabel(cb, 'Modulation Index, z-scored', 'FontSize',16,'Rotation',270)


figure; hold on; axis image
x0=10; y0=10; width=800; height=1500;
set(gcf,'position',[x0,y0,width,height])
imagesc(transpose(comdlgrm_z));
yticks(1:length(HF_steps)); yticklabels(HF_steps); ylabel('frequency for amplitude (Hz)', 'FontSize',16);
xticks(1:length(LF_steps)); xticklabels(LF_steps); xlabel('frequency for phase (Hz)','FontSize',16)
cb = colorbar(); ylabel(cb, 'Modulation Index, z-scored', 'FontSize',16,'Rotation',270)
saveas(gcf, 'comdlgrmz_diffIPSCs15.png')


figure; hold on; axis image
x0=10; y0=10; width=800; height=1500;
set(gcf,'position',[x0,y0,width,height])
imagesc(transpose(comdlgrm));
yticks(1:length(HF_steps)); yticklabels(HF_steps); ylabel('frequency for amplitude (Hz)', 'FontSize',16);
xticks(1:length(LF_steps)); xticklabels(LF_steps); xlabel('frequency for phase (Hz)','FontSize',16)
cb = colorbar(); ylabel(cb, 'Modulation Index', 'FontSize',16,'Rotation',270)
saveas(gcf, 'comdlgrm_diffIPSCs15.png')


% cutTrial = logical(tcutEdge);
% 
% 
% n_trials       = size(datMat,2);
% n_samples_long = size(datMat,1);
% 
% %% initalize output matrix
% clear comdlgrm
% n_HF = length(HF_steps);
% n_LF = length(LF_steps);
% comdlgrm = nan(n_LF,n_HF);
% comdlgrm_surr_mean = nan(n_LF,n_HF);
% comdlgrm_surr_std = nan(n_LF,n_HF);
% phase2power = nan(n_LF,n_HF,n_bins);
% 
% %% Prepare filter input
% 
% EEG = [];
% EEG.srate  = srate;
% EEG.pnts   = n_samples_long;
% EEG.trials = n_trials;
% EEG.nbchan = 1;
% EEG.data   = datMat;
% 
% %% Loop through all frequency pairs
% for i_phase = 1:n_LF
%     disp('Filtering and computing hilbert transform for phase signal...');
% 
%     % filter phase signal
%     LF_bp = [LF_steps(i_phase)-LF_bw/2 LF_steps(i_phase)+LF_bw/2];
%     EEG_p = pop_eegfiltnew(EEG,LF_bp(1),LF_bp(2));
% 
% 
%     %% Hilbert
%     phase_long = angle(hilbert(squeeze(EEG_p.data)));
%     clear EEG_p
% 
%     %% Cutting out time window of interest
%     if cutTrial
%         disp('Cutting out time window of interest in each trial for further analysis...')
%         t2cut_samples = tcutEdge*srate;
%         phase_toi     = phase_long(t2cut_samples+1:end-t2cut_samples,:);
%     else
%         phase_toi = phase_long;
%     end
% 
%     % Transfer data to one long vector
%     n_samples = size(phase_toi,1);
%     numpoints = n_samples * n_trials;
%     phase     = reshape(phase_toi,numpoints,1);
% 
%     clear phase_toi phase_long
% 
%     %%
%     for i_amplitude = 1:n_HF
%         fprintf('Computing PAC between %dHz amplitude and %dHz phase frequency...\n',HF_steps(i_amplitude),LF_steps(i_phase));
% 
%         HF_bp = [HF_steps(i_amplitude)-LF_steps(i_phase) HF_steps(i_amplitude)+LF_steps(i_phase)];
%         EEG_A = pop_eegfiltnew(EEG,HF_bp(1),HF_bp(2));
% 
%         %% Hilbert
%         amplitude_long = abs(hilbert(squeeze(EEG_A.data)));
%         clear EEG_A
% 
%         %% Cutting out time window of interest
%         if cutTrial
%             t2cut_samples = tcutEdge*srate;
%             amplitude_toi = amplitude_long(t2cut_samples+1:end-t2cut_samples,:);
%         else
%             amplitude_toi = amplitude_long;
%         end
%         % Transfer data to one long vector
%         amplitude = reshape(amplitude_toi,numpoints,1);
% 
%         clear amplitude_toi amplitude_long
% 
%         %% Code for calculating MI like Tort
% 
%         phase_degrees = rad2deg(phase); % Phases in degrees
%         % Bining the phases
%         step_length = 360/n_bins;
%         phase_bins = -180:step_length:180;
%         [~,phase_bins_ind] = histc(phase_degrees,phase_bins);
%         clear phase_degrees
% 
%         % Averaging amplitude time series over phase bins
%         amplitude_bins = nan(n_bins,1);
% 
%         for bin = 1:n_bins
%             amplitude_bins(bin,1) = mean(amplitude(phase_bins_ind==bin));
%         end
% 
%         % Normalize amplitudes
%         P = amplitude_bins./repmat(sum(amplitude_bins),n_bins,1);
% 
%         % Compute modulation index and store in comodulogram
%         mi = 1+sum(P.*log(P))./log(n_bins);
%         comdlgrm(i_phase,i_amplitude) = mi;
%         phase2power(i_phase,i_amplitude,:) = P;
% 
%         clear amplitude_bins mi P
% 
%         %% Compute surrogates
%         if n_surrogates
% 
%              mi_surr = nan(n_surrogates,1);
% 
%             % reshape back to trials for shuffling
%             amplitude_trials = reshape(amplitude,n_samples,n_trials);
%             clear amplitude
% 
%             %% compute surrogate values
%             disp('Computing surrogate data...');
%             for s=1:n_surrogates
%                 randind = randperm(n_trials);
%                 surrogate_amplitude = reshape(amplitude_trials(:,randind),numpoints,1);
%                 amplitude_bins_surr = zeros(n_bins,1);
%                 for bin = 1:n_bins
%                     amplitude_bins_surr(bin,1) = mean(surrogate_amplitude(phase_bins_ind==bin));
%                 end
%                 P_surr = amplitude_bins_surr./repmat(sum(amplitude_bins_surr),n_bins,1);
%                 mi_surr(s) = 1+sum(P_surr.*log(P_surr))./log(n_bins);
%             end
% 
%             %% fit gaussian to surrogate data, uses normfit.m from MATLAB Statistics toolbox
%             [surrogate_mean,surrogate_std]=normfit(mi_surr);
% 
%             comdlgrm_surr_mean(i_phase,i_amplitude) = surrogate_mean;
%             comdlgrm_surr_std(i_phase,i_amplitude) = surrogate_std;
% 
%             clear mi_surr
%         end
% 
%     end %loop amplitude
% end %loop phase
% 
% % z-transform raw comod
% comdlgrm_z = (comdlgrm - comdlgrm_surr_mean) ./ comdlgrm_surr_std;
% disp('Done')