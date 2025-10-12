
clear; clc;

% Directory containing all the trained rate RNN model .mat files
model_dir = '/home/nuttidalab/Documents/renee/lfp_input_models_wtrain/models/xor/phase/P_rec_0.2_Taus_4.0_25.0';

mat_files = dir(fullfile(model_dir, '*_4.0_*.mat'));
input_freq = 4;
disp(['input frequency: ', num2str(input_freq)]);

max_perf_before = zeros(length(mat_files),1);
max_perf_after = zeros(length(mat_files),1);
for i = 1:length(mat_files)
  curr_fname = mat_files(i).name;
  curr_full = fullfile(mat_files(i).folder, curr_fname);
  disp(['Analyzing ' curr_fname]);


  % Load the model
  load(curr_full);
  max_perf_before(i) = max(all_perfs);
  max_perf_after(i) = max(all_perfs_new);

end

figure; hold on; 
scatter(max_perf_before, max_perf_after, 'filled');
plot([0.95, 0.95], [0, 1], color='r');
plot([0, 1], [0.95, 0.95], color='r');
xlim([0.5, 1]);
ylim([0.5, 1])
xlabel('model perf, original criteria')
ylabel('model perf, new criteria')