% Plot taus


clear; clc;

% Directory containing all the trained rate RNN model .mat files

model_dir = '/home/nuttidalab/Documents/renee/sternberg/interleaved_0.5/';
mat_files = dir(fullfile(model_dir, '*N_1000*_04_08_*.mat'));

model_taus_sig = zeros(1000,length(mat_files));

for i = 1:length(mat_files)
    curr_fname = mat_files(i).name;
    curr_full = fullfile(mat_files(i).folder, curr_fname);
    disp(['Analyzing ' curr_fname]);
    load(curr_full);
    clear load

    model_taus_sig(:,i) = apply_sigmoid(taus_gaus)*(taus(2)-taus(1)) + taus(1);
    % taus_sig = apply_sigmoid(taus_gaus)*(100-4) + 4;
    
end
% taus_sig = tf.sigmoid(taus_gaus)*(taus[1] - taus[0]) + taus[0]

model_names = arrayfun(@(f) f.name(end-9:end-4), mat_files, 'UniformOutput', false);;

figure; hold on;
boxplot(model_taus_sig, 'Labels', model_names)
xtickangle(45);
ylabel('Tau values');
title('Tau distributions per model')

function [sig_out] = apply_sigmoid(inputs)
    sig_out = 1 ./ (1 + exp(-inputs));
end


% taus_sig = 1 ./ (1+exp(-taus_gaus))*(100-4) + 4;
% boxplot(taus_sig)
% title([mat_files(i).name(end-9:end-4),' taus'])
