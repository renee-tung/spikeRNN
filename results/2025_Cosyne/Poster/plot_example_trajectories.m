%%


% clear; clc

load('/home/nuttidalab/Documents/spikeRNN/results/2025_Cosyne/Poster/pcs_good.mat');
% load('/home/nuttidalab/Documents/spikeRNN/results/2025_Cosyne/Poster/pcs_bad.mat');


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

time_cutoff = 51;


% Color hex codes and labels
colors = ["#9C9C9C","#4C4C4C","#236975", "#49BEA3"];
labels = ["-1/-1","-1/+1","+1/-1","+1/+1"];

% % Function to convert hex color to RGB
function rgb = hex2rgb(hex)
    hex = char(hex); % Convert to char in case it’s a string
    hex = hex(2:end); % Remove the '#' symbol
    rgb = reshape(sscanf(hex, '%2x') / 255, 1, 3); % Convert to RGB
end

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

figure('Position', [100, 100, 1000, 800]);
hold on; view(3);

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

xlabel('PC1')
ylabel('PC2')
zlabel('PC3')


% Show the legend
legend show;


% good model
view(175,-17)
saveas(gcf, 'goodmodel_pca.svg')

% % % bad model
% % view(13,10)
% view(193,-13)
% saveas(gcf, 'badmodel_pca.svg')