function [u,label] = generate_letters_stim(T, stim_on, stim_dur, delay, task_load, probe_letter)
% generates letters task stimulus with specified probe letter
% for random stim, put probe_letter = NaN

n_input_chans = task_load*2;

% letters task
u = zeros(n_input_chans, T);

stim_letters = randperm(n_input_chans, task_load); % load letter choices
if isnan(probe_letter)
    probe_letter = randperm(n_input_chans, 1); % 1 letter choice
end

u(stim_letters, stim_on:stim_on+stim_dur) = 1; % stimulus presentation
u(probe_letter, stim_on+stim_dur+delay:end) = 1; % probe presentation

label = 2*(ismember(probe_letter, stim_letters)) - 1;

end