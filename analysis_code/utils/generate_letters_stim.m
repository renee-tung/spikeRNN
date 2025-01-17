function [u,label, stim_letters, probe_letter] = generate_letters_stim(T, stim_on, stim_dur, delay, task_load, ...
    stim_letters, probe_letter, match)
% generates letters task stimulus with specified probe letter
% for random stim, put probe_letter = NaN
% for random match or not, put match=NaN. Otherwise put true or false

n_input_chans = task_load*2;

% letters task
u = zeros(n_input_chans, T+1);

if sum(isnan(stim_letters)) > 0
    stim_letters = randperm(n_input_chans, task_load); % load letter choices
end


if isnan(probe_letter)
    probe_letter = randperm(n_input_chans, 1); % 1 letter choice
end

label = 2*(ismember(probe_letter, stim_letters)) - 1;

if ~isnan(match) % if we want to choose match or not

    if (match == true) && (label == -1) % if we want match and it's not
        stim_letters(1) = probe_letter;
        label = 2*(ismember(probe_letter, stim_letters)) - 1;
        assert(label == 1, 'Wanted match, but trial does not match')
    elseif (match == false) && (label == 1) % if we want mismatch and it's match
        new_stim = setdiff(1:n_input_chans, stim_letters);
        stim_letters(stim_letters == probe_letter) = new_stim(randi(length(new_stim)));
        label = 2*(ismember(probe_letter, stim_letters)) - 1;
        assert(label == -1, 'Wanted mismatch, but trial matches')
    end
end

u(stim_letters, stim_on:stim_on+stim_dur) = 1; % stimulus presentation
u(probe_letter, stim_on+stim_dur+delay:end) = 1; % probe presentation



end