function [u,match] = generate_input_stim_sternberg(T, stim_on, stim_dur, delay, wm_load, match)
% generates random stimulus from sternberg
% set match to 0 if want random

n_chans = 4;

if match == 0
    if rand() < 0.50
        match = 1;
    else
        match = -1;
    end
end

% Sternberg task
u = zeros(n_chans, T+1); % input stim
stims = randperm(n_chans, wm_load);
if match == 1
    probe = stims(randi(length(stims)));
elseif match == -1
    non_stims = setdiff(1:n_chans, stims);
    probe = non_stims(randi(length(non_stims)));
end

for i = 1:length(stims)
    i_chan = stims(i);

    if i == 1
        u(i_chan, stim_on : stim_on + stim_dur - 1) = 1;
        ending_idx = stim_on + stim_dur - 1;
    else
        u(i_chan, ending_idx + 1 : ending_idx + stim_dur) = 1;
        ending_idx = ending_idx + stim_dur;
    end
end

u(probe, ending_idx + delay : ending_idx + delay + stim_dur - 1) = 1;


end