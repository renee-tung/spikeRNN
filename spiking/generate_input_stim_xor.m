function [u,label] = generate_input_stim_xor(T, stim_on, stim_dur, delay)
% generates random stimulus from xor

% XOR task
u = zeros(2, T+1); % input stim
labs = zeros(2,1);

if rand() < 0.50
    u(1, stim_on:stim_on+stim_dur) = 1;
    labs(1) = 1;
else
    u(1, stim_on:stim_on+stim_dur) = -1;
    labs(1) = -1;
end

if rand() < 0.50
    u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = 1;
    labs(2) = 1;
else
    u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = -1;
    labs(2) = -1;
end


if labs(1)*labs(2) == 1
    label = 'same';
else
    label = 'diff';
end


end