function [u,label] = generate_specific_input_stim_xor(T, stim1, stim2, stim_on, stim_dur, delay)
% generates random stimulus from xor

% XOR task
u = zeros(2, T+1); % input stim

u(1, stim_on:stim_on+stim_dur) = stim1;

u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = stim2;


if stim1*stim2 == 1
    label = 'same';
else
    label = 'diff';
end


end