function [signal_low] = downsample_signal(fs_high, fs_low, signal)

% Ensure that the signal has a duration of 1 second (or adjust for actual duration)
duration = length(signal) / fs_high;  % Duration in seconds

% Time vector for high sampling rate (using the actual signal duration)
t_high = linspace(0, duration, length(signal));  % Time vector for high-sampled signal

% Create a new time vector for the low sampling rate (same duration)
t_low = linspace(0, duration, round(duration * fs_low));  % Time vector for low sampling rate

% repeat for each neuron
n_neurons = size(signal, 1);


signal_low = zeros(n_neurons, length(t_low));
for i = 1:n_neurons
    signal_low(i,:) = interp1(t_high, signal(i,:), t_low, 'linear');  % Resample the signal by interpolating at the new time points
end


end