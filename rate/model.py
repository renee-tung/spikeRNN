#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Name: Robert Kim
# Date: October 11, 2019
# Email: rkim@salk.edu
# Description: Implementation of the continuous rate RNN model

import os, sys
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import scipy.io
import scipy.signal
import pywt
import pdb

'''
CONTINUOUS FIRING-RATE RNN CLASS
'''

class FR_RNN_dale:
    """
    Firing-rate RNN model for excitatory and inhibitory neurons
    Initialization of the firing-rate model with recurrent connections
    """
    def __init__(self, N, P_inh, P_rec, w_in, som_N, w_dist, gain, apply_dale, w_out):
        """
        Network initialization method
        N: number of units (neurons)
        P_inh: probability of a neuron being inhibitory
        P_rec: recurrent connection probability
        w_in: NxN weight matrix for the input stimuli
        som_N: number of SOM neurons (set to 0 for no SOM neurons)
        w_dist: recurrent weight distribution ('gaus' or 'gamma')
        apply_dale: apply Dale's principle ('True' or 'False')
        w_out: Nx1 readout weights

        Based on the probability (P_inh) provided above,
        the units in the network are classified into
        either excitatory or inhibitory. Next, the
        weight matrix is initialized based on the connectivity
        probability (P_rec) provided above.
        """
        self.N = N
        self.P_inh = P_inh
        self.P_rec = P_rec
        self.w_in = w_in
        self.som_N = som_N
        self.w_dist = w_dist
        self.gain = gain
        self.apply_dale = apply_dale
        self.w_out = w_out

        # Assign each unit as excitatory or inhibitory
        inh, exc, NI, NE, som_inh = self.assign_exc_inh()
        self.inh = inh
        self.som_inh = som_inh
        self.exc = exc
        self.NI = NI
        self.NE = NE

        # Initialize the weight matrix
        self.W, self.mask, self.som_mask = self.initialize_W()

    def assign_exc_inh(self):
        """
        Method to randomly assign units as excitatory or inhibitory (Dale's principle)

        Returns
            inh: bool array marking which units are inhibitory
            exc: bool array marking which units are excitatory
            NI: number of inhibitory units
            NE: number of excitatory units
            som_inh: indices of "inh" for SOM neurons
        """
        # Apply Dale's principle
        if self.apply_dale == True:
            inh = np.random.rand(self.N, 1) < self.P_inh
            exc = ~inh
            NI = len(np.where(inh == True)[0])
            NE = self.N - NI

        # Do NOT apply Dale's principle
        else:
            inh = np.random.rand(self.N, 1) < 0 # no separate inhibitory units
            exc = ~inh
            NI = len(np.where(inh == True)[0])
            NE = self.N - NI

        if self.som_N > 0:
            som_inh = np.where(inh==True)[0][:self.som_N]
        else:
            som_inh = 0

        return inh, exc, NI, NE, som_inh

    def initialize_W(self):
        """
        Method to generate and initialize the connectivity weight matrix, W
        The weights are drawn from either gaussian or gamma distribution.

        Returns
            w: NxN weights (all positive)
            mask: NxN matrix of 1's (excitatory units)
                  and -1's (for inhibitory units)
        NOTE: To compute the "full" weight matrix, simply
        multiply w and mask (i.e. w*mask)
        """
        # Weight matrix
        w = np.zeros((self.N, self.N), dtype = np.float32)
        idx = np.where(np.random.rand(self.N, self.N) < self.P_rec)
        if self.w_dist.lower() == 'gamma':
            w[idx[0], idx[1]] = np.random.gamma(2, 0.003, len(idx[0]))
        elif self.w_dist.lower() == 'gaus':
            w[idx[0], idx[1]] = np.random.normal(0, 1.0, len(idx[0]))
            w = w/np.sqrt(self.N*self.P_rec)*self.gain # scale by a gain to make it chaotic

        if self.apply_dale == True:
            w = np.abs(w)
        
        # Mask matrix
        mask = np.eye(self.N, dtype=np.float32)
        mask[np.where(self.inh==True)[0], np.where(self.inh==True)[0]] = -1

        # SOM mask matrix
        som_mask = np.ones((self.N, self.N), dtype=np.float32)
        if self.som_N > 0:
            for i in self.som_inh:
                som_mask[i, np.where(self.inh==True)[0]] = 0

        return w, mask, som_mask

    def load_net(self, model_dir):
        """
        Method to load pre-configured network settings
        """
        settings = scipy.io.loadmat(model_dir)
        self.N = settings['N'][0][0]
        self.som_N = settings['som_N'][0][0]
        self.inh = settings['inh']
        self.exc = settings['exc']
        self.inh = self.inh == 1
        self.exc = self.exc == 1
        self.NI = len(np.where(settings['inh'] == True)[0])
        self.NE = len(np.where(settings['exc'] == True)[0])
        self.mask = settings['m']
        self.som_mask = settings['som_m']
        self.W = settings['w']
        self.w_in = settings['w_in']
        self.b_out = settings['b_out']
        self.w_out = settings['w_out']

        return self
    
    def display(self):
        """
        Method to print the network setup
        """
        print('Network Settings')
        print('====================================')
        print('Number of Units: ', self.N)
        print('\t Number of Excitatory Units: ', self.NE)
        print('\t Number of Inhibitory Units: ', self.NI)
        print('Weight Matrix, W')
        full_w = self.W*self.mask
        zero_w = len(np.where(full_w == 0)[0])
        pos_w = len(np.where(full_w > 0)[0])
        neg_w = len(np.where(full_w < 0)[0])
        print('\t Zero Weights: %2.2f %%' % (zero_w/(self.N*self.N)*100))
        print('\t Positive Weights: %2.2f %%' % (pos_w/(self.N*self.N)*100))
        print('\t Negative Weights: %2.2f %%' % (neg_w/(self.N*self.N)*100))

'''
Task-specific input signals
'''
def generate_input_stim_go_nogo(settings):
    """
    Method to generate the input stimulus matrix for the
    Go-NoGo task

    INPUT
        settings: dict containing the following keys
            T: duration of a single trial (in steps)
            stim_on: stimulus starting time (in steps)
            stim_dur: stimulus duration (in steps)
            taus: time-constants (in steps)
            DeltaT: sampling rate
    OUTPUT
        u: 1xT stimulus matrix
        label: either +1 (Go trial) or 0 (NoGo trial) 
    """
    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']

    u = np.zeros((1, T)) #+ np.random.randn(1, T)
    u_lab = np.zeros((2, 1))
    if np.random.rand() <= 0.50:
        u[0, stim_on:stim_on+stim_dur] = 1
        label = 1
    else:
        label = 0 

    return u, label

def generate_input_stim_xor(settings):
    """
    Method to generate the input stimulus matrix (u)
    for the XOR task

    INPUT
        settings: dict containing the following keys
            T: duration of a single trial (in steps)
            stim_on: stimulus starting time (in steps)
            stim_dur: stimulus duration (in steps)
            delay: delay b/w two stimuli (in steps)
            taus: time-constants (in steps)
            DeltaT: sampling rate
    OUTPUT
        u: 2xT stimulus matrix
        label: 'same' or 'diff'
    """
    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']
    delay = settings['delay']

    # Initialize u
    u = np.zeros((2, T))

    # XOR task
    labs = []
    if np.random.rand() < 0.50:
        u[0, stim_on:stim_on+stim_dur] = 1
        labs.append(1)
    else:
        u[0, stim_on:stim_on+stim_dur] = -1
        labs.append(-1)

    if np.random.rand() < 0.50:
        u[1, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay] = 1
        labs.append(1)
    else:
        u[1, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay] = -1
        labs.append(-1)

    if np.prod(labs) == 1:
        label = 'same'
    else:
        label = 'diff'

    return u, label

def generate_input_stim_mante(settings):
    """
    Method to generate the input stimulus matrix for the
    mante task

    INPUT
        settings: dict containing the following keys
            T: duration of a single trial (in steps)
            stim_on: stimulus starting time (in steps)
            stim_dur: stimulus duration (in steps)
            taus: time-constants (in steps)
            DeltaT: sampling rate
    OUTPUT
        u: 4xT stimulus matrix (first 2 rows for motion/color and the second
        2 rows for context
        label: either +1 or -1
    """
    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']

    # Color/motion sensory inputs
    u = np.zeros((2, T))
    u_lab = np.zeros((2, 1))
    if np.random.rand() <= 0.50:
        u[0, stim_on:stim_on+stim_dur] = np.random.randn(1, stim_dur) + 0.5
        u_lab[0, 0] = 1
    else:
        u[0, stim_on:stim_on+stim_dur] = np.random.randn(1, stim_dur) - 0.5
        u_lab[0, 0] = -1

    if np.random.rand() <= 0.50:
        u[1, stim_on:stim_on+stim_dur] = np.random.randn(1, stim_dur) + 0.5
        u_lab[1, 0] = 1
    else:
        u[1, stim_on:stim_on+stim_dur] = np.random.randn(1, stim_dur) - 0.5
        u_lab[1, 0] = -1

    # Context input
    c = np.zeros((2, T))
    label = 0
    if np.random.rand() <= 0.50:
        c[0, :] = 1

        if u_lab[0, 0] == 1:
            label = 1
        elif u_lab[0, 0] == -1:
            label = -1
    else:
        c[1, :] = 1

        if u_lab[1, 0] == 1:
            label = 1
        elif u_lab[1, 0] == -1:
            label = -1

    return np.vstack((u, c)), label


'''
Task-specific target signals
'''
def generate_target_continuous_go_nogo(settings, label):
    """
    Method to generate a continuous target signal (z) 
    for the Go-NoGo task

    INPUT
        settings: dict containing the following keys
            T: duration of a single trial (in steps)
            stim_on: stimulus starting time (in steps)
            stim_dur: stimulus duration (in steps)
            taus: time-constants (in steps)
            DeltaT: sampling rate
        label: either +1 or -1
    OUTPUT
        z: 1xT target signal
    """
    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']

    z = np.zeros((1, T))
    if label == 1:
        z[0, stim_on+stim_dur:] = 1
    # elif label == 0:
        # z[0, stim_on+stim_dur:] = -1

    return np.squeeze(z)

def generate_target_continuous_xor(settings, label):
    """
    Method to generate a continuous target signal (z) 
    for the XOR task

    INPUT
        settings: dict containing the following keys
            T: duration of a single trial (in steps)
            stim_on: stimulus starting time (in steps)
            stim_dur: stimulus duration (in steps)
            delay: delay b/w two stimuli (in steps)
            taus: time-constants (in steps)
            DeltaT: sampling rate
        label: string value (either 'same' or 'diff')
    OUTPUT
        z: 1xT target signal
    """
    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']
    delay = settings['delay']
    task_end_T = stim_on+2*stim_dur + delay

    z = np.zeros((1, T))
    if label == 'same':
        z[0, 10+task_end_T:10+task_end_T+100] = 1
    elif label == 'diff':
        z[0, 10+task_end_T:10+task_end_T+100] = -1

    return np.squeeze(z)

def generate_target_continuous_mante(settings, label):
    """
    Method to generate a continuous target signal (z) 
    for the MANTE task

    INPUT
        settings: dict containing the following keys
            T: duration of a single trial (in steps)
            stim_on: stimulus starting time (in steps)
            stim_dur: stimulus duration (in steps)
            taus: time-constants (in steps)
            DeltaT: sampling rate
        label: either +1 or -1
    OUTPUT
        z: 1xT target signal
    """
    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']

    z = np.zeros((1, T))
    if label == 1:
        z[0, stim_on+stim_dur:] = 1
    else:
        z[0, stim_on+stim_dur:] = -1

    return np.squeeze(z)

'''
Band-specific LFP target signal
'''
def generate_target_LFP_bandpower(settings):
    """
    Generate a continuous target LFP bandpower signal (y) 
    for the XOR task

    INPUT
        settings: dict containing the following keys
            T: duration of a single trial (in steps)
            stim_on: stimulus starting time (in steps)
            stim_dur: stimulus duration (in steps)
            delay: delay b/w two stimuli (in steps)
            taus: time-constants (in steps)
            DeltaT: sampling rate
            lfp_power_target: target power for the LFP
    OUTPUT
        y: 1xT target signal
    """
    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']
    delay = settings['delay']

    if settings['power_target_period'] == 'full':
        y = np.ones((1, T)) # entire trial duration
    elif settings['power_target_period'] == 'delay':
        y = np.zeros((1, T))
        y[0, stim_on+stim_dur:stim_on+stim_dur+delay] = 1 # maintenance period
    else:
        raise ValueError("Invalid power_target_period. Choose either 'full' or 'delay'.")

    return np.squeeze(y)

def calculate_LFP_bandpower(settings, epsp):
    """
    TF1-graph-friendly CWT bandpower using Morlet wavelets implemented via conv1d.
    Returns [T] tensor of z-scored bandpower.
    """
    # ---- settings ----
    T         = settings['T']
    fs        = settings['fs']           # Hz
    dt        = 1.0 / fs
    fmin, fmax = 4.0, 100.0
    num_freqs = 40
    band_lo, band_hi = settings['lfp_power_target']  # e.g., [30, 80]
    stim_on   = settings['stim_on']

    # ---- epsp -> shape [1, T, 1] ----
    # epsp is length T, each element shape [1,1], so squeeze & stack:
    # pdb.set_trace()
    epsp_vec = tf.squeeze(tf.stack(epsp, axis=0))          # [T] or [T,] float32
    epsp_sig = tf.reshape(epsp_vec, [1, T, 1])                   # [batch=1, time=T, ch=1]

    # ---- frequencies & scales ----
    freqs = tf.exp(tf.linspace(tf.log(fmin), tf.log(fmax), num_freqs))  # [num_freqs]
    # Morlet central frequency for PyWavelets 'morl' ~ 0.8125, but we can treat as hyperparam.
    # Match PyWavelets by using that same fc:
    fc = tf.constant(0.8125, dtype=tf.float32)
    scales = fc / (freqs * dt)                                   # [num_freqs]

    # ---- build Morlet kernels (real & imag) ----
    # Window length: K cycles of the center frequency at each scale.
    # K=6 is a common choice. Use time support symmetric around 0.
    K = 6.0
    # For a Morlet at frequency f, approximate sigma_t ~ K/(2*pi*f)
    # Use a *shared* max half-width across freqs for "same" padding kernels.
    # Choose a kernel that covers the lowest freq sufficiently:
    f_low = fmin
    sigma_t_low = K / (2.0 * np.pi * f_low)
    half_width_sec = 4.0 * sigma_t_low   # ~±4 sigma
    kernel_len = tf.cast(tf.round(2.0 * half_width_sec * fs) + 1, tf.int32)  # odd length
    kernel_len = tf.maximum(kernel_len, 31)  # avoid too-short kernels
    # time vector centered at 0
    t_idx = tf.range(kernel_len, dtype=tf.float32) - tf.cast(kernel_len - 1, tf.float32)/2.0
    t_sec = t_idx / fs  # [L]

    # Morlet wavelet (complex): psi(t; f) = A * exp(-t^2/(2*sigma^2)) * exp(2j*pi*f*t)
    # We normalize each kernel to unit L2 energy so power is comparable across freqs.
    # Build per-frequency kernels (broadcast over [L, num_freqs]).
    two_pi = tf.constant(2.0*np.pi, tf.float32)
    # sigma_t per frequency (vector) — wider at low freq:
    sigma_t = K / (two_pi * freqs)  # [num_freqs]
    # Gaussian envelope
    env = tf.exp(-0.5 * tf.square(tf.expand_dims(t_sec,1) / tf.expand_dims(sigma_t,0)))  # [L, F]
    # carrier
    phase = two_pi * tf.expand_dims(t_sec,1) * tf.expand_dims(freqs,0)                   # [L, F]
    cos_part = env * tf.cos(phase)                                                       # [L, F]
    sin_part = env * tf.sin(phase)                                                       # [L, F]

    # L2 normalize each frequency kernel
    eps = 1e-8
    norm = tf.sqrt(tf.reduce_sum(tf.square(cos_part) + tf.square(sin_part), axis=0, keepdims=True) + eps)  # [1,F]
    cos_part = cos_part / norm
    sin_part = sin_part / norm

    # conv1d wants [filter_width, in_channels, out_channels]
    # in_channels=1; out_channels=F (one filter per frequency)
    cos_filt = tf.expand_dims(cos_part, 1)  # [L,1,F]
    sin_filt = tf.expand_dims(sin_part, 1)  # [L,1,F]

    # ---- convolution: "SAME" to keep T ----
    real_coeff = tf.nn.conv1d(epsp_sig, cos_filt, stride=1, padding='SAME')  # [1,T,F]
    imag_coeff = tf.nn.conv1d(epsp_sig, sin_filt, stride=1, padding='SAME')  # [1,T,F]

    power = tf.square(real_coeff) + tf.square(imag_coeff)  # [1,T,F]
    power = tf.squeeze(power, axis=0)                      # [T,F]

    # ---- select target band & average across freqs in band ----
    band_mask = tf.logical_and(freqs >= band_lo, freqs <= band_hi)         # [F]
    band_mask_f = tf.cast(band_mask, tf.float32)
    # Avoid empty band:
    denom = tf.maximum(tf.reduce_sum(band_mask_f), 1.0)
    band_power = tf.tensordot(power, band_mask_f/denom, axes=[[1],[0]])    # [T]

    # ---- baseline z-score (pure TF) ----
    # baseline: e.g., indices 10 : stim_on (exclusive)
    start = tf.constant(10, dtype=tf.int32)
    stop  = tf.cast(stim_on, tf.int32)
    base_slice = band_power[start:stop]                                     # [stop-start]
    base_mean  = tf.reduce_mean(base_slice) 
    base_std   = tf.math.reduce_std(base_slice) + 1e-8
    lfp_power_z = (band_power - base_mean) / base_std                       # [T]

    return lfp_power_z  # shape [T]

# def calculate_LFP_bandpower(settings, epsp):
#     """
#     Calculate the band-specific LFP power from the EPSP signal

#     INPUT
#         settings: dict containing the following keys
#             T: duration of a single trial (in steps)
#             stim_on: stimulus starting time (in steps)
#             stim_dur: stimulus duration (in steps)
#             delay: delay b/w two stimuli (in steps)
#             taus: time-constants (in steps)
#             DeltaT: sampling rate
#             fs: sampling rate (Hz)
#             lfp_power_target: target power for the LFP (min, max)
#         epsp: EPSP signal from the RNN model
#     OUTPUT
#         lfp_power: 1xT LFP power signal
#     """
#     T = settings['T']
#     fs = settings['fs']  # sampling frequency (Hz)
#     dt = 1 / fs  # time step (s)
#     fmin, fmax = 4, 100  # frequency range for the CWT
#     num_freqs = 40
#     band_lo, band_hi = settings['lfp_power_target']  # target band for LFP power
    
#     # nperseg = int(0.01 * fs)  # 10 ms segment length (delay is 50ms)
#     # f, t, s = scipy.signal.spectrogram(epsp, fs=fs, nperseg=nperseg, noverlap=nperseg//2, 
#     #                                    nfft=nperseg*20, scaling='density')
    
#     # fmin, fmax = 4, 100  # frequency range for the CWT
#     # num_freqs = 40
#     # freqs = np.logspace(fmin, fmax, num=num_freqs)  # logarithmically spaced frequencies
#     # w = 6 # number of cycles
#     # widths = (fs * w) / (2 * freqs * np.pi)
#     # cwt_mtx = scipy.signal.cwt(epsp, scipy.signal.morlet2, widths, w=w)
#     # power = np.abs(cwt_mtx)**2  # dimensions [frequencies x time]
    
    
    
#     freqs = np.logspace(np.log10(fmin), np.log10(fmax), num=num_freqs) # log-spaced freqs
    
#     if tf.executing_eagerly():
#         epsp_np = epsp.numpy()
#     else:
#         raise RuntimeError("epsp is symbolic — can't convert to NumPy in graph mode.")
#     fc = pywt.central_frequency('morl')
#     scales = fc / (freqs * dt)

#     # PyWavelets CWT
#     coeffs, freqs_out = pywt.cwt(epsp_np, scales, 'morl', sampling_period=dt)
#     power = np.abs(coeffs)**2  # dimensions [frequencies x time]
    
#     # calculate power for band of interest
#     idx = np.where((freqs_out >= band_lo) & (freqs_out <= band_hi))[0] # get freqs in band
#     band_power = np.mean(power[idx, :], axis=0)  # average power across frequencies in the band

#     # normalize the band power by the baseline period
#     baseline_period = slice(10, settings['stim_on'])  # baseline period before stimulus onset
#     baseline_mean = np.mean(band_power[baseline_period])
#     baseline_std = np.std(band_power[baseline_period])
#     lfp_power = (band_power - baseline_mean) / baseline_std  # z-score
#     lfp_power = tf.convert_to_tensor(lfp_power, dtype=tf.float32)
    
#     pdb.set_trace()

#     return tf.squeeze(lfp_power)

'''
CONSTRUCT TF GRAPH FOR TRAINING
'''
def construct_tf(fr_rnn, settings, training_params):
    """
    Method to construct a TF graph and return nodes with
    Dale's principle
    INPUT
        fr_rnn: firing-rate RNN class
        settings: dict containing the following keys
            T: duration of a single trial (in steps)
            stim_on: stimulus starting time (in steps)
            stim_dur: stimulus duration (in steps)
            delay: delay b/w two stimuli (in steps)
            taus: time-constants (in steps)
            DeltaT: sampling rate
        training_params: dictionary containing training parameters
            learning_rate: learning rate
    OUTPUT
        TF graph
    """

    # Task params
    T = settings['T']
    taus = settings['taus']
    DeltaT = settings['DeltaT']
    task = settings['task']

    # Training params
    learning_rate = training_params['learning_rate']

    # Excitatory units
    exc_idx_tf = tf.constant(np.where(fr_rnn.exc == True)[0], name='exc_idx', dtype=tf.int32)
    exc_idx = np.where(fr_rnn.exc == True)[0]

    # Inhibitory units
    inh_idx_tf = tf.constant(np.where(fr_rnn.inh == True)[0], name='inh_idx')
    som_inh_idx_tf = tf.constant(fr_rnn.som_inh, name='som_inh_idx')

    # Input node
    # XOR task
    if task == 'xor':
        stim = tf.placeholder(tf.float32, [2, T], name='u')

    # Sensory integration task
    elif task == 'mante':
        stim = tf.placeholder(tf.float32, [4, T], name='u')

    # Go-NoGo task
    elif task == 'go-nogo':
        stim = tf.placeholder(tf.float32, [1, T], name='u')

    # Target node
    z = tf.placeholder(tf.float32, [T,], name='target')
    y = tf.placeholder(tf.float32, [T,], name='lfp_target')

    # Initialize the decay synaptic time-constants (gaussian random).
    # This vector will go through the sigmoid transfer function.
    if len(taus) > 1:
        taus_gaus = tf.Variable(tf.random_normal([fr_rnn.N, 1]), dtype=tf.float32, 
            name='taus_gaus', trainable=True)
    elif len(taus) == 1:
        taus_gaus = tf.Variable(tf.random_normal([fr_rnn.N, 1]), dtype=tf.float32, 
            name='taus_gaus', trainable=False)
        print('Synaptic decay time-constants will not get updated!')

    # Synaptic currents, firing-rates, and EPSP
    x = [] # synaptic currents
    r = [] # firing-rates
    epsp = [] # EPSP
    x.append(tf.random_normal([fr_rnn.N, 1], dtype=tf.float32)/100)

    # Transfer function options
    if training_params['activation'] == 'sigmoid':
        r.append(tf.sigmoid(x[0]))
    elif training_params['activation'] == 'clipped_relu': 
        r.append(tf.clip_by_value(tf.nn.relu(x[0]), 0, 20))
    elif training_params['activation'] == 'softplus':
        r.append(tf.clip_by_value(tf.nn.softplus(x[0]), 0, 20))
        
    # Initialize EPSP
    epsp.append(tf.abs(tf.random.normal([1], dtype=tf.float32)/100))

    # Initialize recurrent weight matrix, mask, input & output weight matrices
    w = tf.get_variable('w', initializer = fr_rnn.W, dtype=tf.float32, trainable=True)
    m = tf.get_variable('m', initializer = fr_rnn.mask, dtype=tf.float32, trainable=False)
    som_m = tf.get_variable('som_m', initializer = fr_rnn.som_mask, dtype=tf.float32,
            trainable=False)
    w_in = tf.get_variable('w_in', initializer = fr_rnn.w_in, dtype=tf.float32, trainable=False)
    w_out = tf.get_variable('w_out', initializer = fr_rnn.w_out, dtype=tf.float32, 
            trainable=True)

    b_out = tf.Variable(0, dtype=tf.float32, name='b_out', trainable=True)

    # Forward pass
    o = [] # output (i.e. weighted linear sum of rates, r)
    for t in range(1, T):
        if fr_rnn.apply_dale == True:
            # Parametrize the weight matrix to enforce exc/inh synaptic currents
            w = tf.nn.relu(w)

        # next_x is [N x 1]
        ww = tf.matmul(w, m)
        ww = tf.multiply(ww, som_m)

        # Pass the synaptic time constants thru the sigmoid function
        if len(taus) > 1:
            taus_sig = tf.sigmoid(taus_gaus)*(taus[1] - taus[0]) + taus[0]
        elif len(taus) == 1: # one scalar synaptic decay time-constant
            taus_sig = taus[0]

        next_x = tf.multiply((1 - DeltaT/taus_sig), x[t-1]) + \
                tf.multiply((DeltaT/taus_sig), ((tf.matmul(ww, r[t-1]))\
                + tf.matmul(w_in, tf.expand_dims(stim[:, t-1], 1)))) +\
                tf.random_normal([fr_rnn.N, 1], dtype=tf.float32)/10
        x.append(next_x)

        if training_params['activation'] == 'sigmoid':
            r.append(tf.sigmoid(next_x))
        elif training_params['activation'] == 'clipped_relu': 
            r.append(tf.clip_by_value(tf.nn.relu(next_x), 0, 20))
        elif training_params['activation'] == 'softplus':
            r.append(tf.clip_by_value(tf.nn.softplus(next_x), 0, 20))

        r_exc = tf.gather(r[t-1], exc_idx_tf)
        ww_exc = tf.gather(ww, exc_idx_tf, axis=1)
        next_epsp = tf.multiply((1 - DeltaT/taus_sig), x[t-1]) + \
                    tf.multiply((DeltaT/taus_sig), 
                                ((tf.matmul(ww_exc, r_exc)))) 
        next_epsp = tf.reduce_mean(next_epsp, axis=0, keepdims=False)  # average over excitatory neurons
        
        # next_epsp = tf.multiply((1 - DeltaT/taus_sig), 
        #                         tf.expand_dims(x[t-1], 1)) + \
        #             tf.multiply((DeltaT/taus_sig), 
        #                         ((tf.multiply(ww, tf.expand_dims(r[t-1], 1))))) # [N x 1]
        # next_epsp = tf.reduce_mean(next_epsp, axis=0, keepdims=True)  # average over all neurons
        
        epsp.append(next_epsp)

        next_o = tf.matmul(w_out, r[t]) + b_out
        o.append(next_o)

    return stim, z, y, x, r, epsp, o, w, w_in, m, som_m, w_out, b_out, taus_gaus

'''
DEFINE LOSS AND OPTIMIZER
'''
def loss_op(o, z, epsp, y, training_params, settings):
    """
    Method to define loss and optimizer for target signal (output and epsp)
    INPUT
        o: list of output values
        z: target values
        training_params: dictionary containing training parameters
            learning_rate: learning rate
        epsp: list of EPSP values
        y: target LFP bandpower values

    OUTPUT
        loss: loss function
        training_op: optimizer
    """
    # get epsp band power
    lfp_power = calculate_LFP_bandpower(settings, epsp)
    
    # Loss function
    loss = tf.zeros(1)
    loss_fn = training_params['loss_fn']
    # for i in range(0, len(o)):
    #     if loss_fn.lower() == 'l1':
    #         loss += tf.norm(o[i] - z[i])
    #     elif loss_fn.lower() == 'l2':
    #         loss += tf.square(o[i] - z[i]) + tf.norm(lfp_power[i] - y[i])
    # if loss_fn.lower() == 'l2':
    #     loss = tf.sqrt(loss)
    o_full = [tf.zeros_like(o[0])] + o  # length T
    o_vec  = tf.squeeze(tf.stack(o_full, axis=0))  # [T]
    if loss_fn == 'l1':
        loss_out = tf.reduce_sum(tf.abs(o_vec - z))
        loss_lfp = tf.reduce_sum(tf.square(lfp_power - y))  # L2 on bandpower target
        loss = loss_out + loss_lfp
    else:  # 'l2'
        loss_out = tf.reduce_sum(tf.square(o_vec - z))
        loss_lfp = tf.reduce_sum(tf.norm(lfp_power - y)) # norm for bandpower
        # loss_lfp = tf.reduce_sum(tf.square(lfp_power - y))
        loss = tf.sqrt(1.5*loss_out + loss_lfp + 1e-12)
        # loss = tf.sqrt(loss_out + loss_lfp + 1e-12)
        
        # loss_out = tf.reduce_sum(tf.square(o_vec - z))
        # loss_lfp = tf.reduce_sum(tf.square(lfp_power - y))
        # loss = tf.sqrt(1.5*loss_out + loss_lfp + 1e-12)

    # Optimizer function
    with tf.name_scope('ADAM'):
        optimizer = tf.train.AdamOptimizer(learning_rate = training_params['learning_rate']) 

    training_op = optimizer.minimize(loss) 

    return loss, loss_out, loss_lfp, training_op

'''
EVALUATE THE TRAINED MODEL
NOTE: NEED TO BE UPDATED!!
'''
def eval_tf(model_dir, settings, u, lesion='', lesion_perc=0.5, calc_epsp=True):
    """
    Method to evaluate a trained TF graph
    INPUT
        model_dir: full path to the saved model .mat file
        stim_params: dictionary containig the following keys
        u: 12xT stimulus matrix
            NOTE: There are 12 rows (one per dot pattern): 6 cues and 6 probes.
    OUTPUT
        o: 1xT output vector
    """
    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']
    # delay = settings['delay']
    DeltaT = settings['DeltaT']

    # Load the trained mat file
    var = scipy.io.loadmat(model_dir)

    # Get some additional params
    N = var['N'][0][0]
    # exc_ind = [bool(i) for i in var['exc']]

    # Get the delays
    taus_gaus = var['taus_gaus']
    taus = var['taus'][0] # tau [min, max]
    taus_sig = (1/(1+np.exp(-taus_gaus))*(taus[1] - taus[0])) + taus[0] 

    # Synaptic currents and firing-rates; + EPSP
    x = np.zeros((N, T)) # synaptic currents
    r = np.zeros((N, T)) # firing-rates
    epsp = np.zeros((T)) # EPSP
    x[:, 0] = np.random.randn(N, )/100
    r[:, 0] = 1/(1 + np.exp(-x[:, 0]))
    epsp[0] = np.abs(np.random.randn(1)/100)
    # r[:, 0] = np.minimum(np.maximum(x[:, 0], 0), 1) #clipped relu
    # r[:, 0] = np.clip(np.minimum(np.maximum(x[:, 0], 0), 1), None, 10) #clipped relu
    # r[:, 0] = np.clip(np.log(np.exp(x[:, 0])+1), None, 10) # softplus
    # r[:, 0] = np.minimum(np.maximum(x[:, 0], 0), 6)/6 #clipped relu6


    # Output
    o = np.zeros((T, ))
    o_counter = 0

    # Recurrent weights and masks
    # w = var['w0'] #!!!!!!!!!!!!
    w = var['w']

    m = var['m']
    som_m = var['som_m']
    som_N = var['som_N'][0][0]

    # Identify excitatory/inhibitory neurons
    exc = var['exc']
    exc_ind = np.where(exc == 1)[0]
    exc_idx_tf = tf.constant(exc_ind, name='exc_idx', dtype=tf.int32)
    inh = var['inh']
    inh_ind = np.where(inh == 1)[0]
    som_inh_ind = inh_ind[:som_N]


    # lesioning
    if len(lesion) != 0:
        lesion_mask = np.ones_like(w)
        if lesion == 'ii': # Inh -> Inh
            lesion_mask[np.ix_(inh_ind, inh_ind)] = lesion_perc
        elif lesion == 'ei': # Inh -> Exc
            lesion_mask[np.ix_(exc_ind, inh_ind)] = lesion_perc
        elif lesion == 'ie':  # Exc -> Inh
            lesion_mask[np.ix_(inh_ind, exc_ind)] = lesion_perc 
        elif lesion == 'ee': # Exc -> Exc
            lesion_mask[np.ix_(exc_ind, exc_ind)] = lesion_perc
        w = np.multiply(w, lesion_mask)

    for t in range(1, T):
        # next_x is [N x 1]
        ww = np.matmul(w, m)
        ww = np.multiply(ww, som_m)

        # next_x = (1 - DeltaT/tau)*x[:, t-1] + \
                # (DeltaT/tau)*(np.matmul(ww, r[:, t-1]) + \
                # np.matmul(var['w_in'], u[:, t-1])) + \
                # np.random.randn(N, )/10

        next_x = np.multiply((1 - DeltaT/taus_sig), np.expand_dims(x[:, t-1], 1)) + \
                np.multiply((DeltaT/taus_sig), ((np.matmul(ww, np.expand_dims(r[:, t-1], 1)))\
                + np.matmul(var['w_in'], np.expand_dims(u[:, t-1], 1)))) +\
                np.random.randn(N, 1)/10
        
        if calc_epsp == True:
            next_epsp = np.multiply((1 - DeltaT/taus_sig), np.expand_dims(x[:,t-1], 1)) + \
                    np.multiply((DeltaT/taus_sig), ((np.matmul(ww[:,exc_ind], np.expand_dims(r[exc_ind, t-1], 1))))) 
            next_epsp = np.mean(next_epsp)  # average over all neurons
            epsp[t] = next_epsp
            # next_epsp = tf.multiply((1 - DeltaT/taus_sig), tf.expand_dims(x[:, t-1], 1)) + \
            #         tf.multiply((DeltaT/taus_sig[:,:]), 
            #                     ((tf.matmul(ww, tf.expand_dims(r[:, t-1], 1)))))
            # next_epsp = tf.reduce_mean(next_epsp, axis=0, keepdims=True)  # average over excitatory neurons
            
        x[:, t] = np.squeeze(next_x)
        r[:, t] = 1/(1 + np.exp(-x[:, t]))
        


        # r[:, t] = np.minimum(np.maximum(x[:, t], 0), 1)
        # r[:, t] = np.clip(np.minimum(np.maximum(x[:, t], 0), 1), None, 10)
        # r[:, t] = np.clip(np.log(np.exp(x[:, t])+1), None, 10) # softplus
        # r[:, t] = np.minimum(np.maximum(x[:, t], 0), 6)/6


        wout = var['w_out']
        wout_exc = wout[0, exc_ind]
        wout_inh = wout[0, inh_ind]
        r_exc = r[exc_ind, :]
        r_inh = r[inh_ind, :]

        o[o_counter] = np.matmul(wout, r[:, t]) + var['b_out']
        # o[o_counter] = np.matmul(wout_exc, r[exc_ind, t]) + var['b_out'] # excitatory output
        # o[o_counter] = np.matmul(wout_inh, r[inh_ind, t]) + var['b_out'] # inhibitory output
        o_counter += 1
    return x, r, o, epsp

