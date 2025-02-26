import os
import pdb
import sys
import argparse
import numpy as np
import scipy.io as si
import torch
rootpath = os.path.join(os.getcwd(), '..')
sys.path.append(rootpath)
from tRNN import tRNN

def load_RNN_model(rnn_path):
    base_folder = '/home/cfxuser/Documents/Neurips/models/'
    RNN_model_file = 'Task_instr_N_1000_Taus_4.0_25.0_Act_sigmoid_2023_04_24_013958'    
    #rnn_data = si.loadmat(os.path.join(base_folder, RNN_model_file))
    rnn_data = si.loadmat(rnn_path)
    # Loading model parameters
    taus_gaus = torch.from_numpy(rnn_data['taus_gaus'])
    taus = torch.from_numpy(rnn_data['taus'][0])
    w = torch.from_numpy(rnn_data['w'])
    m = torch.from_numpy(rnn_data['m'])
    w_in = torch.from_numpy(rnn_data['w_in'])
    w_out = torch.from_numpy(rnn_data['w_out'])
    b_out = torch.from_numpy(rnn_data['b_out'])
    taus_sig = torch.sigmoid(taus_gaus)*(taus[1] - taus[0]) + taus[0] # Neural time constant
    ww = torch.matmul(w, m) # Recurrent weight matrix
    # Custom RNN class for sigmoid non-linearity
    torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_size = 2
    hidden_size = 200
    mode = 'LSTM'
    rnn = tRNN(mode,input_size,hidden_size,taus_sig, ww, w_in, w_out, b_out)
    #rnn = tRNN(taus_sig, ww, w_in, w_out, b_out)
    
    # For debugging
    #h0 = torch.Tensor(np.zeros(1000))
    #input = torch.Tensor([0,0,1])
    #output, hn = rnn(input, h0)
    return rnn
