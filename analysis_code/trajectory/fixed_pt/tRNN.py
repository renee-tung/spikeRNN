import torch
import numpy as np
import torch.nn as nn
import pdb
from torch.nn.modules.rnn import RNNBase
'''
Implementation for custom RNN module implemented by Kim et al. 

- Sigmoid non-linearity
- Neural decay time constant (tau) is individually fit for each neuron

'''

class tRNN(RNNBase):
    def __init__(self,mode,input_size,hidden_size,taus_sig, ww, w_in, w_out, b_out):
        super().__init__(mode,input_size,hidden_size)
        self.taus_sig = torch.Tensor(taus_sig)
        self.ww = torch.Tensor(ww)
        self.w_in = torch.Tensor(w_in)
        self.w_out = torch.Tensor(w_out)
        self.b_out = torch.Tensor(b_out)

    def forward(self,input,hidden):
        hidden = torch.squeeze(hidden)
        input = torch.squeeze(input)
        if len(hidden.shape) == 1:
            hidden = torch.unsqueeze(hidden,0)
        if len(input.shape) == 1:
            input = torch.unsqueeze(input,0)                
        rate = torch.transpose(torch.sigmoid(hidden),0,1)
        x_update = torch.transpose(torch.mul((1 - 1./self.taus_sig), torch.transpose(hidden,0,1)),0,1)      
        rate_update = torch.transpose(torch.matmul(self.ww, rate),0,1)
        input_update = torch.transpose(torch.matmul(self.w_in, torch.transpose(input,0,1)),0,1)
        next_hidden = x_update + torch.transpose(torch.mul(   (1./self.taus_sig), torch.transpose((rate_update + input_update),0,1)),0,1)
        next_rate = torch.sigmoid(next_hidden)
        output = torch.transpose(torch.matmul(self.w_out, torch.transpose(next_rate,0,1)),0,1) + self.b_out         
        return output, torch.unsqueeze(next_hidden,0)
    

    def forward(self, input, hidden):
        # Squeeze input and hidden to ensure they are the correct shapes
        hidden = torch.squeeze(hidden)
        input = torch.squeeze(input)
        
        # Ensure both are at least 2D tensors (if they are 1D)
        if len(hidden.shape) == 1:
            hidden = torch.unsqueeze(hidden, 0)
        if len(input.shape) == 1:
            input = torch.unsqueeze(input, 0)

        # Compute rate (sigmoid of hidden)
        rate = torch.sigmoid(hidden).transpose(0, 1)  # Transpose for appropriate shape
        
        # Compute x_update
        x_update = (1 - 1./ self.taus_sig) * hidden.transpose(0, 1)  # Element-wise multiplication
        
        # Compute rate_update
        rate_update = torch.matmul(self.ww, rate)  # Matrix multiplication
        
        # Compute input_update
        input_update = torch.matmul(self.w_in, input.transpose(0, 1))  # Matrix multiplication
        
        # Compute next_hidden
        next_hidden = x_update + (1./ self.taus_sig) * (rate_update + input_update).transpose(0, 1)
        
        # Compute next_rate
        next_rate = torch.sigmoid(next_hidden)
        
        # Compute output
        output = torch.matmul(self.w_out, next_rate.transpose(0, 1)) + self.b_out
        
        return output.transpose(0, 1), next_hidden.unsqueeze(0)  # Return output and next hidden state
