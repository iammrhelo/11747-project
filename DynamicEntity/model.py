import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

class RNNLM(nn.Module):
    def __init__(self, rnn, vocab_size, embed_size, hidden_size, num_layers, dropout=0.5):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        if rnn == "LSTM":
            self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        elif rnn == "GRU":
            self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size) # Linear
        self.dropout = nn.Dropout(dropout)
        
        self.init_weights()

    def init_weights(self):
        for param in self.parameters():
            if param.dim() > 1:
                init.xavier_uniform(param,gain=np.sqrt(2))  

    def forward(self, x, hidden_states):
        embed_x = self.dropout(self.embed(x))
        
        rnn_output, hidden_states = self.rnn(embed_x, hidden_states)
        
        output = self.linear(self.dropout(rnn_output))

        return output, hidden_states
    
    def init_hidden_states(self, batch_size):
        dim1 = self.rnn.num_layers
        dim2 = batch_size
        dim3 = self.rnn.hidden_size
        
        if self.rnn.__class__ == nn.LSTM:
            return ( Variable(torch.zeros(dim1,dim2,dim3)), Variable(torch.zeros(dim1,dim2,dim3)) )
        elif self.rnn.__class__ == nn.GRU:
            return Variable(torch.zeros(dim1,dim2,dim3) ) 


class EntityNLM(nn.Module):
    def __init__(self, rnn, vocab_size, embed_size=300, hidden_size=128, num_layers=2, dropout=0.5):
        super(EntityNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=False)
        
        self.x = nn.Linear(hidden_size, vocab_size) # Linear
        self.r = nn.Linear(hidden_size, 1) # Is Entity?
        self.l = nn.Linear(hidden_size, 25) # Remaining length
        self.e = nn.Linear(hidden_size, 1) # Randomly Sample from Entity Set

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        
        self.init_weights()

    def init_weights(self):
        # Glorot Uniform
        for param in self.parameters():
            if param.dim() > 1:
                init.xavier_uniform(param,gain=np.sqrt(2))  

    def forward(self, x, hidden_states):
        embed_x = self.dropout(self.embed(x))
        
        rnn_output, hidden_states = self.rnn(embed_x, hidden_states)
        
        X = self.x(self.dropout(rnn_output))
        R = self.sigmoid(self.r(self.dropout(rnn_output)))
        L = self.l(self.dropout(rnn_output))
        E = self.e(self.dropout(rnn_output))

        return X, R, L, E, hidden_states
    
    def init_hidden_states(self, batch_size):
        dim1 = self.rnn.num_layers
        dim2 = batch_size
        dim3 = self.rnn.hidden_size
        
        return ( Variable(torch.zeros(dim1,dim2,dim3)), Variable(torch.zeros(dim1,dim2,dim3)) )
