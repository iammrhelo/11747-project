import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.5):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
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
        return ( Variable(torch.zeros(dim1,dim2,dim3)), Variable(torch.zeros(dim1,dim2,dim3)) )
