import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from torch.distributions import Normal

import torch
import torchtext.vocab as vocab

use_cuda = True if torch.cuda.is_available() else False

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
    def __init__(self, vocab_size, embed_size=256, hidden_size=128, entity_size=128, dropout=0.5):
        super(EntityNLM, self).__init__()

        assert embed_size == hidden_size and hidden_size == entity_size, "embed_size, hidden_size, entity_size should be equal"

        self.embed = nn.Embedding(vocab_size, embed_size)

        self.rnn = nn.LSTMCell(input_size=embed_size, hidden_size=hidden_size)

        self.x = nn.Linear(hidden_size, vocab_size) # Linear      
        self.e = nn.Bilinear(hidden_size, hidden_size, 1) # Randomly Sample from Entity Set

        # Entity Creation
        self.r_embedding = Variable(torch.FloatTensor(hidden_size,), requires_grad=True)
        if use_cuda: 
            self.r_embeddings = self.r_embeddings.cuda()

        # For distance feature
        self.w_dist = nn.Linear(1, 1)

        self.delta = nn.Bilinear(entity_size, hidden_size, 1) # Delta Matrix

        self.Te = nn.Linear(entity_size, hidden_size)
        self.Tc = nn.Linear(entity_size, hidden_size)

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
    
        self.entities = []
        self.entities_dist = []

        self.init_weights()

    def create_entity(self, nsent=0.0):
        # Get r1
        
        # Sample from normal distribution
        # Code implementation
        e_var = self.r_embedding + torch.normal(means=torch.zeros_like(self.r_embedding), std=0.01).view(1,-1)
        e_var /= torch.norm(e_var)
        # Not sure if this line is redundant
        if use_cuda: 
            e_var = e_var.cuda()

        self.entities.append(e_var)
        self.entities_dist.append(nsent)

    def update_entity(self, entity_idx, h_t, sent_idx):
        # Update Entity Here
        entity_embedding = self.entities[ entity_idx ]
        delta = self.sigmoid(
            self.delta(entity_embedding, h_t)
        ).view(-1)

        # New Entity Embedding
        updated_embedding = delta * entity_embedding + ( 1 - delta ) * h_t
        updated_embedding /= torch.norm(updated_embedding)

        # Update Entity List
        self.entities[ entity_idx ] = updated_embedding
        self.entities_dist[ entity_idx ] = sent_idx # Update nearest entity apperance in sentencef

        return updated_embedding

    def get_entity(self, entity_idx):
        if entity_idx >= len(self.entities):
            return self.entities[0] # Dummy Embedding
        else:
            return self.entities[entity_idx]

    def get_dist_feat(self, nsent):
        dist_feat = Variable((torch.FloatTensor(self.entities_dist) - nsent).view(-1,1), requires_grad=False) 
        if use_cuda: 
            dist_feat = dist_feat.cuda()
        return dist_feat

    def predict_entity(self, h_t, sent_idx):
        # Concatenate entities to a block
        entity_stack = torch.cat(self.entities)

        dist_feat = self.get_dist_feat(sent_idx) # Sentence distance feature

        pred_e = self.e(self.dropout(entity_stack), self.dropout(h_t.expand_as(entity_stack)) ) + self.w_dist(self.dropout(dist_feat))
        
        pred_e = pred_e.transpose(0,1)
        return pred_e

    def predict_word(self, h_t):
        # Word Prediction
        pred_x = self.x(self.dropout(h_t))
        #pred_x = self.x(self.dropout(h_t + self.Te(self.dropout(entity_current)))) # May use this later
        return pred_x

    def clear_entities(self):
        # Refresh
        self.entities = []
        self.entities_dist = []

    def init_weights(self):
        # Glorot Uniform
        for param in self.parameters():
            if param.dim() > 1:
                init.xavier_uniform(param,gain=np.sqrt(2))  
    
    def init_hidden_states(self, batch_size):
        dim1 = batch_size
        dim2 = self.rnn.hidden_size

        var1 = Variable(torch.zeros(dim1,dim2))
        var2 = Variable(torch.zeros(dim1,dim2))

        if use_cuda:
            var1 = var1.cuda()
            var2 = var2.cuda()

        return var1, var2