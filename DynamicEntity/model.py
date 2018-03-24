import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from torch.distributions import Normal

import torch
import torchtext.vocab as vocab

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
    def __init__(self, vocab_size, embed_size=300, hidden_size=128, entity_size=128, dropout=0.5, use_cuda=False):
        super(EntityNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)

        self.rnn = nn.LSTMCell(input_size=embed_size, hidden_size=hidden_size)

        self.x = nn.Linear(hidden_size, vocab_size) # Linear
        self.r = nn.Linear(hidden_size, 1) # Is Entity? Binary
        self.l = nn.Linear(2 * hidden_size, 25) # Remaining Length Prediction, Categorial 
        
        self.e = nn.Bilinear(hidden_size, hidden_size, 1, bias=False) # Randomly Sample from Entity Set

        self.delta = nn.Bilinear(entity_size, hidden_size, 1, bias=False) # Delta Matrix

        self.Te = nn.Linear(entity_size, hidden_size, bias=False)
        self.Tc = nn.Linear(entity_size, hidden_size, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
    
        self.entities = []
        self.entities_dist = []

        self.use_cuda = use_cuda

        self.init_weights()

    def load_pretrained(self, dictionary):
        # Load pretrained vectors for embedding layer
        glove = vocab.GloVe(name='6B', dim=self.embed.embedding_dim)

        # Build weight matrix here
        pretrained_weight = self.embed.weight.data
        for word, idx in dictionary.items():
            if word.lower() in glove.stoi:     
                vector = glove.vectors[ glove.stoi[word.lower()] ]
                pretrained_weight[ idx ] = vector

        self.embed.weight = nn.Parameter(pretrained_weight)

    def create_entity(self, nsent=0.0):
        
        # Sample from normal distribution
        def sample_normal():
            # Sample from normal distribution
            e = np.random.normal(loc=0.0,scale=0.01,size=(1,self.rnn.hidden_size))
            # Project to unit ball
            e /= np.linalg.norm(e)
            # Convert to PyTorch Tensor
            e_tensor = torch.from_numpy(e).type(torch.FloatTensor)
            return e_tensor

        e_tensor = sample_normal()

        e_var = Variable(e_tensor, requires_grad=True)
        if self.use_cuda:
            e_var = e_var.cuda()

        self.entities.append(e_var)
        self.entities_dist.append(nsent)
        return e_tensor

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
        self.entities_dist[ entity_idx ] = sent_idx

        return updated_embedding

    def get_entity(self, entity_idx):
        if entity_idx >= len(self.entities):
            return self.entities[0]
        else:
            return self.entities[entity_idx]

    def get_dist_feat(self, nsent):
        dist_feat = Variable((torch.FloatTensor(self.entities_dist) - nsent).view(-1,1), requires_grad=False) 
        if self.use_cuda:
            dist_feat = dist_feat.cuda()

        return dist_feat


    def predict_type(self, h_t):
        pred_r = self.sigmoid(
            self.r(
                self.dropout(h_t)
            )
        )
        pred_r = torch.squeeze(pred_r)
        return pred_r

    def predict_entity(self, h_t, sent_idx, lambda_dist):
        # Concatenate entities to a block
        entity_stack = torch.cat(self.entities)

        dist_feat = self.get_dist_feat(sent_idx)

        pred_e = self.e(entity_stack, h_t.expand_as(entity_stack)) + torch.exp(dist_feat * lambda_dist)
        pred_e = pred_e.transpose(0,1)

        return pred_e

    def predict_length(self, h_t, entity_embedding):
        pred_l = self.l(
            self.dropout(
                torch.cat((h_t, entity_embedding),dim=1)
                )
            )
        return pred_l

    def predict_word(self, next_entity_index, h_t, last_entity):
        # Word Prediction
        pred_x = None
        if next_entity_index > 0:
            entity_embedding = self.get_entity(next_entity_index)
            
            pred_x = self.x(
                h_t + self.Te(
                    self.dropout(
                        entity_embedding)
                    )
                )

        else:
            pred_x = self.x(
                h_t + self.Tc(
                    self.dropout(
                        last_entity)
                    )
                )
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

        if self.use_cuda:
            var1 = var1.cuda()
            var2 = var2.cuda()

        return var1, var2
