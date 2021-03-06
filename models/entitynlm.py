import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from torch.distributions import Normal

use_cuda = torch.cuda.is_available()

class EntityNLM(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=256, entity_size=256, num_layers=1, dropout=0.5):
        super(EntityNLM, self).__init__()

        assert hidden_size == entity_size, "hidden_size should be equal to entity_size"

        self.embed = nn.Embedding(vocab_size, embed_size)

        self.rnn = nn.LSTM( input_size=embed_size, 
                            hidden_size=hidden_size,
                            num_layers=num_layers )

        self.x = nn.Linear(hidden_size, vocab_size) # Linear
        self.r = nn.Bilinear(hidden_size, hidden_size, 1) # Is Entity? Binary
        self.l = nn.Linear(2 * hidden_size, 25) # Remaining Length Prediction, Categorial         
        self.e = nn.Bilinear(hidden_size, hidden_size, 1) # Randomly Sample from Entity Set

        # For r embeddings
        r_embeddings = Variable(torch.FloatTensor(2, hidden_size), requires_grad=True)
        if use_cuda: r_embeddings = r_embeddings.cuda()
        init.xavier_uniform(r_embeddings,gain=np.sqrt(2)) 
        self.r_embeddings = r_embeddings

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
        # Get r1
        r1 = self.r_embeddings[1]
        
        # Sample from normal distribution
        # Code implementation
        e_var = r1 + torch.normal(means=torch.zeros_like(r1), std=0.01).view(1,-1)
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

    def predict_type(self, h_t):
        pred_r = self.r( self.dropout(self.r_embeddings), self.dropout(h_t.expand_as(self.r_embeddings)) ).transpose(0,1)
        return pred_r

    def predict_entity(self, h_t, sent_idx):
        # Concatenate entities to a block
        entity_stack = torch.cat(self.entities)

        dist_feat = self.get_dist_feat(sent_idx) # Sentence distance feature

        pred_e = self.e(self.dropout(entity_stack), self.dropout(h_t.expand_as(entity_stack)) ) + self.w_dist(self.dropout(dist_feat))
        
        pred_e = pred_e.transpose(0,1)
        return pred_e

    def predict_length(self, h_t, entity_embedding):
        pred_l = self.l(self.dropout(torch.cat((h_t, entity_embedding),dim=1)))
        return pred_l

    def predict_word(self, next_entity_index, h_t, entity_current):
        # Word Prediction
        pred_x = self.x(self.dropout(h_t + self.Te(self.dropout(entity_current))))
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
        dim1 = 1
        dim2 = batch_size
        dim3 = self.rnn.hidden_size

        var1 = Variable(torch.zeros(dim1,dim2,dim3))
        var2 = Variable(torch.zeros(dim1,dim2,dim3))

        if use_cuda:
            var1 = var1.cuda()
            var2 = var2.cuda()

        return var1, var2


def build_model(vocab_size, args, dictionary):
    model = EntityNLM(vocab_size=vocab_size, 
                        embed_size=args.embed_dim, 
                        hidden_size=args.hidden_size,
                        entity_size=args.entity_size,
                        dropout=args.dropout)

    if use_cuda:
        model = model.cuda()
    
    if args.model_path is not None:
        print("Loading from {}".format(args.model_path))
        model.load_state_dict(torch.load(args.model_path))
    elif args.pretrained: 
        model.load_pretrained(dictionary)
    
    return model