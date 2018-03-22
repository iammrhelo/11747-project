import argparse
import math
import os
import random

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable

from corpus import Corpus
from model import EntityNLM

device = 0 if torch.cuda.is_available() else -1

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=str,default='./data/InScript')
    parser.add_argument('--rnn',type=str,default="LSTM",help="GRU | LSTM")
    parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--bptt_len',type=int,default=15)
    parser.add_argument('--embed_dim',type=int,default=300)
    parser.add_argument('--hidden_size',type=int,default=128)
    parser.add_argument('--num_layers',type=int,default=2)
    parser.add_argument('--dropout',type=float,default=0.5)
    parser.add_argument('--num_epochs',type=int,default=10)
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--early_stop',type=int,default=3)
    parser.add_argument('--shuffle',action="store_true",default=False)
    args = parser.parse_args()
    return args

args = parse_arguments()

##################
#  Data Loading  #
##################
data = args.data
dict_pickle = os.path.join(data,'train','dict.pickle')

# Set Corpus
train_corpus = Corpus(os.path.join(data,'train'), dict_pickle)
#valid_corpus = Corpus(os.path.join(data,'valid'),dict_pickle)
#test_corpus = Corpus(os.path.join(data,'test'),dict_pickle)

vocab_size = len(train_corpus.dictionary)+1 # 0 for unknown

shuffle = args.shuffle

##################
#   Model Setup  #
##################
embed_dim = args.embed_dim
hidden_size = args.hidden_size
num_layers = args.num_layers
dropout = args.dropout

model = EntityNLM(vocab_size=vocab_size, embed_size=128, hidden_size=128,dropout=0.5)

lambda_dist = 1e-6

#####################
#  Training Config  #
#####################
num_epochs = args.num_epochs
lr = args.lr
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

crossentropy = nn.CrossEntropyLoss()
binarycrossentropy = nn.BCELoss()

def repack(h_t, c_t):
    return Variable(h_t.data), Variable(c_t.data)

####################
#   Main Program   #
####################
for epoch in range(1,num_epochs+1,1):
    print("Epoch",epoch)

    epoch_loss = 0
    # Training
    model.train()

    # Shuffle Documents Here
    if shuffle: random.shuffle(train_corpus.documents)
    
    for doc in tqdm(train_corpus.documents):
        
        doc_loss = 0

        X, R, E, L = doc[0]

        nsent = len(X)
        
        # For every document
        h_t, c_t = model.init_hidden_states(1)
        model.create_entity() # Dummy
        last_entity = model.entities[0]

        # Check
        assert len(model.entities) == 1 # Only 1 dummy entity

        for sent_idx in range(nsent):
            # Learn for every sentence
            losses = []
            optimizer.zero_grad()

            h_t, c_t = repack(h_t,c_t)

            X_tensor = Variable(torch.from_numpy(np.array(X[sent_idx])).type(torch.LongTensor))
            R_tensor = Variable(torch.from_numpy(np.array(R[sent_idx])).type(torch.FloatTensor))
            E_tensor = Variable(torch.from_numpy(np.array(E[sent_idx])).type(torch.LongTensor))
            L_tensor = Variable(torch.from_numpy(np.array(L[sent_idx])).type(torch.LongTensor))

            for pos in range(0,len(X[sent_idx])-1): # 1 to N-1
                curr_x = X_tensor[pos]
                curr_r = R_tensor[pos]
                curr_e = E_tensor[pos]
                curr_l = L_tensor[pos]

                next_x = X_tensor[pos+1]
                next_r = R_tensor[pos+1]
                next_e = E_tensor[pos+1]
                next_l = L_tensor[pos+1]

                # Forward and Get Hidden State
                embed_curr_x = model.embed(curr_x)
                h_t, c_t = model.rnn(embed_curr_x, (h_t, c_t))

                # Update Entity
                if curr_r.data[0] > 0 and curr_e.data[0] > 0:
                    # Next Entity Type

                    entity_idx = int(curr_e.data[0])
                    try:
                        assert entity_idx == curr_e.data[0] and entity_idx <= len(model.entities)
                    except:
                        import pdb; pdb.set_trace()

                    # Create if it's a new entity
                    if entity_idx == len(model.entities):
                        model.create_entity(nsent=sent_idx)
                    
                    # Update Entity Here
                    last_entity = model.update_entity(entity_idx, h_t, sent_idx)
                
                # l == 1, End of Mention
                if curr_l.data[0] == 1:

                    mention_length = int(curr_l.data[0])
                    
                    assert mention_length == curr_l.data[0], "{} : {}".format(mention_length, curr_l.data[0])

                    pred_r = model.predict_type(h_t)
                    
                    # TODO: OK
                    type_loss = binarycrossentropy(pred_r,next_r)
                    losses.append(type_loss)
                            
                    # Entity Prediction
                    if next_r.data[0] > 0: # If the next word is an entity
                        
                        next_entity_index = int(next_e.data[0])
                        assert next_entity_index == next_e.data[0]

                        # Concatenate entities to a block
                        pred_e = model.predict_entity(next_entity_index, h_t, sent_idx, lambda_dist)

                        if next_entity_index < len(model.entities):
                            next_e = Variable(torch.LongTensor([next_entity_index]), requires_grad=False)
                        else:
                            next_e = Variable(torch.zeros(1).type(torch.LongTensor), requires_grad=False)

                        # TODO: FAILURE
                        e_loss = crossentropy(pred_e, next_e)
                        losses.append(e_loss)

                    # Entity Length Prediction
                    if int(next_e.data[0]) > 0: # Has Entity

                        # Get entity embedding
                        entity_idx = int(next_e.data[0])
                        entity_embedding = model.entities[entity_idx] if entity_idx < len(model.entities) else model.entities[0]

                        pred_l = model.predict_length(h_t, entity_embedding)
                        # TODO: OK
                        losses.append(crossentropy(pred_l, next_l))

                # Word Prediction
                next_entity_index = int(next_e.data[0])
                assert next_entity_index == next_e.data[0]

                pred_x = model.predict_word(next_entity_index, h_t, last_entity)

                # TODO: OK
                x_loss = crossentropy(pred_x, next_x)
                losses.append(x_loss)
            
            last_entity = h_t # Take hidden state as last entity embedding for next sentence

            if len(losses):
                total_loss = sum(losses)
                total_loss.backward(retain_graph=True)
                optimizer.step()
                
                doc_loss += total_loss.data[0]
        
        # End of document
        # Clear Entities
        model.clear_entities()

    epoch_loss += doc_loss
    print("Epoch loss: {}".format(epoch_loss))

     
