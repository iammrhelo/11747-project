import argparse
import math
import os
import pprint
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
    parser.add_argument('--early_stop',type=int,default=10)
    parser.add_argument('--shuffle',action="store_true",default=False)
    parser.add_argument('--debug',action="store_true",default=False)
    args = parser.parse_args()
    return args

args = parse_arguments()
print(args)

##################
#  Data Loading  #
##################
data = args.data
debug = args.debug
dict_pickle = os.path.join(data,'train','dict.pickle')

# Set Corpus
if debug:
    train_corpus = Corpus(os.path.join(data,'debug_train'), dict_pickle)
    valid_corpus = Corpus(os.path.join(data,'debug_valid'),dict_pickle)
    test_corpus = Corpus(os.path.join(data,'debug_test'),dict_pickle)
else:
    train_corpus = Corpus(os.path.join(data,'train'), dict_pickle)
    valid_corpus = Corpus(os.path.join(data,'valid'),dict_pickle)
    test_corpus = Corpus(os.path.join(data,'test'),dict_pickle)

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


def run_corpus(corpus, train_mode=False):
    print("train_mode",train_mode)

    if train_mode: 
        model.train()
    else:
        model.eval()
    
    corpus_loss = 0
    entity_count = 0
    entity_correct_count = 0

    for doc_idx, (doc_name, doc) in enumerate(corpus.documents,1):
        
        doc_loss = 0
        doc_entity_count = 0
        doc_entity_correct_count = 0

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
            if train_mode: optimizer.zero_grad()

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

                # Need to predict the next entity
                if curr_r.data[0] == 0 and next_r.data[0] == 1: 
                    next_entity_index = int(next_e.data[0])
                    assert next_entity_index == next_e.data[0]

                    # Concatenate entities to a block
                    pred_e = model.predict_entity(h_t, sent_idx, lambda_dist)

                    if next_entity_index < len(model.entities):
                        next_e = Variable(torch.LongTensor([next_entity_index]), requires_grad=False)
                    else:
                        next_e = Variable(torch.zeros(1).type(torch.LongTensor), requires_grad=False)

                    # TODO: FAILURE
                    pred_entity_index = pred_e.squeeze().max(0)[1].data[0]
                    next_entity_index = next_e.data[0]

                    entity_count += 1
                    entity_correct_count += pred_entity_index == next_entity_index

                # Update Entity
                if curr_r.data[0] > 0 and curr_e.data[0] > 0:
                    
                    # Next Entity Type
                    entity_idx = int(curr_e.data[0])
                    assert entity_idx == curr_e.data[0] and entity_idx <= len(model.entities)

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
                        pred_e = model.predict_entity(h_t, sent_idx, lambda_dist)

                        if next_entity_index < len(model.entities):
                            next_e = Variable(torch.LongTensor([next_entity_index]), requires_grad=False)
                        else:
                            next_e = Variable(torch.zeros(1).type(torch.LongTensor), requires_grad=False)

                        # TODO: OK
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

            sent_loss = sum(losses)
            doc_loss += sent_loss.data[0]

            if train_mode:
                sent_loss.backward(retain_graph=True)
                optimizer.step()

        # End of document
        # Clear Entities
        model.clear_entities()

        doc_entity_acc = entity_correct_count / entity_count

        progress = "{}/{}".format(doc_idx,len(corpus.documents))
        print("progress",progress,"doc_name",doc_name,"doc_loss",doc_loss,'doc_entity_acc',doc_entity_acc)

        corpus_loss += doc_loss
        entity_count += doc_entity_count
        entity_correct_count += doc_entity_correct_count

    corpus_entity_acc = entity_correct_count / entity_count
    return corpus_loss, corpus_entity_acc

####################
#   Main Program   #
####################

best_valid_loss = None
early_stop_count = 0
model_path = 'models/entitynlm_best.pt'
for epoch in range(1,num_epochs+1,1):
    print("Epoch",epoch)

    epoch_loss = 0
    
    # Shuffle Documents Here
    if shuffle: random.shuffle(train_corpus.documents)
    train_loss, train_entity_acc = run_corpus(train_corpus, train_mode=True)
    print("train_loss",train_loss,"train_entity_acc",train_entity_acc)
    
    valid_loss, valid_entity_acc = run_corpus(valid_corpus, train_mode=False)
    print("valid_loss",valid_loss,"valid_entity_acc",valid_entity_acc)

    # Early stopping conditioning on validation set loss
    if best_valid_loss == None or valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(),model_path)
        early_stop_count = 0
    else:
        early_stop_count += 1

print("Test set evaluation")
model.load_state_dict(torch.load(model_path))
test_loss, test_entity_acc = run_corpus(test_corpus, train_mode=False)
print("test_loss",test_loss,"test_entity_acc",test_entity_acc)


     