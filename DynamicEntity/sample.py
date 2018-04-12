import argparse
import math
import os
import pprint
import random
import sys
import time

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from corpus import Corpus
from model import EntityNLM
from util import timeit

use_cuda = torch.cuda.is_available()

device = 0 if use_cuda else -1

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=str,default='./data/InScript')
    parser.add_argument('--embed_dim',type=int,default=100)
    parser.add_argument('--hidden_size',type=int,default=128)
    parser.add_argument('--entity_size',type=int,default=128)
    parser.add_argument('--num_layers',type=int,default=2)
    parser.add_argument('--dropout',type=float,default=0.5)
    parser.add_argument('--num_epochs',type=int,default=40)
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--early_stop',type=int,default=3)
    parser.add_argument('--pretrained',action="store_true",default=False)
    parser.add_argument('--model_path',type=str,default=None)
    parser.add_argument('--exp',type=str,default="exp")
    parser.add_argument('--debug',action="store_true",default=False)
    parser.add_argument('--every_entity',action="store_true",default=True)
    parser.add_argument('--skip_sentence',type=int,default=3)
    parser.add_argument('--max_entity',type=int,default=30)
    parser.add_argument('--out_dir',type=str,default=None)
    args = parser.parse_args()
    return args

args = parse_arguments()
print(args)
print("use_cuda",use_cuda)

##################
#  Data Loading  #
##################
data = args.data
debug = args.debug
dict_pickle = os.path.join(data,'train','dict.pickle')


print("Loading corpus...")
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
print("vocab_size",vocab_size)

##################
#   Model Setup  #
##################
embed_dim = args.embed_dim
hidden_size = args.hidden_size
entity_size = args.entity_size
num_layers = args.num_layers
dropout = args.dropout
pretrained = args.pretrained
model_path = args.model_path

model = EntityNLM(vocab_size=vocab_size, 
                    embed_size=embed_dim, 
                    hidden_size=hidden_size,
                    entity_size=entity_size,
                    dropout=dropout,
                    use_cuda=use_cuda)

if model_path is not None:
    print("Loading from {}".format(model_path))
    model.load_state_dict(torch.load(model_path))
elif pretrained: 
    model.load_pretrained(train_corpus.dictionary)

if use_cuda:
    model = model.cuda()

lambda_dist = 1e-6

# Evaluation
every_entity = args.every_entity
skip_sentence = args.skip_sentence
max_entity = args.max_entity

def repack(h_t, c_t):
    return Variable(h_t.data), Variable(c_t.data)

@timeit
def run_sample(corpus, out_dir):
    print("sample")

    model.eval()
    
    
    import pdb; pdb.set_trace()

    for doc_idx, (doc_name, doc) in enumerate(corpus.documents,1):

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
            h_t, c_t = repack(h_t,c_t)
            
            X_tensor = Variable(torch.from_numpy(np.array(X[sent_idx])).type(torch.LongTensor))
            R_tensor = Variable(torch.from_numpy(np.array(R[sent_idx])).type(torch.LongTensor))
            E_tensor = Variable(torch.from_numpy(np.array(E[sent_idx])).type(torch.LongTensor))
            L_tensor = Variable(torch.from_numpy(np.array(L[sent_idx])).type(torch.LongTensor))

            if use_cuda:
                h_t = h_t.cuda()
                c_t = c_t.cuda()
                X_tensor = X_tensor.cuda()
                R_tensor = R_tensor.cuda()
                E_tensor = E_tensor.cuda()
                L_tensor = L_tensor.cuda()

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

                # Predict Every Next Entity
                if next_r.data[0] == 1:
                    next_entity_index = int(next_e.data[0])
                    assert next_entity_index == next_e.data[0]

                    # Concatenate entities to a block
                    pred_e = model.predict_entity(h_t, sent_idx, lambda_dist)

                    if next_entity_index < len(model.entities):
                        next_e = Variable(torch.LongTensor([next_entity_index]), requires_grad=False)
                    else:
                        next_e = Variable(torch.zeros(1).type(torch.LongTensor), requires_grad=False)

                    pred_entity_index = pred_e.squeeze().max(0)[1].data[0]
                    next_entity_index = next_e.data[0]

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
                        
                        if use_cuda:
                            next_e = next_e.cuda()


                    # Entity Length Prediction
                    if int(next_e.data[0]) > 0: # Has Entity

                        # User predicted entity's embedding
                        entity_idx = int(next_e.data[0])
                        entity_embedding = model.get_entity(entity_idx)

                        pred_l = model.predict_length(h_t, entity_embedding)

                # Word Prediction
                next_entity_index = int(next_e.data[0])
                assert next_entity_index == next_e.data[0]

                pred_x = model.predict_word(next_entity_index, h_t, last_entity)


            last_entity = h_t # Take hidden state as last entity embedding for next sentence

        # End of document
        # Clear Entities
        model.clear_entities()


####################
#   Main Program   #
####################

model_name = "embed_{}_hidden_{}_entity_{}_dropout_{}_pretrained_{}_every_{}_skip_{}_max_{}_best.pt"\
            .format(embed_dim,hidden_size,entity_size,dropout,pretrained,every_entity, skip_sentence, max_entity)
if debug: model_name = "debug_" + model_name

exp_dir = args.exp
if not os.path.exists(exp_dir): os.makedirs(exp_dir)

model_path = os.path.join(exp_dir, model_name) if model_path is None else model_path

print("Model will be saved to {}".format(model_path))


out_dir = args.out_dir
if not os.path.exists(out_dir): os.makedirs(out_dir)
print("Test set sampling")
model.load_state_dict(torch.load(model_path))
run_sample(test_corpus, out_dir=out_dir)
