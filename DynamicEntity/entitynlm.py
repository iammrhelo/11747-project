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
    parser.add_argument('--embed_dim',type=int,default=128)
    parser.add_argument('--hidden_size',type=int,default=128)
    parser.add_argument('--entity_size',type=int,default=128)
    parser.add_argument('--num_layers',type=int,default=2)
    parser.add_argument('--dropout',type=float,default=0.5)
    parser.add_argument('--num_epochs',type=int,default=40)
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--early_stop',type=int,default=5)
    parser.add_argument('--shuffle',action="store_true",default=True)
    parser.add_argument('--pretrained',action="store_true",default=False)
    parser.add_argument('--model_path',type=str,default=None)
    parser.add_argument('--exp',type=str,default="exp")
    parser.add_argument('--tensorboard',type=str,default="runs")
    parser.add_argument('--debug',action="store_true",default=False)
    parser.add_argument('--every_entity',action="store_true",default=False)
    parser.add_argument('--skip_sentence',type=int,default=3)
    parser.add_argument('--max_entity',type=int,default=-1)
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

#####################
#  Training Config  #
#####################
num_epochs = args.num_epochs
lr = args.lr
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

crossentropy = nn.CrossEntropyLoss()
if use_cuda: crossentropy = crossentropy.cuda()

shuffle = args.shuffle

# Evaluation
every_entity = args.every_entity
skip_sentence = args.skip_sentence
max_entity = args.max_entity

def repack(h_t, c_t):
    return Variable(h_t.data), Variable(c_t.data)

@timeit
def run_corpus(corpus, epoch, train_mode=False, writer=None):
    print("train_mode",train_mode)

    if train_mode: 
        model.train()
    else:
        model.eval()
    
    corpus_loss = 0
    corpus_x_loss = 0
    corpus_r_loss = 0
    corpus_l_loss = 0
    corpus_e_loss = 0
    
    entity_count = 0
    entity_correct_count = 0
    
    prev_entity_count = 0
    prev_entity_correct_count = 0

    new_entity_count = 0
    new_entity_correct_count = 0

    predict_entity_count = 0

    for doc_idx, (doc_name, doc) in enumerate(corpus.documents,1):
        
        doc_loss = 0
        doc_x_loss = 0
        doc_r_loss = 0
        doc_e_loss = 0
        doc_l_loss = 0

        doc_entity_count = 0
        doc_entity_correct_count = 0

        doc_prev_entity_count = 0
        doc_prev_entity_correct_count = 0

        doc_new_entity_count = 0
        doc_new_entity_correct_count = 0

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

                # Need to predict the next entity
                test_condition = ( sent_idx >= skip_sentence ) and ( max_entity < 0 or predict_entity_count < max_entity )

                if (train_mode or test_condition) and ( every_entity or curr_r.data[0] == 0 ) and next_r.data[0] == 1:
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

                    doc_entity_count += 1
                    doc_entity_correct_count += pred_entity_index == next_entity_index

                    if next_entity_index == 0:
                        doc_new_entity_correct_count += pred_entity_index == next_entity_index
                        doc_new_entity_count += 1
                    else:
                        doc_prev_entity_correct_count += pred_entity_index == next_entity_index
                        doc_prev_entity_count += 1

                    predict_entity_count += 1

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
                    type_loss = crossentropy(pred_r,next_r)
                    doc_r_loss += type_loss.data[0]
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
                        
                        if use_cuda:
                            next_e = next_e.cuda()

                        # TODO: OK
                        e_loss = crossentropy(pred_e, next_e)
                        doc_e_loss += e_loss.data[0]
                        losses.append(e_loss)

                    # Entity Length Prediction
                    if int(next_e.data[0]) > 0: # Has Entity

                        # User predicted entity's embedding
                        entity_idx = int(next_e.data[0])
                        entity_embedding = model.get_entity(entity_idx)

                        pred_l = model.predict_length(h_t, entity_embedding)
                        
                        l_loss = crossentropy(pred_l, next_l)
                        doc_l_loss += l_loss.data[0]
                        losses.append(l_loss)

                # Word Prediction
                next_entity_index = int(next_e.data[0])
                assert next_entity_index == next_e.data[0]

                pred_x = model.predict_word(next_entity_index, h_t, last_entity)

                # TODO: OK
                x_loss = crossentropy(pred_x, next_x)
                doc_x_loss += x_loss.data[0]
                losses.append(x_loss)

            last_entity = h_t # Take hidden state as last entity embedding for next sentence
            if len(losses):
                sent_loss = sum(losses)
                doc_loss += sent_loss.data[0]

                if train_mode:
                    sent_loss.backward(retain_graph=True)
                    optimizer.step()

        # End of document
        # Clear Entities
        model.clear_entities()

        doc_entity_acc = doc_entity_correct_count / doc_entity_count if doc_entity_count > 0 else 0

        progress = "{}/{}".format(doc_idx,len(corpus.documents))
        progress_msg = "progress {}, doc_name {}, doc_loss {}, doc_entity_acc {}/{}={:.2f}"\
                        .format(progress, doc_name, doc_loss, doc_entity_correct_count, doc_entity_count, doc_entity_acc)
        #print(progress_msg,end='\r')
        print(progress_msg)

        corpus_x_loss += doc_x_loss
        corpus_r_loss += doc_r_loss
        corpus_e_loss += doc_e_loss
        corpus_l_loss += doc_l_loss
        
        corpus_loss += doc_loss
     
        entity_count += doc_entity_count
        entity_correct_count += doc_entity_correct_count

        prev_entity_count += doc_prev_entity_count
        prev_entity_correct_count += doc_prev_entity_correct_count

        new_entity_count += doc_new_entity_count
        new_entity_correct_count += doc_new_entity_correct_count
    
    # Write to tensorboard
    corpus_loss /= len(corpus.documents)
    corpus_x_loss /= len(corpus.documents)
    corpus_r_loss /= len(corpus.documents)
    corpus_e_loss /= len(corpus.documents)
    corpus_l_loss /= len(corpus.documents)

    writer.add_scalar('loss/x',corpus_x_loss, epoch)
    writer.add_scalar('loss/r',corpus_r_loss, epoch)
    writer.add_scalar('loss/l',corpus_l_loss, epoch)
    writer.add_scalar('loss/e',corpus_e_loss, epoch)
    writer.add_scalar('loss/total',corpus_loss, epoch)

    corpus_entity_acc = entity_correct_count / entity_count
    corpus_prev_entity_acc = prev_entity_correct_count / prev_entity_count
    corpus_new_entity_acc = new_entity_correct_count / new_entity_count

    writer.add_scalar('accuracy/entity',corpus_entity_acc, epoch)
    writer.add_scalar('accuracy/prev_entity',corpus_prev_entity_acc, epoch)
    writer.add_scalar('accuracy/new_entity',corpus_new_entity_acc, epoch)
    
    return corpus_loss, corpus_entity_acc

####################
#   Main Program   #
####################

best_valid_loss = None
early_stop_count = 0
early_stop_threshold = args.early_stop

model_name = "embed_{}_hidden_{}_entity_{}_dropout_{}_pretrained_{}_best.pt"\
            .format(embed_dim,hidden_size,entity_size,dropout,pretrained)
if debug: model_name = "debug_" + model_name

exp_dir = args.exp
if not os.path.exists(exp_dir): os.makedirs(exp_dir)
model_path = os.path.join(exp_dir, model_name) if model_path is None else model_path

tensorboard_dir = args.tensorboard

print("Model will be saved to {}".format(model_path))

train_writer = SummaryWriter('{}/{}/{}'.format(tensorboard_dir,model_name,'train'))
valid_writer = SummaryWriter('{}/{}/{}'.format(tensorboard_dir,model_name,'valid'))
test_writer = SummaryWriter('{}/{}/{}'.format(tensorboard_dir,model_name,'test'))

for epoch in range(1,num_epochs+1,1):
    print("Epoch",epoch)

    epoch_loss = 0
    
    # Shuffle Documents Here
    if shuffle: random.shuffle(train_corpus.documents)
    train_loss, train_entity_acc = run_corpus(train_corpus, epoch, train_mode=True, writer=train_writer)
    print("train_loss",train_loss,"train_entity_acc",train_entity_acc)
   	
    if len(valid_corpus.documents) == 0: continue 
    valid_loss, valid_entity_acc = run_corpus(valid_corpus, epoch, train_mode=False, writer=valid_writer)
    print("valid_loss",valid_loss,"valid_entity_acc",valid_entity_acc)

    # Early stopping conditioning on validation set loss
    if best_valid_loss == None or valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(),model_path)
        early_stop_count = 0
    else:
        early_stop_count += 1

    if early_stop_count >= early_stop_threshold:
        print("Early stopping criteria met!")
        break

print("Test set evaluation")
model.load_state_dict(torch.load(model_path))
test_loss, test_entity_acc = run_corpus(test_corpus, num_epochs, train_mode=False, writer=test_writer)
print("test_loss",test_loss,"test_entity_acc",test_entity_acc)

train_writer.close()
valid_writer.close()
test_writer.close()
