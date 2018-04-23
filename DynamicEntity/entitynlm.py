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
from data_utils import LetsGoCorpus, LetsGoEntityDataLoader
from model import EntityNLM
from opts import build_model_name, parse_arguments
from vocab import Vocab, VocabEntry
from util import timeit

# CUDA
use_cuda = torch.cuda.is_available()
device = 0 if use_cuda else -1
print("use_cuda",use_cuda)

def load_corpus(args):
    dataset_name = args.dataset
    print("Loading dataset",dataset_name)

    if dataset_name == "debug":
        data_dir = './data/modi'
        dict_pickle = os.path.join(data_dir,'train','dict.pickle')
        train_corpus = Corpus(os.path.join(data_dir,'debug_train'), dict_pickle, use_cuda=use_cuda)
        valid_corpus = Corpus(os.path.join(data_dir,'debug_valid'), dict_pickle, use_cuda=use_cuda)
        test_corpus = Corpus(os.path.join(data_dir,'debug_test'), dict_pickle, use_cuda=use_cuda)
        dictionary = train_corpus.dictionary
    elif dataset_name == "inscript":
        data_dir = './data/modi'
        dict_pickle = os.path.join(data_dir,'train','dict.pickle')
        train_corpus = Corpus(os.path.join(data_dir,'train'), dict_pickle, use_cuda=use_cuda)
        valid_corpus = Corpus(os.path.join(data_dir,'valid'),dict_pickle, use_cuda=use_cuda)
        test_corpus = Corpus(os.path.join(data_dir,'test'),dict_pickle, use_cuda=use_cuda)
        dictionary = train_corpus.dictionary
    elif dataset_name == "letsgo":
        vocab = torch.load('./data/vocab.bin')
        corpus = LetsGoCorpus('./data/union_data-1ab.p')
        train_corpus = LetsGoEntityDataLoader(corpus.train, vocab.src, use_cuda=use_cuda)
        valid_corpus = LetsGoEntityDataLoader(corpus.valid, vocab.src, use_cuda=use_cuda)
        test_corpus = LetsGoEntityDataLoader(corpus.test, vocab.src, use_cuda=use_cuda)
        dictionary = train_corpus.vocab.word2id
    else:
        raise ValueError("Invalid dataset:",dataset_name)

    return train_corpus, valid_corpus, test_corpus, dictionary

def build_model(vocab_size, args, dictionary):
    model = EntityNLM(vocab_size=vocab_size, 
                        embed_size=args.embed_dim, 
                        hidden_size=args.hidden_size,
                        entity_size=args.entity_size,
                        dropout=args.dropout,
                        use_cuda=use_cuda)

    if use_cuda:
        model = model.cuda()
    
    if args.model_path is not None:
        print("Loading from {}".format(args.model_path))
        model.load_state_dict(torch.load(args.model_path))
    elif args.pretrained: 
        model.load_pretrained(dictionary)
    
    return model

def build_optimizer(args, model):
    if args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
    return optimizer

def repack(h_t, c_t):
    if use_cuda:
        return Variable(h_t.data).cuda(), Variable(c_t.data).cuda()
    else:
        return Variable(h_t.data), Variable(c_t.data)

def build_model_path(exp_dir, model_name, model_path):
    if not os.path.exists(exp_dir): 
        os.makedirs(exp_dir)
    if model_path is None:
        model_path = os.path.join(exp_dir, model_name + '.pt')
    return model_path

@timeit
def run_corpus(corpus, model, optimizer, criterion, config, train_mode=False):
    if train_mode: 
        model.train()
    else:
        model.eval()
    
    ignore_x = config['ignore_x']
    ignore_r = config['ignore_r']
    ignore_l = config['ignore_l']
    ignore_e = config['ignore_e']
    max_entity = config['max_entity']
    skip_sentence = config['skip_sentence']

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
        
        doc_predict_entity_count = 0

        X, R, E, L = doc[0]

        nsent = len(X)
        
        # For every document
        h_t, c_t = model.init_hidden_states(1)
        model.create_entity() # Dummy
        entity_current = model.entities[0]

        # Check
        assert len(model.entities) == 1 # Only 1 dummy entity

        for sent_idx in range(nsent):
            # Learn for every sentence
            losses = []
            if train_mode: 
                optimizer.zero_grad()

            X_tensor = Variable(X[sent_idx])
            R_tensor = Variable(R[sent_idx])
            E_tensor = Variable(E[sent_idx])
            L_tensor = Variable(L[sent_idx])

            h_t, c_t = repack(h_t,c_t)

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

                #######################
                #  Entity Prediction  #
                #######################
                test_condition = ( sent_idx >= skip_sentence ) and ( max_entity < 0 or doc_predict_entity_count < max_entity )

                if (train_mode or test_condition) and next_r.data[0] == 1:
                    next_entity_index = int(next_e.data[0])
                    assert next_entity_index == next_e.data[0]

                    # Concatenate entities to a block
                    pred_e = model.predict_entity(h_t, sent_idx)

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

                    doc_predict_entity_count += 1

                # Update Entity
                if curr_r.data[0] > 0 and curr_e.data[0] > 0:
                    
                    # Next Entity Type
                    entity_idx = int(curr_e.data[0])
                    assert entity_idx == curr_e.data[0] and entity_idx <= len(model.entities)

                    # Create if it's a new entity
                    if entity_idx == len(model.entities):
                        model.create_entity(nsent=sent_idx)
                    
                    # Update Entity Here
                    entity_current = model.update_entity(entity_idx, h_t, sent_idx)
    
                # l == 1, End of Mention
                if curr_l.data[0] == 1:

                    mention_length = int(curr_l.data[0])
                    assert mention_length == curr_l.data[0], "{} : {}".format(mention_length, curr_l.data[0])

                    pred_r = model.predict_type(h_t)
                    # TODO: OK
                    if not ignore_r:
                        type_loss = criterion(pred_r,next_r)
                        doc_r_loss += type_loss.data[0]
                        losses.append(type_loss)
                            
                    # Entity Prediction
                    if next_r.data[0] > 0: # If the next word is an entity
                        
                        next_entity_index = int(next_e.data[0])
                        assert next_entity_index == next_e.data[0]

                        # Concatenate entities to a block
                        pred_e = model.predict_entity(h_t, sent_idx)

                        if next_entity_index < len(model.entities):
                            next_e = Variable(torch.LongTensor([next_entity_index]), requires_grad=False)
                        else:
                            next_e = Variable(torch.zeros(1).type(torch.LongTensor), requires_grad=False)
                        
                        if use_cuda:
                            next_e = next_e.cuda()

                        # TODO: OK
                        if not ignore_e:
                            e_loss = criterion(pred_e, next_e)
                            doc_e_loss += e_loss.data[0]
                            losses.append(e_loss)

                    # Entity Length Prediction
                    if int(next_e.data[0]) > 0: # Has Entity

                        # User predicted entity's embedding
                        entity_idx = int(next_e.data[0])
                        entity_embedding = model.get_entity(entity_idx)

                        pred_l = model.predict_length(h_t, entity_embedding)
                        
                        if not ignore_l: 
                            l_loss = criterion(pred_l, next_l)
                            doc_l_loss += l_loss.data[0]
                            losses.append(l_loss)

                # Word Prediction
                next_entity_index = int(next_e.data[0])
                assert next_entity_index == next_e.data[0]

                pred_x = model.predict_word(next_entity_index, h_t, entity_current)

                # TODO: OK
                if not ignore_x:
                    x_loss = criterion(pred_x, next_x)
                    doc_x_loss += x_loss.data[0]
                    losses.append(x_loss)
  
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
        print(progress_msg,end='\r')

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

    corpus_losses = {
        'loss': corpus_loss,
        'x_loss': corpus_x_loss,
        'r_loss': corpus_r_loss,
        'e_loss': corpus_e_loss,
        'l_loss': corpus_l_loss,
    }

    corpus_entity_acc = entity_correct_count / entity_count
    corpus_prev_entity_acc = prev_entity_correct_count / entity_count
    corpus_new_entity_acc = new_entity_correct_count / entity_count

    corpus_accuracies = {
        'entity_acc': corpus_entity_acc,
        'prev_entity_acc': corpus_prev_entity_acc,
        'new_entity_acc': corpus_new_entity_acc,
    }

    return corpus_losses, corpus_accuracies

def record_to_writer(writer, epoch, losses, accuracies):

    x_loss = losses['x_loss']
    r_loss = losses['r_loss']
    l_loss = losses['l_loss']
    e_loss = losses['e_loss']
    loss = losses['loss']

    writer.add_scalar('loss/x', x_loss, epoch)
    writer.add_scalar('loss/r', r_loss, epoch)
    writer.add_scalar('loss/l', l_loss, epoch)
    writer.add_scalar('loss/e', e_loss, epoch)
    writer.add_scalar('loss/total', loss, epoch)

    entity_acc = accuracies['entity_acc']
    prev_entity_acc = accuracies['prev_entity_acc']
    new_entity_acc = accuracies['new_entity_acc']

    writer.add_scalar('accuracy/entity', entity_acc, epoch)
    writer.add_scalar('accuracy/prev_entity', prev_entity_acc, epoch)
    writer.add_scalar('accuracy/new_entity', new_entity_acc, epoch)

####################
#   Main Program   #
####################

def main():
    args = parse_arguments()
    print(args)

    ##################
    #  Data Loading  #
    ##################
    train_corpus, valid_corpus, test_corpus, dictionary = load_corpus(args)
    vocab_size = len(dictionary)
    print("vocab_size",vocab_size)

    ##################
    #   Model Setup  #
    ##################
    model = build_model(vocab_size, args, dictionary)

    criterion = nn.CrossEntropyLoss()
    
    if use_cuda: 
        criterion = criterion.cuda()

    optimizer = build_optimizer(args, model)

    #####################
    #  Training Config  #
    #####################
    num_epochs = args.num_epochs

    config = {
        'ignore_x': args.ignore_x,
        'ignore_r': args.ignore_r,
        'ignore_l': args.ignore_l,
        'ignore_e': args.ignore_e,
        'skip_sentence': args.skip_sentence,
        'max_entity': args.max_entity
    }

    best_valid_loss = None
    early_stop_count = 0
    early_stop_threshold = args.early_stop

    model_name = build_model_name(args)
    model_path = build_model_path(args.exp, model_name, args.model_path)

    tensorboard_dir = args.tensorboard

    print("Model will be saved to {}".format(model_path))

    train_writer = SummaryWriter('{}/{}/{}'.format(tensorboard_dir,model_name,'train'))
    valid_writer = SummaryWriter('{}/{}/{}'.format(tensorboard_dir,model_name,'valid'))
    test_writer = SummaryWriter('{}/{}/{}'.format(tensorboard_dir,model_name,'test'))

    for epoch in range(1,num_epochs+1,1):
        print("Epoch",epoch)
        
        # Run training
        random.shuffle(train_corpus.documents)
        train_losses, train_accuracies = run_corpus(train_corpus, model, optimizer, criterion, config, train_mode=True)
        train_loss, train_entity_acc = train_losses['loss'], train_accuracies['entity_acc']
        print("train_loss",train_loss,"train_entity_acc",train_entity_acc)
        record_to_writer(train_writer, epoch, train_losses, train_accuracies)
        
        # Run validation
        valid_losses, valid_accuracies = run_corpus(valid_corpus, model, optimizer, criterion, config, train_mode=False)
        valid_loss, valid_entity_acc = valid_losses['loss'], valid_accuracies['entity_acc']
        print("valid_loss",valid_loss,"valid_entity_acc",valid_entity_acc)
        record_to_writer(valid_writer, epoch, valid_losses, valid_accuracies)
        
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

    test_losses, test_accuracies = run_corpus(test_corpus, model, optimizer, criterion, config, train_mode=False)
    test_loss, test_entity_acc = test_losses['loss'], test_accuracies['entity_acc']
    print("test_loss",test_loss,"test_entity_acc",test_entity_acc)
    record_to_writer(test_writer, epoch, test_losses, test_accuracies)

    train_writer.close()
    valid_writer.close()
    test_writer.close()

if __name__ == "__main__":
    main()
