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
from data_utils import LetsGoCorpus, LetsGoDataLoader
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
        train_corpus = LetsGoDataLoader(corpus.train, vocab.src)
        valid_corpus = LetsGoDataLoader(corpus.valid, vocab.src)
        test_corpus = LetsGoDataLoader(corpus.test, vocab.src)
        test_corpus.build_documents()
        dictionary = train_corpus.vocab.word2id
    else:
        raise ValueError("Invalid dataset:",dataset_name)

    return train_corpus, valid_corpus, test_corpus, dictionary

def build_model(vocab_size, args, dictionary):
    model = EntityNLM(vocab_size=vocab_size, 
                        embed_size=args.embed_dim, 
                        hidden_size=args.hidden_size,
                        entity_size=args.entity_size,
                        dropout=args.dropout)

    if use_cuda:
        model = model.cuda()
    
    #assert args.model_path is not None, "Specify a model file!"

    if args.model_path is not None:
        print("Loading from {}".format(args.model_path))
        model.load_state_dict(torch.load(args.model_path))
    elif args.pretrained: 
        model.load_pretrained(dictionary)
    
    return model

def repack(h_t, c_t):
    if use_cuda:
        return Variable(h_t.data).cuda(), Variable(c_t.data).cuda()
    else:
        return Variable(h_t.data), Variable(c_t.data)

@timeit
def run_generate(corpus, model, max_output_length=30):
    model.eval()

    results = []

    for doc_idx, (doc_name, doc) in enumerate(corpus.documents,1):

        context = []

        X, R, E, L = doc[0]

        nsent = len(X)
        
        # For every document
        h_t, c_t = model.init_hidden_states(1)
        model.create_entity() # Dummy
        entity_current = model.entities[0]

        # Check
        assert len(model.entities) == 1 # Only 1 dummy entity

        for sent_idx in range(nsent):

            X_tensor = Variable(X[sent_idx])
            R_tensor = Variable(R[sent_idx])
            E_tensor = Variable(E[sent_idx])
            L_tensor = Variable(L[sent_idx])

            h_t, c_t = repack(h_t,c_t)

            # Run Encoder
            sentence = []
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

                embed_curr_x = embed_curr_x.unsqueeze(0)
                h_t, (_, c_t) = model.rnn(embed_curr_x, (h_t, c_t))

                # We only need to update entity in this case
                
                h_t = h_t.squeeze(0)
                c_t = c_t.squeeze(0)

                context.append(h_t)

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

                    # Entity Length Prediction
                    if int(next_e.data[0]) > 0: # Has Entity

                        # User predicted entity's embedding
                        entity_idx = int(next_e.data[0])
                        entity_embedding = model.get_entity(entity_idx)

                        pred_l = model.predict_length(h_t, entity_embedding)

                # Word Prediction
                next_entity_index = int(next_e.data[0])
                assert next_entity_index == next_e.data[0]

                pred_x = model.predict_word(next_entity_index, h_t, entity_current)

        print(model.entities)
        print()    
        ##################
        #   Generation   #
        ##################

        # Initialize variables
        _START_IDX = corpus.vocab['<s>']
        curr_x = Variable(torch.LongTensor([_START_IDX]))
        curr_r = Variable(torch.LongTensor([0]))
        curr_l = Variable(torch.LongTensor([0]))
        curr_e = Variable(torch.LongTensor([0]))

        if use_cuda:
            curr_x = curr_x.cuda()
            curr_r = curr_r.cuda()
            curr_l = curr_l.cuda()
            curr_e = curr_e.cuda()

        for _ in range(max_output_length):

            # Forward and Get Hidden State
            embed_curr_x = model.embed(curr_x)
            h_t, c_t = model.rnn(embed_curr_x, (h_t, c_t))

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

                            # Forward and Get Hidden State
            embed_curr_x = model.embed(curr_x)
            h_t, c_t = model.rnn(embed_curr_x, (h_t, c_t))

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
                        
                _, next_r_index = pred_r.data.max(1)
                next_r = Variable(next_r_index)

                # Entity Prediction
                if next_r.data[0] > 0: # If the next word is an entity
                    pred_e = model.predict_entity(h_t, sent_idx)
                    _, next_e_index = pred_e.data.max(1)
                    next_e = Variable(next_e_index)

                # Entity Length Prediction
                if int(next_e.data[0]) > 0: # Has Entity

                    # User predicted entity's embedding
                    entity_idx = int(next_e.data[0])
                    entity_embedding = model.get_entity(entity_idx)

                    pred_l = model.predict_length(h_t, entity_embedding)
                    _, next_l_index = pred_l.data.max(1)
                    next_l = Variable(next_l_index)
            else:
                next_l = Variable(curr_l.data - 1)
                next_r = Variable(curr_r.data)
                next_e = Variable(curr_e.data)

            # Word Prediction
            next_entity_index = int(next_e.data[0])
            assert next_entity_index == next_e.data[0]

            pred_x = model.predict_word(next_entity_index, h_t, entity_current)
            _, next_x_index = pred_x.data.max(1)
            next_x = Variable(next_x_index)

            word_id = next_x.data[0]
            if word_id == corpus.vocab['</s>']:
                break
            
            # Append Word Here
            sentence.append(word_id)

            curr_x = next_x
            curr_r = next_r
            curr_l = next_l
            curr_e = next_e
        
        # End of document
        # Clear Entities
        sentence = id2word(sentence, corpus.vocab)

        results.append(sentence)
        model.clear_entities()
        
        progress = "{}/{}".format(doc_idx,len(corpus.documents))
        progress_msg = "progress {}, doc_name {}".format(progress, doc_name)
        print(progress_msg,end='\r')
        
    return results

def id2word(id_list, vocab):
    sentence = []
    for word_id in id_list:
        word = vocab[word_id]
        sentence.append(word)
    return sentence

def write_to_file(sentences, filepath):
    with open(filepath, 'w') as fout:
        for tokens in sentences:
            line = ' '.join(tokens) + '\n'
            fout.write(line)
            

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


    test_corpus.build_documents()
    #valid_sentences = run_generate(valid_corpus, model)
    test_sentences = run_generate(test_corpus, model)

    write_to_file(test_sentences, 'results.txt')
    import pdb; pdb.set_trace()
    

if __name__ == "__main__":
    main()
