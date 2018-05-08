import argparse
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from data_utils import load_corpus
from models.entitynlm import build_model
from opts import build_model_name, build_model_path, parse_arguments
from vocab import Vocab, VocabEntry # For pickling
import pickle

# CUDA
use_cuda = torch.cuda.is_available()

print("use_cuda",use_cuda)
def repack(h_t, c_t):
    if use_cuda:
        return Variable(h_t.data).cuda(), Variable(c_t.data).cuda()
    else:
        return Variable(h_t.data), Variable(c_t.data)

def run_corpus(corpus, model, split = 'train', train_mode=False):
    print(split)
    print(len(corpus.src))

    counter = 0

    results = []

    model.eval()
    
    for doc_idx, doc in enumerate(corpus.src, 1):

        
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
            # Learn for every sentence

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
                
                # reshape
                embed_curr_x = embed_curr_x.unsqueeze(0)
                h_t, (_, c_t) = model.rnn(embed_curr_x, (h_t, c_t))
     
                #######################
                #  Entity Prediction  #
                #######################

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


                h_t = h_t.unsqueeze(0)
                c_t = c_t.unsqueeze(0)   

        # grab things to cpu
        context = [i.data.cpu().numpy() for i in context]
        entities = [i.data.cpu().numpy() for i in model.entities]
        h = h_t.squeeze(0).data.cpu().numpy()
        c = c_t.squeeze(0).data.cpu().numpy()
        target = corpus.tgt[doc_idx-1]
        # grab things to cpu
        results.append((context, entities, (h, c), target))

        print(len(results))

        if len(results)%1000 == 0:
            counter += 1
            pickle.dump( results, open( split + "_extracted_data_" + str(counter) + ".p", "wb" ) )
            results = []
        # End of document
        # Clear Entities
        model.clear_entities()
            
        
    pickle.dump( results, open( split + "_extracted_data_last.p", "wb" ) )

    return model, model


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

    # print(len(train_corpus.src))
    # print(len(train_corpus.tgt))
    # print(train_corpus.src[2])
    # print(train_corpus.tgt[2])

    # for doc_idx, doc in enumerate(train_corpus.src,1):
    #     print(doc_idx)
    #     print(doc)
    #     break
    # exit()


    vocab_size = len(dictionary)
    print("vocab_size",vocab_size)

    # For target vocab
    vocab = torch.load('./data/vocab.bin')
    vocab_tgt = vocab.tgt
    vocab_tgt_size = len(vocab_tgt)
    

    ##################
    #   Model Setup  #
    ##################
    model = build_model(vocab_size, args, dictionary)
    # decoder = Decoder(args.embed_dim, args.hidden_size, out_vocab_size)
    # if use_cuda:
    #     decoder.cuda()

    #train_losses, train_accuracies = run_corpus(train_corpus, model, 'train' , train_mode=False)
    train_losses, train_accuracies = run_corpus(test_corpus, model, 'test', train_mode=False)
    #train_losses, train_accuracies = run_corpus(valid_corpus, model, 'valid', train_mode=False)
    

    

    # print("Test set evaluation")
    # model.load_state_dict(torch.load(model_path))

    # test_losses, test_accuracies = run_corpus(test_corpus, model, decoder, optimizer, criterion, config, train_mode=False)


if __name__ == "__main__":
    main()
