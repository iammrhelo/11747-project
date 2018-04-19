import os
import math
import argparse
import torch
import torch.nn as nn
from torch import optim
import torch.nn.utils
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn import functional as F
from models import CNNEncoder, Encoder, Decoder, Seq2Seq
from data_utils import *
from config import init_config
import sys
from vocab import *
import time



def word2id(sents, vocab):
    if type(sents[0]) == list:
        return [[vocab[w] for w in s] for s in sents]
    else:
        return [vocab[w] for w in sents]

def input_transpose(sents, pad_token):
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    sents_t = []
    masks = []
    for i in range(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in range(batch_size)])
        masks.append([1 if len(sents[k]) > i else 0 for k in range(batch_size)])

    return sents_t, masks


def to_input_variable(sents, vocab, cuda=False, is_test=False):
    """
    return a tensor of shape (src_sent_len, batch_size)
    """

    word_ids = word2id(sents, vocab)
    sents_t, masks = input_transpose(word_ids, vocab['<pad>'])

    sents_var = Variable(torch.LongTensor(sents_t), volatile=is_test, requires_grad=False)
    if cuda:
        sents_var = sents_var.cuda()

    return sents_var


def input_transpose_src(sents, pad_token, max_len):
    batch_size = len(sents)

    sents_t = []
    masks = []
    for i in range(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in range(batch_size)])
        masks.append([1 if len(sents[k]) > i else 0 for k in range(batch_size)])

    return sents_t, masks

def to_input_variable_src(src_data, vocab, cuda=False, is_test=False):
    """
    return a tensor of shape (src_sent_len, batch_size)
    """
    # sys_sents = [turn[0] for turn in context for context in src_data]
    # usr_sents = [turn[1] for turn in context for context in src_data]
    # scores = [turn[2] for turn in context for context in src_data]


    # word_ids = word2id(sys_sents, vocab)
    # sys_sents_t, masks = input_transpose(word_ids, vocab['<pad>'])

    # sys_sents_var = Variable(torch.LongTensor(sys_sents_t), volatile=is_test, requires_grad=False)
    # if cuda:
    #     sys_sents_var = sys_sents_var.cuda()

    # word_ids = word2id(usr_sents, vocab)
    # usr_sents_t, masks = input_transpose(word_ids, vocab['<pad>'])

    # usr_sents_var = Variable(torch.LongTensor(usr_sents_t), volatile=is_test, requires_grad=False)
    # if cuda:
    #     usr_sents_var = usr_sents_var.cuda()

    # score_var = Variable(torch.FloatTensor(scores), volatile=is_test, requires_grad=False)

    # return sys_sents_var, usr_sents_var, score_var

    """
    return a tensor of shape (src_sent_len, batch_size)
    """
    ret = []
    max_len = 30
    for each in src_data:
        word_ids = word2id(each, vocab)
        sents_t, masks = input_transpose_src(word_ids, vocab['<pad>'], max_len)
        ret.append(sents_t)
    sents_var = Variable(torch.LongTensor(ret), volatile=is_test, requires_grad=False)
    if cuda:
        sents_var = sents_var.cuda()
        #ret.append(sents_var)
    return sents_var


def to_input_variable_conf(src_data, cuda=False, is_test=False):
    ret = Variable(torch.FloatTensor(src_data), volatile=is_test, requires_grad=False)
    if cuda:
        ret = ret.cuda()
    return ret 

def init_model(args):
    vocab = torch.load(args.vocab)

    cnn_encoder = CNNEncoder(len(vocab.src), args.embed_size)
    encoder = Encoder(cnn_encoder.out_size, args.hidden_size)
    devoder = Decoder(args.embed_size, args.hidden_size, len(vocab.tgt))

    model = Seq2Seq(cnn_encoder, encoder, devoder, args, vocab)
    model.load_state_dict(torch.load(args.load_model_path))
    model.eval()

    return vocab, model

def generate():

    args = init_config()
    vocab = torch.load('./data/vocab.bin')
    corpus = LetsGoCorpus('./data/union_data-1ab.p')
    # train_loader = FakeLetsGoDataLoader(corpus.train)
    # dev_loader = FakeLetsGoDataLoader(corpus.valid)
    # test_loader = FakeLetsGoDataLoader(corpus.test)

    #train_loader = LetsGoDataLoader(corpus.train)
    #dev_loader = LetsGoDataLoader(corpus.valid)
    test_loader = LetsGoDataLoader(corpus.test)


    #train_data = list(zip(train_loader.get_src(), train_loader.get_tgt()))
    #dev_data = list(zip(dev_loader.get_src(), dev_loader.get_tgt()))
    test_data = list(zip(test_loader.get_src(), test_loader.get_tgt()))


    vocab, model = init_model(args)


    for sent, tgt in data_iter_test(test_data):

        sys_utt = [[turn[0] for turn in dial] for dial in sent]
        usr_utt = [[turn[1] for turn in dial] for dial in sent]
        conf = [[turn[2] for turn in dial] for dial in sent]

        src_sents_sys_vars = to_input_variable_src(sys_utt, vocab.src, cuda=False)
        src_sents_usr_vars = to_input_variable_src(usr_utt, vocab.src, cuda=False)
        src_sents_conf_vars = to_input_variable_conf(conf, cuda=False)

        #print(src_sents_sys_vars.size())
        #print(src_sents_usr_vars.size())
        #exit(0)
        src_sent_len = [len(s) for s in sent]


        sampled_ids_all, scores_, attn_ = model.greedy(src_sents_sys_vars, src_sents_usr_vars, src_sents_conf_vars, src_sent_len)

        sentences = []
        for sampled_ids in sampled_ids_all[0]: # just a hack, todo
            # Decode word_ids to words
            sampled_words = []
            for word_id in sampled_ids:
                word = vocab.tgt.id2word[word_id]
                sampled_words.append(word)
                if word == '</s>':
                    break
            sentence = ' '.join(sampled_words[:-1])
            sentences.append(sentence)

            # Print generated sequence
            print(sentence)


def print_data():

    args = init_config()
    vocab = torch.load('./data/vocab.bin')
    corpus = LetsGoCorpus('./data/union_data-1ab.p')
    train_loader = LetsGoDataLoader(corpus.train)
    dev_loader = LetsGoDataLoader(corpus.valid)
    test_loader = LetsGoDataLoader(corpus.test)

    train_data = list(zip(train_loader.get_src(), train_loader.get_tgt()))
    dev_data = list(zip(dev_loader.get_src(), dev_loader.get_tgt()))
    test_data = list(zip(test_loader.get_src(), test_loader.get_tgt()))

    for each in test_data:
        print(' '.join(each[1][1:-1]))
    


def main():
    #print_data()
    generate()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
