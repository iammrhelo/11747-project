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




def evaluate(model, val_iter, vocab_size, DE, EN):
    model.eval()
    pad = EN.vocab.stoi['<pad>']
    total_loss = 0
    for b, batch in enumerate(val_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        src = Variable(src.data.cuda(), volatile=True)
        trg = Variable(trg.data.cuda(), volatile=True)
        output = model(src, trg)
        loss = F.cross_entropy(output[1:].view(-1, vocab_size),
                               trg[1:].contiguous().view(-1),
                               ignore_index=pad)
        total_loss += loss.data[0]
    return total_loss / len(val_iter)

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
    max_len = max([max([len(s) for s in sents]) for sents in src_data])
    for each in src_data:
        word_ids = word2id(each, vocab)
        sents_t, masks = input_transpose_src(word_ids, vocab['<pad>'], max_len)
        ret.append(sents_t)
    sents_var = Variable(torch.LongTensor(ret), volatile=is_test, requires_grad=False)
    if cuda:
        sents_var = sents_var.cuda()
        #ret.append(sents_var)
    return sents_var


def init_training(args):
    vocab = torch.load(args.vocab)

    cnn_encoder = CNNEncoder(len(vocab.src), args.embed_size)
    encoder = Encoder(cnn_encoder.out_size, args.hidden_size)
    devoder = Decoder(args.embed_size, args.hidden_size, len(vocab.tgt))

    model = Seq2Seq(cnn_encoder, encoder, devoder, args, vocab)
    model.train()


    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt['<pad>']] = 0

    nll_loss = nn.NLLLoss(weight=vocab_mask, size_average=False)
    cross_entropy_loss = nn.CrossEntropyLoss(weight=vocab_mask, size_average=False)

    if args.cuda:
        model = model.cuda()
        nll_loss = nll_loss.cuda()
        cross_entropy_loss = cross_entropy_loss.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    return vocab, model, optimizer, nll_loss, cross_entropy_loss

def train():

    args = init_config()
    vocab = torch.load('./data/vocab.bin')
    corpus = LetsGoCorpus('./data/union_data-1ab.p')
    train_loader = FakeLetsGoDataLoader(corpus.train)
    dev_loader = FakeLetsGoDataLoader(corpus.valid)
    test_loader = FakeLetsGoDataLoader(corpus.test)

    train_data = zip(train_loader.get_src(), train_loader.get_tgt())
    dev_data = zip(dev_loader.get_src(), dev_loader.get_tgt())
    test_data = zip(test_loader.get_src(), test_loader.get_tgt())


    vocab, model, optimizer, nll_loss, cross_entropy_loss = init_training(args)

    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = cum_batches = report_examples = epoch = valid_num = best_model_iter = 0
    # hist_valid_scores = []
    # train_time = begin_time = time.time()

    # print('begin Maximum Likelihood training')

    while True:
        epoch += 1
        for src_sents, tgt_sents in data_iter(train_data, batch_size=args.batch_size):
            train_iter += 1

            src_sents_vars = to_input_variable_src(src_sents, vocab.src, cuda=args.cuda)
            tgt_sents_var = to_input_variable(tgt_sents, vocab.tgt, cuda=args.cuda)

            # print(src_sents_vars)
            # print(tgt_sents_var)
            # sys.exit(0)


            batch_size = len(src_sents)
            src_sents_len = [len(s) for s in src_sents]
            pred_tgt_word_num = sum(len(s[1:]) for s in tgt_sents) # omitting leading `<s>`

            optimizer.zero_grad()

            # (tgt_sent_len, batch_size, tgt_vocab_size)
            scores = model(src_sents_vars, src_sents_len, tgt_sents_var[:-1])


            word_loss = cross_entropy_loss(scores.view(-1, scores.size(2)), tgt_sents_var[1:].view(-1))
            loss = word_loss / batch_size
            word_loss_val = word_loss.data[0]
            loss_val = loss.data[0]


            loss.backward()
            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)
            optimizer.step()

            report_loss += word_loss_val
            cum_loss += word_loss_val
            report_tgt_words += pred_tgt_word_num
            cum_tgt_words += pred_tgt_word_num
            report_examples += batch_size
            cum_examples += batch_size
            cum_batches += batch_size




            if train_iter % args.log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         np.exp(report_loss / report_tgt_words),
                                                                                         cum_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

           


def main():
    train()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
