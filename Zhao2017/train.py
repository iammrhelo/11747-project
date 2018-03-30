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



def evaluate_loss(model, data, crit, args):
    model.eval()
    cum_loss = 0.
    cum_tgt_words = 0.
    for src_sents, tgt_sents in data_iter(data, batch_size=args.batch_size, shuffle=False):

        sys_utt = [[turn[0] for turn in dial] for dial in src_sents]
        usr_utt = [[turn[1] for turn in dial] for dial in src_sents]
        conf = [[turn[2] for turn in dial] for dial in src_sents]

        src_sents_sys_vars = to_input_variable_src(sys_utt, vocab.src, cuda=args.cuda)
        src_sents_usr_vars = to_input_variable_src(usr_utt, vocab.src, cuda=args.cuda)
        src_sents_conf_vars = to_input_variable_conf(conf, cuda=args.cuda)

        tgt_sents_var = to_input_variable(tgt_sents, vocab.tgt, cuda=args.cuda)

        src_sents_len = [len(s) for s in src_sents]
        pred_tgt_word_num = sum(len(s[1:]) for s in tgt_sents) # omitting leading `<s>`

        scores, hidden_, attn_ = model(src_sents_sys_vars, src_sents_usr_vars, src_sents_conf_vars, 
                                                        src_sents_len, tgt_sents_var[:-1])

        loss = crit(scores.view(-1, scores.size(2)), tgt_sents_var[1:].view(-1))

        cum_loss += loss.data[0]
        cum_tgt_words += pred_tgt_word_num

    loss = cum_loss / cum_tgt_words
    return loss


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


def to_input_variable_conf(src_data, cuda=False, is_test=False):
    ret = Variable(torch.FloatTensor(src_data), volatile=is_test, requires_grad=False)
    if cuda:
        ret = ret.cuda()
    return ret 

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
    # train_loader = FakeLetsGoDataLoader(corpus.train)
    # dev_loader = FakeLetsGoDataLoader(corpus.valid)
    # test_loader = FakeLetsGoDataLoader(corpus.test)

    train_loader = LetsGoDataLoader(corpus.train)
    dev_loader = LetsGoDataLoader(corpus.valid)
    test_loader = LetsGoDataLoader(corpus.test)


    train_data = list(zip(train_loader.get_src(), train_loader.get_tgt()))
    dev_data = list(zip(dev_loader.get_src(), dev_loader.get_tgt()))
    test_data = list(zip(test_loader.get_src(), test_loader.get_tgt()))


    vocab, model, optimizer, nll_loss, cross_entropy_loss = init_training(args)

    patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = cum_batches = report_examples = epoch = valid_num = best_model_epoch = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()

    # print('begin Maximum Likelihood training')

    while True:
        epoch += 1
        for src_sents, tgt_sents in data_iter(train_data, batch_size=args.batch_size):

            sys_utt = [[turn[0] for turn in dial] for dial in src_sents]
            usr_utt = [[turn[1] for turn in dial] for dial in src_sents]
            conf = [[turn[2] for turn in dial] for dial in src_sents]

            src_sents_sys_vars = to_input_variable_src(sys_utt, vocab.src, cuda=args.cuda)
            src_sents_usr_vars = to_input_variable_src(usr_utt, vocab.src, cuda=args.cuda)
            src_sents_conf_vars = to_input_variable_conf(conf, cuda=args.cuda)

            tgt_sents_var = to_input_variable(tgt_sents, vocab.tgt, cuda=args.cuda)


            batch_size = len(src_sents)
            src_sents_len = [len(s) for s in src_sents]
            pred_tgt_word_num = sum(len(s[1:]) for s in tgt_sents) # omitting leading `<s>`


            optimizer.zero_grad()

            # (tgt_sent_len, batch_size, tgt_vocab_size)
            #print(src_sents_vars.shape)
            scores, hidden_, attn_ = model(src_sents_sys_vars, src_sents_usr_vars, src_sents_conf_vars,
                                             src_sents_len, tgt_sents_var[:-1])

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



            print('Training: epoch %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch,
                                                                                         report_loss / report_examples,
                                                                                         np.exp(report_loss / report_tgt_words),
                                                                                         cum_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time))

       
            train_time = time.time()
            report_loss = report_tgt_words = report_examples = 0.



        if epoch % args.valid_nepoch == 0:
            print('Validation: epoch %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch,
                                                                                     cum_loss / cum_batches,
                                                                                     np.exp(cum_loss / cum_tgt_words),
                                                                                     cum_examples))

            cum_loss = cum_batches = cum_tgt_words = 0.
            valid_num += 1

            print('begin validation ...')
            model.eval()

            # compute dev. ppl and bleu

            dev_loss = evaluate_loss(model, dev_data, cross_entropy_loss, args)
            dev_ppl = np.exp(dev_loss)

            valid_metric = -dev_ppl
            print('validation: epoch %d, dev. ppl %f' % (epoch, dev_ppl))

            model.train()

            is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
            is_better_than_last = len(hist_valid_scores) == 0 or valid_metric > hist_valid_scores[-1]
            hist_valid_scores.append(valid_metric)

            if valid_num > args.save_model_after:
                model_file = args.save_to + 'current.bin'
                print('Save current model to [%s] at Epoch: [%d]' % (model_file, epoch) )
                torch.save(model.state_dict(), model_file)

            if (not is_better_than_last) and args.lr_decay:
                lr = optimizer.param_groups[0]['lr'] * args.lr_decay
                print('decay learning rate to %f' % lr)
                optimizer.param_groups[0]['lr'] = lr

            if is_better:
                patience = 0
                best_model_epoch = epoch
                model_file = args.save_to + 'best.bin'
                print('Save best model to [%s] at Epoch: [%d]' % (model_file, epoch) )
                torch.save(model.state_dict(), model_file)

            else:
                patience += 1
                print('hit patience %d' % patience)
                if patience == args.patience:
                    print('early stop!')
                    exit(0)

def main():
    train()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
