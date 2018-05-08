import os
import math
import argparse
from torch import optim
import torch.nn.utils
from torch.nn.utils import clip_grad_norm
from torch.nn import functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from models.attention import GlobalAttention
from config import init_config
from vocab import Vocab, VocabEntry # For pickling
import pickle
from collections import defaultdict
import time
import sys

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size, args, vocab, num_layers=1, dropout=0.5):
        super(Decoder, self).__init__()

        self.args = args
        self.vocab = vocab

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm_cell = nn.LSTMCell(self.embed_size + self.hidden_size, self.hidden_size)

        self.out = nn.Linear(self.hidden_size, len(vocab.tgt), bias=False)
        self.attn = GlobalAttention(self.hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.tgt_embed = nn.Embedding(len(vocab.tgt), args.embed_size, padding_idx=vocab.tgt['<pad>'])


    def decode(self, src_encoded, h_0, c_0, tgt):

        # t > 0
        tgt_embed = self.tgt_embed(tgt)

        batch_size = src_encoded.shape[1]
        scores = []
        new_tensor = c_0.data.new
        output = Variable(new_tensor(batch_size, self.hidden_size).zero_(), requires_grad=False)

        hidden = (h_0, c_0)

        for embed_t in tgt_embed.split(split_size=1):


            embed_t = embed_t.squeeze(0)
            embed_t = torch.cat([embed_t, output], 1)


            h_t, c_t = self.lstm_cell(embed_t, hidden)
            h_t = h_t.squeeze(0)

            # print(output.shape)
            if len(h_t.shape) < 2:
                h_t = h_t.unsqueeze(0)

            hidden = h_t, c_t

            # print(src_encoded.shape)
            attn_t, attn = self.attn(h_t.unsqueeze(1), src_encoded.transpose(0, 1))
            attn_t = attn_t.squeeze(0)
            output = self.dropout(attn_t)
            score = self.out(output)
            scores += [score]


        scores = torch.stack(scores)
        return scores, hidden, attn

    def greedy(self, src_encoded, h_0, c_0, length = 30):


        sampled_ids = []
        outputs = []

        new_tensor = c_0.data.new
        output = Variable(new_tensor(1, self.hidden_size).zero_(), requires_grad=False)
        hidden = (h_0, c_0)


        start = torch.LongTensor([self.vocab.tgt['<s>']])
        start = Variable(start, volatile = True, requires_grad=False)


        embed_t = self.tgt_embed(start)

        for i in range(length):

            embed_t = torch.cat([embed_t, output], 1)

            h_t, c_t = self.lstm_cell(embed_t, hidden)

            h_t = h_t.squeeze(0)
            # print(output.shape)
            if len(h_t.shape) < 2:
                h_t = h_t.unsqueeze(0)

            hidden = h_t, c_t

            attn_t, attn = self.attn(h_t.unsqueeze(1), src_encoded.transpose(0, 1))
            output = attn_t.squeeze(0)
            outputs.append(output)

            predicted = self.out(output).max(1)[1]
            sampled_ids.append(predicted)

            embed_t = self.tgt_embed(predicted)

        outputs = torch.stack(outputs)
        sampled_ids = torch.stack(sampled_ids, 1)
        sampled_ids = sampled_ids.cpu().data.numpy().flatten()


        return [[sampled_ids]], outputs, attn


def evaluate_loss(model, data, crit, args):
    model.eval()
    cum_loss = 0.
    cum_tgt_words = 0.
    for context, (h, c), tgt_sents in data_iter(data, batch_size=args.batch_size, shuffle=False):

        context_var = list2variable(context, cuda = args.cuda)
        h_var = list2state(h, cuda = args.cuda)
        c_var = list2state(c, cuda = args.cuda)
        context_var = context_var.squeeze(2).transpose(0, 1)
        h_var = h_var.squeeze(1)
        c_var = c_var.squeeze(1)

        tgt_sents_var = to_input_variable(tgt_sents, model.vocab.tgt, cuda=args.cuda)

        pred_tgt_word_num = sum(len(s[1:]) for s in tgt_sents) # omitting leading `<s>`

        scores, hidden_, attn_ = model.decode(context_var, h_var, c_var, tgt_sents_var[:-1])

        loss = crit(scores.view(-1, scores.size(2)), tgt_sents_var[1:].view(-1))

        cum_loss += loss.data[0]
        cum_tgt_words += pred_tgt_word_num

    loss = cum_loss / cum_tgt_words
    return loss


def init_training(args):
    vocab = torch.load(args.vocab)

    model = Decoder(args.embed_size, args.hidden_size, len(vocab.tgt), args, vocab)

    model.train()

    for p in model.parameters():
        p.data.uniform_(-args.uniform_init, args.uniform_init)
    
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

def data_iter(data, batch_size, shuffle=False):

    buckets = defaultdict(list)
    for line in data:
        buckets[len(line[0]) + len(line[1])].append(line)

    batched_data = []
    for src_len in buckets:
        lines = buckets[src_len]
        if shuffle: np.random.shuffle(lines)
        batched_data.extend(list(batch_slice(lines, batch_size)))

    if shuffle:
        np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch


def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i

        # one_context = [data[i * batch_size + b][0] for b in range(cur_batch_size)]
        # one_entities = [data[i * batch_size + b][1] for b in range(cur_batch_size)]
        one_context = [data[i * batch_size + b][0] + data[i * batch_size + b][1] for b in range(cur_batch_size)]
        one_h = [data[i * batch_size + b][2][0] for b in range(cur_batch_size)]
        one_c = [data[i * batch_size + b][2][1] for b in range(cur_batch_size)]
        one_target = [data[i * batch_size + b][3] for b in range(cur_batch_size)]

        #one_batch = (one_context, one_entities, (one_h, one_c), one_target)
        one_batch = (one_context, (one_h, one_c), one_target)
        yield one_batch


def load_data(split = 'train'):
    num_file = 0
    if split == 'train': 
        num_file = 56
    elif split == 'test':
        num_file = 3
    elif split == 'valid':
        num_file = 2

    ret = []
    for i in range(num_file):
        data = pickle.load( open( "./data_yulun/"+ split +"_extracted_data_" + str(i+1) + ".p", "rb" ) )
        ret.extend(data)

    return ret


def list2variable(list_X, cuda=False, is_test=False):
    """
    return a tensor of shape (src_sent_len, batch_size)
    """

    list_X = torch.stack([torch.from_numpy(np.stack(each)) for each in list_X])
    list_X = Variable(torch.FloatTensor(list_X), volatile=is_test, requires_grad=False)
    if cuda:
        list_X = list_X.cuda()

    return list_X


def list2state(list_X, cuda=False, is_test=False):
    """
    return a tensor of shape (src_sent_len, batch_size)
    """

    list_X = torch.from_numpy(np.stack(list_X))
    list_X = Variable(torch.FloatTensor(list_X), volatile=is_test, requires_grad=False)
    if cuda:
        list_X = list_X.cuda()

    return list_X


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


def train():

    args = init_config()
    vocab = torch.load('./data/vocab.bin')
    
    # data loader
    train_data = load_data('train')
    valid_data = load_data('valid')
    test_data = load_data('test')

    vocab, model, optimizer, nll_loss, cross_entropy_loss = init_training(args)

    patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = cum_batches = report_examples = epoch = valid_num = best_model_epoch = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()

    # print('begin Maximum Likelihood training')

    while True:
        epoch += 1
        for context, (h, c), tgt_sents in data_iter(train_data, args.batch_size, shuffle=True):

            context_var = list2variable(context, cuda = args.cuda)
            h_var = list2state(h, cuda = args.cuda)
            c_var = list2state(c, cuda = args.cuda)
            context_var = context_var.squeeze(2).transpose(0, 1)
            h_var = h_var.squeeze(1)
            c_var = c_var.squeeze(1)

            tgt_sents_var = to_input_variable(tgt_sents, vocab.tgt, cuda=args.cuda)

            batch_size = len(tgt_sents)

            pred_tgt_word_num = sum(len(s[1:]) for s in tgt_sents) # omitting leading `<s>`

            optimizer.zero_grad()

            # (tgt_sent_len, batch_size, tgt_vocab_size)
            #print(src_sents_vars.shape)
            scores, hidden_, attn_ = model.decode(context_var, h_var, c_var, tgt_sents_var[:-1])

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

            dev_loss = evaluate_loss(model, valid_data, cross_entropy_loss, args)
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

def init_model(args):
    vocab = torch.load(args.vocab)

    model = Decoder(args.embed_size, args.hidden_size, len(vocab.tgt), args, vocab)
    model.load_state_dict(torch.load(args.load_model_path))
    model.eval()

    return vocab, model

def generate():

    args = init_config()
    vocab = torch.load('./data/vocab.bin')
    
    # data loader
    # train_data = load_data('train')
    # valid_data = load_data('valid')
    test_data = load_data('test')

    vocab, model = init_model(args)

    for context, (h, c), tgt_sents in data_iter(test_data, 1):

        context_var = list2variable(context)
        h_var = list2state(h)
        c_var = list2state(c)
        context_var = context_var.squeeze(2).transpose(0, 1)
        h_var = h_var.squeeze(1)
        c_var = c_var.squeeze(1)

        sampled_ids_all, scores_, attn_  = model.greedy(context_var, h_var, c_var)

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

def print_test():

    args = init_config()
    vocab = torch.load('./data/vocab.bin')
    
    # data loader
    # train_data = load_data('train')
    # valid_data = load_data('valid')
    test_data = load_data('test')
    for context, (h, c), tgt_sents in data_iter(test_data, 1):
        for each in tgt_sents:
            sentence = ' '.join(each[1:-1])
            print(sentence)

if __name__ == '__main__':
    #train()
    #generate()
    print_test()