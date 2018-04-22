import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import math
import random
import sys
from attention import GlobalAttention, ZhaoAttention
from beam import Beam

class CNNEncoder(nn.Module):
    
    def __init__(self, vocab_size, embed_size, kernel_num = 100, kernel_sizes = [1, 2, 3], dropout=0.4):
        super(CNNEncoder, self).__init__()
        
        V = vocab_size
        D = embed_size
        Ci = 1
        Co = kernel_num
        Ks = kernel_sizes
        self.out_size = len(kernel_sizes) * kernel_num

        self.embed = nn.Embedding(V, D, padding_idx=0)

        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D), padding = (1, 0)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(dropout)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
        
        # if self.args.static:
        #     x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)
        return x


class Encoder(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers=1, dropout=0.4):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size

        self.embed_size = embed_size * 2 + 1
        self.bidirectional = False
        
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, num_layers)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src_embed, hidden=None):
        outputs, hidden = self.lstm(src_embed, hidden)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size, num_layers=1, dropout=0.4):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, num_layers)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden)
        output = self.dropout(output) # (1, B, N)
        #output = output.squeeze(0)  # (1,B,N) -> (B,N)
        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, cnn_encoder, encoder, decoder, args, vocab):
        super(Seq2Seq, self).__init__()

        self.args = args
        self.vocab = vocab

        # Encoder
        self.cnn_encoder = cnn_encoder
        self.encoder = encoder

        # Decoder
        self.tgt_embed = nn.Embedding(len(vocab.tgt), args.embed_size, padding_idx=vocab.tgt['<pad>'])
        self.decoder = decoder

        # prediction layer of the target vocabulary
        self.out = nn.Linear(self.decoder.hidden_size, len(vocab.tgt), bias=False)

        # dropout layer
        self.dropout = nn.Dropout(args.dropout)

        #self.attn = GlobalAttention(self.decoder.hidden_size)
        self.attn = ZhaoAttention()

        #self.bi2decoder_ctx = nn.Linear(self.encoder.hidden_size, self.decoder.hidden_size, bias = False)

        # bidirectional encoder feed
        #self.bi2decoder_c = nn.Linear(self.encoder.hidden_size * 2, self.decoder.hidden_size)
        self.att_src_linear = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
        self.att_vec_linear = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)

        self.use_cuda = args.cuda

    def encode(self, src_sys, src_usr, src_conf, src_len):
        # src: #batch, len(sent), #turns
        n_batch = src_sys.shape[0]
        n_turn = src_sys.shape[2]

        # print(src)
        # print(src.view(-1, src.shape[1]))

        # input here: n_bath*n_turn, len(sent)
        src_sys_embed = self.cnn_encoder(src_sys.view(-1, src_sys.shape[1]))
        src_usr_embed = self.cnn_encoder(src_usr.view(-1, src_usr.shape[1]))

        src_conf_embed = src_conf.view(-1, 1)
        src_embed = torch.cat([src_sys_embed, src_usr_embed, src_conf_embed], dim = 1)

        
        #print(src_embed.view(n_batch, n_turn, -1))
        # src_embed: n_bath*n_turn, hidden_size*#direction
        # input here:n_turn, n_batch, hidden_size*#direction
        src_embed = pack_padded_sequence(src_embed.view(n_turn, n_batch, -1), src_len)
        encoded, (h_n, c_n) = self.encoder(src_embed)
        encoded, _ = pad_packed_sequence(encoded)
        """
        dec_c_0 = self.bi2decoder_c(torch.cat([c_n[0], c_n[1]], 1))
        #dec_c_0 = torch.cat([c_n[0], c_n[1]], 1)
        dec_h_0 = F.tanh(dec_c_0)

        encoded = self.bi2decoder_ctx(encoded)
        """
        return encoded, (h_n, c_n)

    def decode(self, src_encoded, src_len, decoder_init_hidden, tgt):
        
        # Used for teacher Forcing 
        #tgt_embed = self.tgt_embed(tgt)

        trg_seq_len, batch_size = tgt.shape

        src_seq_len, batch_size, src_hidden_size = src_encoded.shape

        # Create data input
        START_IDX = self.vocab.tgt['<s>']
        decoder_input = Variable(torch.LongTensor([ START_IDX ] * batch_size)).unsqueeze(0) # (1, batch_size)
        if self.use_cuda:
            decoder_input = decoder_input.cuda()
        
        decoder_hidden = decoder_init_hidden

        scores_list = []
        attn_weights_list = []
        for t in range(trg_seq_len):
            # Run through decoding timestep
            decoder_embed = self.tgt_embed(decoder_input)
            decoder_output, decoder_hidden = self.decoder(decoder_embed, decoder_hidden)

            # Compute attention weights and output scores
            attn_weights = self.attn(decoder_output, src_encoded)
            attn_weights_list.append(attn_weights)

            context = (attn_weights * src_encoded).sum(0).unsqueeze(0) #(1, batch_size, hidden_size)
            attn_vector = torch.cat([decoder_output, context],dim=-1)
            score_t = self.out(F.tanh(self.att_vec_linear(attn_vector)))

            scores_list.append(score_t)
            # Create Decoder Input
            _, prediction = score_t.max(dim=-1)
            decoder_input = Variable(prediction.data)
          
        # Concatenate to block
        scores = torch.cat(scores_list)
        attn_weights = torch.cat(attn_weights_list).squeeze()
        return scores, decoder_hidden, attn_weights

    def forward(self, src_sys, src_usr, src_conf, src_len, tgt):
        src_encoded, dec_init = self.encode(src_sys, src_usr, src_conf, src_len)
        scores, hidden, attn_weights = self.decode(src_encoded, src_len, dec_init, tgt)
        return scores, hidden, attn_weights

    def greedy(self, src_sys, src_usr, src_conf, src_len, length = 30):

        # n_batch = src.shape[0]
        n_turn = src_sys.shape[2]

        # input here: n_bath*n_turn, len(sent)
        src_sys_embed = self.cnn_encoder(src_sys.view(-1, src_sys.shape[1]))
        src_usr_embed = self.cnn_encoder(src_usr.view(-1, src_usr.shape[1]))
        src_conf_embed = src_conf.view(-1, 1)
        src_embed = torch.cat([src_sys_embed, src_usr_embed, src_conf_embed], dim = 1)

        # src_embed: n_bath*n_turn, hidden_size*#direction
        # input here:n_turn, n_batch, hidden_size*#direction


        src_embed = pack_padded_sequence(src_embed.view(n_turn, 1, -1), src_len)
        encoded, (h_n, c_n) = self.encoder(src_embed)
        encoded, _ = pad_packed_sequence(encoded)


        dec_c_0 = self.bi2decoder_c(torch.cat([c_n[0], c_n[1]], 1))
        dec_h_0 = F.tanh(dec_c_0)

        src_encoded = self.bi2decoder_ctx(encoded)

        decoder_init = (dec_h_0, dec_c_0)

        sampled_ids = []
        outputs = []

        new_tensor = decoder_init[1].data.new
        output = Variable(new_tensor(1, self.decoder.hidden_size).zero_(), requires_grad=False)
        hidden = (decoder_init[0], decoder_init[1])

        start = torch.LongTensor([self.vocab.tgt['<s>']])
        start = Variable(start, volatile = True, requires_grad=False)
        embed_t = self.tgt_embed(start)

        for i in range(length):

            embed_t = torch.cat([embed_t, output], 1)

            h_t, c_t = self.decoder(embed_t, hidden)

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


    # def beam_search(self, src, length = 30, beam_size = 5):

    #     # n_batch = src.shape[0]
    #     # n_turn = src.shape[2]

    #     # input here: n_bath*n_turn, len(sent)
    #     src_embed = self.cnn_encoder(src.view(-1, src.shape[1]))
    #     # src_embed: n_bath*n_turn, hidden_size*#direction
    #     # input here:n_turn, n_batch, hidden_size*#direction
    #     encoded, (h_n, c_n) = self.encoder(src_embed)

    #     dec_c_0 = self.bi2decoder_c(torch.cat([c_n[0], c_n[1]], 1))
    #     dec_h_0 = F.tanh(dec_c_0)

    #     src_encoded = self.bi2decoder_ctx(encoded)

    #     decoder_init = (dec_h_0, dec_c_0)

    #     sampled_ids = []
    #     outputs = []

    #     new_tensor = decoder_init[1].data.new
    #     output = Variable(new_tensor(1, self.decoder.hidden_size).zero_(), requires_grad=False)
    #     hidden = (decoder_init[0], decoder_init[1])

    #     embed_t = self.tgt_embed(self.vocab.tgt['<s>'])

    #     for i in range(length):

    #         embed_t = torch.cat([embed_t, output], 1)

    #         h_t, c_t = self.decoder(embed_t, hidden)
    #         h_t = h_t.unsqueeze(0)
    #         hidden = h_t, c_t

    #         attn_t, attn = self.attn(h_t.unsqueeze(1), src_encoded.transpose(0, 1))
    #         output = attn_t.squeeze(0)

    #         predicted = self.out(output).max(1)[1]
    #         sampled_ids.append(predicted)

    #         embed_t = self.tgt_embed(predicted)

    #     outputs = torch.stack(outputs)
    #     sampled_ids = torch.stack(sampled_ids, 1)
    #     sampled_ids = sampled_ids.cpu().data.numpy().flatten()

    #     return [[sampled_ids]], outputs, attn


