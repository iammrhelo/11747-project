import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import math
import random
import sys
from attention import GlobalAttention

class CNNEncoder(nn.Module):
    
    def __init__(self, vocab_size, embed_size, kernel_num = 100, kernel_sizes = [1, 2, 3], dropout=0.5):
        super(CNNEncoder, self).__init__()
        
        V = vocab_size
        D = embed_size
        Ci = 1
        Co = kernel_num
        Ks = kernel_sizes
        self.out_size = len(kernel_sizes) * kernel_num

        self.embed = nn.Embedding(V, D)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
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
    def __init__(self, embed_size, hidden_size, num_layers=1, dropout=0.5, bidirectional = True):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size

        self.embed_size = embed_size
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            dropout = dropout, bidirectional = bidirectional)

    def forward(self, src_embed, hidden=None):
        outputs, hidden = self.lstm(src_embed, hidden)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size, num_layers=1, dropout=0.5):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm_cell = nn.LSTMCell(hidden_size + embed_size, hidden_size)

    def forward(self, input, hidden):

        output, hidden = self.lstm_cell(input, hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)

        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, cnn_encoder, encoder, decoder, args, vocab):
        super(Seq2Seq, self).__init__()

        self.args = args
        self.vocab = vocab

        self.tgt_embed = nn.Embedding(len(vocab.tgt), args.embed_size, padding_idx=vocab.tgt['<pad>'])

        self.cnn_encoder = cnn_encoder
        self.encoder = encoder
        self.decoder = decoder

        # prediction layer of the target vocabulary
        self.out = nn.Linear(self.decoder.hidden_size, len(vocab.tgt), bias=False)

        # dropout layer
        self.dropout = nn.Dropout(args.dropout)

        self.attn = GlobalAttention(self.decoder.hidden_size)
        self.bi2decoder_ctx = nn.Linear(self.encoder.hidden_size*2, self.decoder.hidden_size, bias = False)

        # bidirectional encoder feed
        self.bi2decoder_c = nn.Linear(self.encoder.hidden_size * 2, self.decoder.hidden_size)
        self.att_src_linear = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
        self.att_vec_linear = nn.Linear(args.hidden_size * 2 + args.hidden_size, args.hidden_size, bias=False)

    def encode(self, src, src_len):
        # src: #batch, len(sent), #turns
        n_batch = src.shape[0]
        n_turn = src.shape[2]

        # print(src)
        # print(src.view(-1, src.shape[1]))

        # input here: n_bath*n_turn, len(sent)
        src_embed = self.cnn_encoder(src.view(-1, src.shape[1]))
        #print(src_embed.view(n_batch, n_turn, -1))
        # src_embed: n_bath*n_turn, hidden_size*#direction
        # input here:n_turn, n_batch, hidden_size*#direction
        src_embed = pack_padded_sequence(src_embed.view(n_turn, n_batch, -1), src_len)
        encoded, (h_n, c_n) = self.encoder(src_embed)
        encoded, _ = pad_packed_sequence(encoded)

        dec_c_0 = self.bi2decoder_c(torch.cat([c_n[0], c_n[1]], 1))
        #dec_c_0 = torch.cat([c_n[0], c_n[1]], 1)
        dec_h_0 = F.tanh(dec_c_0)

        encoded = self.bi2decoder_ctx(encoded)

        return encoded, (dec_h_0, dec_c_0)

    def decode(self, src_encoded, src_len, decoder_init, tgt):

        # print(src_encoded.shape) # seqlen, batch_size, hidden_size 
        # print(src_len) 
        # print(tgt.shape)
        # sys.exit(0)
        

        # pcyin attention
        # new_tensor = decoder_init[1].data.new
        # batch_size = src_encoded.size(1)

        # # (batch_size, src_sent_len, hidden_size * 2)
        # src_encoded = src_encoded.permute(1, 0, 2)

        # # (batch_size, src_sent_len, hidden_size)
        # src_encoded_att_linear = tensor_transform(self.att_src_linear, src_encoded)


        # # initialize attentional vector
        # att_tm1 = Variable(new_tensor(batch_size, self.decoder.hidden_size).zero_(), requires_grad=False)

        # t > 0
        tgt_embed = self.tgt_embed(tgt)
        
        batch_size = src_encoded.shape[1]
        scores = []
        new_tensor = decoder_init[1].data.new
        output = Variable(new_tensor(batch_size, self.decoder.hidden_size).zero_(), requires_grad=False)
        hidden = (decoder_init[0], decoder_init[1])
        for embed_t in tgt_embed.split(split_size=1):

            embed_t = embed_t.squeeze(0)

            # print(embed_t.shape)
            # print(output.shape)
            embed_t = torch.cat([embed_t, output], 1)


            output, h_t = self.decoder(embed_t, hidden)

            # print(output.shape)
            if len(output.shape) < 2:
                output = output.unsqueeze(0)
            # print(src_encoded.shape)
            output, attn = self.attn(output.unsqueeze(1), src_encoded.transpose(0, 1))
            output = output.squeeze(0)
            output = self.dropout(output)
            score = self.out(output)
            scores += [score]
            hidden = output, h_t

            # pcyin
            # x = torch.cat([y_embed.squeeze(0), att_tm1], 1)


            # h_t, c_t = self.decoder(x, hidden)

            # h_t = self.dropout(h_t)

            # ctx_t, alpha_t = self.dot_prod_attention(h_t, src_encoded, src_encoded_att_linear)

            # att_t = F.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))   # E.q. (5)
            # att_t = self.dropout(att_t)

            # score_t = self.out(att_t)
            # scores.append(score_t)

            # att_tm1 = att_t

            # hidden = h_t, c_t

        scores = torch.stack(scores)
        return scores, hidden, attn


    def dot_prod_attention(self, h_t, src_encoding, src_encoding_att_linear, mask=None):
        """
        :param h_t: (batch_size, hidden_size)
        :param src_encoding: (batch_size, src_sent_len, hidden_size * 2)
        :param src_encoding_att_linear: (batch_size, src_sent_len, hidden_size)
        :param mask: (batch_size, src_sent_len)
        """
        # (batch_size, src_sent_len)
        att_weight = torch.bmm(src_encoding_att_linear, h_t.unsqueeze(2)).squeeze(2)
        if mask:
            att_weight.data.masked_fill_(mask, -float('inf'))

        att_weight = F.softmax(att_weight, dim = 1)

        att_view = (att_weight.size(0), 1, att_weight.size(1))
        # (batch_size, hidden_size)
        ctx_vec = torch.bmm(att_weight.view(*att_view), src_encoding).squeeze(1)

        return ctx_vec, att_weight

    def forward(self, src, src_len, tgt):
        src_encoded, dec_init = self.encode(src, src_len)

        scores = self.decode(src_encoded, src_len, dec_init, tgt)
        return scores

    def greedy(self, src, length = 30):

        src_encoded, dec_init = encode(src, src_len)
        hidden = (dec_init[0], dec_init[1])

        sampled_ids = []
        outputs = []

        emb_t = self.tgt_embed(self.vocab.tgt['<s>'])
        for i in range(length):
            h_t, c_t = self.decoder(embed_t, hidden)
            predicted = self.out(h_t).max(1)[1]
            sampled_ids.append(predicted)
            outputs.append(h_t)

            emb_t = self.tgt_embed(predicted)

        outputs = torch.stack(outputs)
        sampled_ids = sampled_ids.cpu().data.numpy().flatten()

        return [[sampled_ids]], outputs, h_t

    
    def beam_search(self):
        return;
    


def tensor_transform(linear, X):
    # X is a 3D tensor
    return linear(X.contiguous().view(-1, X.size(2))).view(X.size(0), X.size(1), -1)
