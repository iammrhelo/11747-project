
import torch
import torch.nn as nn
from torch.autograd import Variable

class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.5):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size) # Linear
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden_state):
        embed_x = self.embed(x)

        lstm_output, hidden_state = self.rnn(embed_x)

        output = self.linear(lstm_output)

        return output, hidden_state
