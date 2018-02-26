from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchtext import data, datasets
from torchtext.vocab import GloVe


device = 0 if torch.cuda.is_available() else -1
"""
    Config
"""
batch_size = 32
bptt_len = 30
embed_dim = 300

# Approach 1:
# set up fields
TEXT = data.Field(lower=True, batch_first=True)

# make splits for data
train, valid, test = datasets.WikiText2.splits(TEXT)

# print information about the data

# build the vocabulary
TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=embed_dim))

# print vocab information
print('len(TEXT.vocab)', len(TEXT.vocab))

# make iterator for splits
train_iter, valid_iter, test_iter = data.BPTTIterator.splits(
    (train, valid, test), batch_size=batch_size, bptt_len=bptt_len, device=device)

"""
    Model Config
"""
class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size) # Probability

    def forward(self, x, hidden_state):
        embed_x = self.embed(x)

        lstm_output, hidden_state = self.lstm(embed_x)

        output = self.linear(lstm_output)

        return output, hidden_state


hidden_size = 100
num_layers = 1

model = RNNLM(len(TEXT.vocab), embed_dim, hidden_size, num_layers)

"""
    Training Config
"""

num_epochs = 10
learning_rate = 1e-3

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(1,num_epochs+1):
    print("Epoch {}:".format(epoch))

    epoch_loss = 0.
    for batch in tqdm(train_iter):
        states = Variable(torch.zeros(batch_size, num_layers, hidden_size))

        # batch_size, bptt_len
        inputs = batch.text.transpose(0,1)
        targets = batch.target.transpose(0,1)

        # Remove previous gradients
        model.zero_grad()

        outputs, states = model.forward(inputs, states)

        transposed_outputs = outputs.transpose(0,2).transpose(0,1)
        transposed_targets = targets.transpose(0,1)

        # Loss
        loss = criterion(transposed_outputs, transposed_targets)
        loss.backward()

        # Accumulate statistics
        epoch_loss += loss.data
    print("Epoch",epoch,"Train loss", epoch_loss)