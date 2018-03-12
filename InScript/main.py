import math

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchtext import data, datasets
from torchtext.vocab import GloVe


from datasets import InScript, EntityLocationIterator

device = 0 if torch.cuda.is_available() else -1
"""
    Config
"""
batch_size = 32
bptt_len = 20
embed_dim = 300

# Approach 1:
# set up fields
TEXT = data.Field(lower=True, batch_first=True)
LOCATION = data.Field(sequential=True, use_vocab=False, tensor_type=torch.ByteTensor)

# make splits for data
#train, valid, test = InScript.splits(TEXT)
train, valid, test = InScript.splits( fields= [("text", TEXT),("location",LOCATION)])

# build the vocabulary
TEXT.build_vocab(train, vectors=GloVe(name="6B", dim=embed_dim))

# print vocab information
print("len(TEXT.vocab)", len(TEXT.vocab))

# make iterator for splits
train_iter, valid_iter, test_iter = EntityLocationIterator.splits(
    (train, valid, test), batch_size=batch_size, bptt_len=bptt_len, device=device, repeat=False)

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


hidden_size = 300
num_layers = 1

model = RNNLM(len(TEXT.vocab), embed_dim, hidden_size, num_layers)
if torch.cuda.is_available(): model.cuda(device)

"""
    Training Config
"""
def run_epoch(data_iter, model, train=False, optimizer=None):
    if train:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.
    correct = 0
    count = 0

    entity_count = 0
    entity_correct = 0

    for batch in tqdm(data_iter):

        states = Variable(torch.zeros(batch_size, num_layers, hidden_size))
        # batch_size, bptt_len
        inputs = batch.text.transpose(0,1)
        targets = batch.target.transpose(0,1)
        location = batch.location

        if train:
            optimizer.zero_grad()

        outputs, states = model.forward(inputs, states)

        transposed_outputs = outputs.transpose(0,2).transpose(0,1)
        transposed_targets = targets.transpose(0,1)

        # Loss
        loss = criterion(transposed_outputs, transposed_targets)
        if train:
            loss.backward()
            optimizer.step()

        _, predicted = transposed_outputs.max(1)

        compare = (predicted == transposed_targets)

        correct += compare.sum().data[0]

        # Accumulate statistics
        epoch_loss += loss.data[0]

        count += 1
        for dim1 in range(location.shape[0]):
            for dim2 in range(location.shape[1]):
                compare_is_one = (compare[dim1][dim2] == 1 ).data[0]
                location_is_one = (location[dim1][dim2] == 1 ).data[0]
                entity_correct += ( compare_is_one and location_is_one )
        entity_count += location.sum().data[0]

    return epoch_loss / count, correct / (count * batch_size * bptt_len), entity_correct / entity_count

num_epochs = 15
learning_rate = 1e-3

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

best_valid_loss = None

for epoch in range(1,num_epochs+1):
    train_loss, train_acc, train_entity_acc = run_epoch(train_iter, model, True, optimizer)
    valid_loss, valid_acc, valid_entity_acc = run_epoch(valid_iter, model, False)
    print("Epoch",epoch)
    print("train_loss", train_loss,"train ppl",math.exp(train_loss),"train_acc",train_acc, 'train_entity_acc', train_entity_acc)
    print("valid_loss", valid_loss,"valid ppl",math.exp(valid_loss),"valid_acc",valid_acc, 'valid_entity_acc', valid_entity_acc)
    if best_valid_loss == None or valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(),'best.pt')

print("Running model with the best valid loss...")
model.load_state_dict(torch.load('best.pt'))
valid_loss, valid_acc, valid_entity_acc = run_epoch(valid_iter, model, False)
print("valid_loss", valid_loss,"valid ppl",math.exp(valid_loss),"valid_acc",valid_acc, 'valid_entity_acc', valid_entity_acc)
test_loss, test_acc, test_entity_acc = run_epoch(test_iter, model, False)
print("test_loss", test_loss,"test ppl",math.exp(test_loss),"test_acc",test_acc, 'test_entity_acc', test_entity_acc)
