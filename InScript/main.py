import math

from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable

from datasets import load_inscript
from model import RNNLM

device = 0 if torch.cuda.is_available() else -1

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--bptt_len',type=int,default=15)
    parser.add_argument('--embed_dim',type=int,default=300)
    parser.add_argument('--hidden_size',type=int,default=300)
    parser.add_argument('--num_layers',type=int,default=2)
    parser.add_argument('--dropout',type=float,default=0.5)
    parser.add_argument('--num_epochs',type=int,default=20)
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--early_stop',type=int,default=3)
    args = parser.parse_args()
    return args

args = parse_arguments()

batch_size = args.batch_size
bptt_len = args.bptt_len
embed_dim = args.embed_dim
hidden_size = args.hidden_size
num_layers = args.num_layers
dropout = args.dropout
num_epochs = args.num_epochs
lr = args.lr
early_stop = args.early_stop

# Load iterators
train_iter, valid_iter, test_iter, vocab_size = load_inscript(embed_dim, batch_size, bptt_len, device)

# Model Related
model = RNNLM(vocab_size, embed_dim, hidden_size, num_layers, dropout)
if torch.cuda.is_available(): model.cuda(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# Training Functions
def run_epoch(data_iter, model, optimizer=None):
    train = True if optimizer is not None else False
    if train:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.
    correct = 0
    count = 0

    entity_correct = 0
    entity_count = 0

    hidden_states = model.init_hidden_states(data_iter.batch_size)
    if torch.cuda.is_available():
        hidden_states = [ state.cuda() for state in hidden_states ]

    for batch in tqdm(data_iter):
        
        hidden_states = [ Variable(state.data) for state in hidden_states ]
        if torch.cuda.is_available():
            hidden_states = [ state.cuda() for state in hidden_states ]

        # batch_size, bptt_len
        inputs = batch.text.transpose(0,1)
        targets = batch.target.transpose(0,1)
        location = batch.location

        if train:
            optimizer.zero_grad()

        outputs, hidden_states = model.forward(inputs, hidden_states)

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


best_valid_loss = None
early_stop_count = 0

for epoch in range(1,num_epochs+1):
    train_loss, train_acc, train_entity_acc = run_epoch(train_iter, model, optimizer)
    valid_loss, valid_acc, valid_entity_acc = run_epoch(valid_iter, model)
    print("Epoch",epoch)
    print("train_loss", train_loss,"train ppl",math.exp(train_loss),"train_acc",train_acc, 'train_entity_acc', train_entity_acc)
    print("valid_loss", valid_loss,"valid ppl",math.exp(valid_loss),"valid_acc",valid_acc, 'valid_entity_acc', valid_entity_acc)
    if best_valid_loss == None or valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(),'best.pt')
        early_stop_count = 0
    else:
        early_stop_count += 1

    if early_stop_count >= early_stop:
        print("Early stopping criteria met!")
        break

print("Running model with the best valid loss...")
model.load_state_dict(torch.load('best.pt'))
valid_loss, valid_acc, valid_entity_acc = run_epoch(valid_iter, model)
print("valid_loss", valid_loss,"valid ppl",math.exp(valid_loss),"valid_acc",valid_acc, 'valid_entity_acc', valid_entity_acc)
test_loss, test_acc, test_entity_acc = run_epoch(test_iter, model)
print("test_loss", test_loss,"test ppl",math.exp(test_loss),"test_acc",test_acc, 'test_entity_acc', test_entity_acc)
