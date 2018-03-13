import math

from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable

from datasets import load_inscript
from model import EntityNLM

device = 0 if torch.cuda.is_available() else -1

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn',type=str,default="LSTM",help="GRU | LSTM")
    parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--bptt_len',type=int,default=15)
    parser.add_argument('--embed_dim',type=int,default=300)
    parser.add_argument('--hidden_size',type=int,default=128)
    parser.add_argument('--num_layers',type=int,default=2)
    parser.add_argument('--dropout',type=float,default=0.5)
    parser.add_argument('--num_epochs',type=int,default=30)
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--early_stop',type=int,default=3)
    args = parser.parse_args()
    return args

args = parse_arguments()

rnn = args.rnn
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
model = EntityNLM(rnn, vocab_size, embed_dim, hidden_size, num_layers, dropout)
if torch.cuda.is_available(): model.cuda(device)

binarycrossentropy = nn.BCELoss()
crossentropy = nn.CrossEntropyLoss()
mse = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def repack(hidden_states):
    if type(hidden_states) == list or type(hidden_states) == tuple:
        hidden_states = [ Variable(state.data) for state in hidden_states ]
    else:
        hidden_states =  Variable(hidden_states.data)
    
    if torch.cuda.is_available():
        if type(hidden_states) == list or type(hidden_states) == tuple:
            hidden_states = [ state.cuda() for state in hidden_states ]
        else:
            hidden_states = hidden_states.cuda()
    
    return hidden_states

# Training Functions
def run_epoch(data_iter, model, optimizer=None):
    train = True if optimizer is not None else False
    if train:
        model.train()
    else:
        model.eval()

    epoch_lossX = 0
    epoch_lossR = 0
    epoch_lossL = 0
    epoch_lossE = 0

    count = 0

    hidden_states = model.init_hidden_states(data_iter.batch_size)

    for batch in tqdm(data_iter):
        # hidden states
        hidden_states = repack(hidden_states)

        # batch_size, bptt_len
        inputs = batch.text
        
        targetX = batch.target
        targetR = batch.R
        targetL = batch.L
        targetE = batch.E

        if train:
            optimizer.zero_grad()

        X, R, L, E, hidden_states = model.forward(inputs, hidden_states)

        # Loss
        lossX = crossentropy(X.transpose(1,2), targetX)
        lossR = binarycrossentropy(R.transpose(1,2), targetR)
        lossL = crossentropy(L.transpose(1,2), targetL)
        lossE = mse(E.transpose(1,2),targetE)
        if train:
            lossX.backward(retain_graph=True)
            lossR.backward(retain_graph=True)
            lossL.backward(retain_graph=True)
            lossE.backward(retain_graph=True)
            optimizer.step()

        # Accumulate statistics
        epoch_lossX += lossX.data[0] * inputs.shape[1]
        epoch_lossR += lossR.data[0] * inputs.shape[1]
        epoch_lossL += lossL.data[0] * inputs.shape[1]
        epoch_lossE += lossE.data[0] * inputs.shape[1]

        count += inputs.shape[1]

    epoch_lossX /= count
    epoch_lossR /= count
    epoch_lossL /= count
    epoch_lossE /= count

    return epoch_lossX, epoch_lossR, epoch_lossL, epoch_lossE


best_valid_loss = None
early_stop_count = 0

for epoch in range(1,num_epochs+1):
    train_loss = run_epoch(train_iter, model, optimizer)
    valid_loss = run_epoch(valid_iter, model)
    print("Epoch",epoch,'train_loss', train_loss, 'valid_loss', valid_loss)
    print("Epoch",epoch,'train ppl', math.exp(train_loss[0]), 'valid ppl',math.exp(valid_loss[0]))
    if best_valid_loss == None or valid_loss[0] < best_valid_loss:
        best_valid_loss = valid_loss[0]
        torch.save(model.state_dict(),'best.pt')
        early_stop_count = 0
    else:
        early_stop_count += 1

    if early_stop_count >= early_stop:
        print("Early stopping criteria met!")
        break

print("Running model with the best valid loss...")
model.load_state_dict(torch.load('best.pt'))
valid_loss = run_epoch(valid_iter, model)
print("valid_loss", valid_loss,"valid ppl",math.exp(valid_loss[0]))
test_loss = run_epoch(test_iter, model)
print("test_loss", test_loss,"test ppl",math.exp(test_loss[0]))
