import argparse

from datasets import load_inscript

parser = argparse.ArgumentParser()
parser.add_argument("--model_pickle",type=str,default="best.pt")
args = parser.parse_args()

batch_size = 1
embed_dim = 300
bptt_len = 15
device = -1
model_pickle = args.model_pickle

train_iter, valid_iter, test_iter, vocab_size = load_inscript(embed_dim,batch_size,bptt_len,device)


import pdb; pdb.set_trace()