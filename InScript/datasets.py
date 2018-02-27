from glob import glob
import os
import random
import xml.etree.ElementTree


import torch
from torchtext import data, datasets

class InScript(datasets.LanguageModelingDataset):

    urls = []
    dirname = 'InScript'
    name = 'data'

    @classmethod
    def splits(cls, text_field, root='.', train='inscript_train.txt',
               validation='inscript_valid.txt', test='inscript_test.txt',
               **kwargs):
        return super(InScript, cls).splits(
            root=root, train=train, validation=validation, test=test,
            text_field=text_field, **kwargs)

    @classmethod
    def iters(cls, batch_size=32, bptt_len=15, device=-1, root='./data',
              vectors=None, **kwargs):
        TEXT = data.Field()

        train, val, test = cls.splits(TEXT, root=root, **kwargs)

        TEXT.build_vocab(train, vectors=vectors)

        return data.BPTTIterator.splits(
            (train, val, test), batch_size=batch_size, bptt_len=bptt_len,
            device=device)