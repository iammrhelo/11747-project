from glob import glob
import os
import random
import xml.etree.ElementTree

import spacy

import torch
from torchtext import data, datasets


class InScriptPilot(datasets.LanguageModelingDataset):

    urls = []
    dirname = 'pilot'
    name = 'InScript'

    @classmethod
    def splits(cls, text_field, root='./data', train='pilot_esd_train.txt',
               validation='pilot_esd_valid.txt', test='pilot_esd_test.txt',
               **kwargs):
        return super(InScriptPilot, cls).splits(
            root=root, train=train, validation=validation, test=test,
            text_field=text_field, **kwargs)

    @classmethod
    def iters(cls, batch_size=32, bptt_len=35, device=0, root='.data',
              vectors=None, **kwargs):
        TEXT = data.Field()

        train, val, test = cls.splits(TEXT, root=root, **kwargs)

        TEXT.build_vocab(train, vectors=vectors)

        return data.BPTTIterator.splits(
            (train, val, test), batch_size=batch_size, bptt_len=bptt_len,
            device=device)

class InScriptSecond(datasets.LanguageModelingDataset):

    urls = []
    dirname = 'second'
    name = 'InScript'

    @classmethod
    def splits(cls, text_field, root='./data', train='second_esd_train.txt',
               validation='second_esd_valid.txt', test='second_esd_test.txt',
               **kwargs):
        return super(InScriptSecond, cls).splits(
            root=root, train=train, validation=validation, test=test,
            text_field=text_field, **kwargs)

    @classmethod
    def iters(cls, batch_size=32, bptt_len=30, device=0, root='.data',
              vectors=None, **kwargs):
        TEXT = data.Field()

        train, val, test = cls.splits(TEXT, root=root, **kwargs)

        TEXT.build_vocab(train, vectors=vectors)

        return data.BPTTIterator.splits(
            (train, val, test), batch_size=batch_size, bptt_len=bptt_len,
            device=device)