import csv
import io
import math
import os

import torch
from torchtext.data import Example
from torchtext import data, datasets
from torchtext.data.batch import Batch
from torchtext.data.dataset import Dataset

class InScript(data.Dataset):

    urls = []
    dirname = 'InScript'
    name = 'data'

    def __init__(self, path, fields, newline_eos=True,
                 encoding='utf-8', **kwargs):
        """Create a LanguageModelingDataset given a path and a field.
        Arguments:
            path: Path to the data file.
            text_field: The field that will be used for text data.
            newline_eos: Whether to add an <eos> token for every newline in the
                data file. Default: True.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        #fields = [('text', text_field)]
        text_field = fields[0][1]
        location_fields = fields[1][1]
        text = []
        entity = []
        with io.open(path, encoding=encoding) as f:
            for line in f:
                columns = line.strip('\n').split('\t')
                if len(columns) == 0:
                    continue
                sentence, entity_locations = columns
                text += text_field.preprocess(sentence)
                entity += location_fields.preprocess(entity_locations)
                if newline_eos:
                    text.append(u'<eos>')
                    entity.append(0)
        
        entity = list(map(int, entity))
        assert len(text) == len(entity) 

        examples = [data.Example.fromlist([text, entity], fields)]
        
        super(InScript, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, fields, root='.', train='inscript_train.tsv',
               validation='inscript_valid.tsv', test='inscript_test.tsv',
               **kwargs):
        return super(InScript, cls).splits(
            root=root, train=train, validation=validation, test=test,
            fields=fields, **kwargs)

    @classmethod
    def iters(cls, batch_size=32, bptt_len=15, device=-1, root='./data',
              vectors=None, **kwargs):
        TEXT = data.Field()

        train, val, test = cls.splits(TEXT, root=root, **kwargs)

        TEXT.build_vocab(train, vectors=vectors)

        return data.BPTTIterator.splits(
            (train, val, test), batch_size=batch_size, bptt_len=bptt_len,
            device=device)

class EntityLocationIterator(data.BPTTIterator):
    
    def __iter__(self):
        text = self.dataset[0].text
        TEXT = self.dataset.fields['text']
        
        TEXT.eos_token = None
        text = text + ([TEXT.pad_token] * int(math.ceil(len(text) / self.batch_size) *
                                              self.batch_size - len(text)))
                                              
        data = TEXT.numericalize([text], device=self.device, train=self.train)

        data = data.view(self.batch_size, -1).t().contiguous()
        #
        location = self.dataset[0].location
        LOCATION = self.dataset.fields['location']

        location = location + ([0] * int(math.ceil(len(location) / self.batch_size) *
                                              self.batch_size - len(location)))
        location_data = LOCATION.numericalize([location], device=self.device, train=self.train).transpose(0,1)

        location_data = location_data.view(self.batch_size,-1).t().contiguous()

        dataset = Dataset(examples=self.dataset.examples, fields=[
            ('text', TEXT), ('target', TEXT), ('location', LOCATION)])

        while True:
            for i in range(0, len(self) * self.bptt_len, self.bptt_len):
                seq_len = min(self.bptt_len, len(data) - i - 1)
                yield Batch.fromvars(
                    dataset, self.batch_size, train=self.train,
                    text=data[i:i + seq_len],
                    location=location_data[i:i+seq_len],
                    target=data[i + 1:i + 1 + seq_len])
            if not self.repeat:
                raise StopIteration