import csv
import io
import math
import os

import torch
from torchtext import data, datasets
from torchtext.data import Example
from torchtext.vocab import GloVe
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
        text_field = fields[0][1]
        R_field = fields[1][1]
        E_field = fields[2][1]
        L_field = fields[3][1]
        
        text = []
        R = []
        E = []
        L = []
        with io.open(path, encoding=encoding) as f:
            for line in f:
                columns = line.strip('\n').split('\t')
                if len(columns) == 0:
                    continue
                sentence, is_entity, entity_indices, remaining_length = columns
                text += text_field.preprocess(sentence)
                R += R_field.preprocess(is_entity)
                E += E_field.preprocess(entity_indices)
                L += L_field.preprocess(remaining_length)
        
        def str2int(array):
            return list(map(int,array))

        R = str2int(R)
        E = str2int(E)
        L = str2int(L)
        assert len(text) == len(R)
        assert len(R) == len(E)
        assert len(E) == len(L) 

        examples = [data.Example.fromlist([text, R, E, L], fields)]
        
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

class InScriptIterator(data.BPTTIterator):
    
    def __iter__(self):
        text = self.dataset[0].text
        TEXT = self.dataset.fields['text']
        TEXT.eos_token = None

        R = self.dataset[0].R
        R_field = self.dataset.fields['R']
        R_field.pad_token = 0
        R_field.eos_token = None

        E = self.dataset[0].E
        E_field = self.dataset.fields['E']
        E_field.pad_token = 0
        E_field.eos_token = None

        L = self.dataset[0].L
        L_field = self.dataset.fields['L']
        L_field.pad_token = 0
        L_field.eos_token = None
 
        def convert2tensor(data, data_field):
            data = data + ([data_field.pad_token] * int(math.ceil(len(data) / self.batch_size) *
                                              self.batch_size - len(data)))
                                              
            data_tensor = data_field.numericalize([data], device=self.device, train=self.train)

            data_tensor = data_tensor.view(self.batch_size, -1).t().contiguous()
            return data_tensor

        text_tensor = convert2tensor(text,TEXT)
        R_tensor = convert2tensor(R,R_field)
        E_tensor = convert2tensor(E,E_field)
        L_tensor = convert2tensor(L,L_field)

  
        dataset = Dataset(examples=self.dataset.examples, fields=[
            ('text', TEXT), ('target', TEXT), ('R', R_field),('E',E_field),('L',L_field)])

        while True:
            for i in range(0, len(self) * self.bptt_len, self.bptt_len):
                seq_len = min(self.bptt_len, len(text_tensor) - i - 1)
                if seq_len == 0:
                    raise StopIteration

                yield Batch.fromvars(
                    dataset, self.batch_size, train=self.train,
                    text=text_tensor[i:i + seq_len],
                    R=R_tensor[i:i + seq_len],
                    E=E_tensor[i:i + seq_len],
                    L=L_tensor[i:i + seq_len],
                    target=text_tensor[i + 1:i + 1 + seq_len])
            if not self.repeat:
                raise StopIteration

def load_inscript(embed_dim, batch_size, bptt_len, device):
    # Approach 1:
    # set up fields
    TEXT = data.Field(sequential=True, lower=False, batch_first=True)
    R = data.Field(sequential=True, use_vocab=False, tensor_type=torch.FloatTensor)
    E = data.Field(sequential=True, use_vocab=False, tensor_type=torch.FloatTensor)
    L = data.Field(sequential=True, use_vocab=False, tensor_type=torch.LongTensor)

    # make splits for data
    #train, valid, test = InScript.splits(TEXT)
    train, valid, test = InScript.splits( fields= [("text", TEXT),("R",R),("E",E),("L",L)])
   
    # build the vocabulary
    TEXT.build_vocab(train, vectors=GloVe(name="6B", dim=embed_dim))

    # print vocab information
    vocab_size = len(TEXT.vocab)
    print("Vocabulary", vocab_size)

    # make iterator for splits
    train_iter, valid_iter, test_iter = InScriptIterator.splits(
        (train, valid, test), batch_size=batch_size, bptt_len=bptt_len, device=device, repeat=False)

    return train_iter, valid_iter, test_iter, vocab_size


if __name__ == "__main__":
    embed_dim = 300
    batch_size = 32
    bptt_len = 35
    device = -1

    train_iter, valid_iter, test_iter, vocab_size = load_inscript(embed_dim,batch_size,bptt_len,device)
    
    for batch in train_iter:
        
        import pdb; pdb.set_trace()