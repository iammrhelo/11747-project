from collections import defaultdict
import copy
from glob import glob
import os
import pickle as pkl
import xml.etree.ElementTree

import numpy as np
import torch

from vocab import Vocab, VocabEntry

np.set_printoptions(precision=3)
use_cuda = torch.cuda.is_available()


def to_tensor(doc):
    X, R, E, L = doc[0]
    
    tX = []
    tR = []
    tE = []
    tL = []

    for sent_idx in range(len(X)):
    
        tx = torch.from_numpy(np.array(X[sent_idx]))
        tr = torch.from_numpy(np.array(R[sent_idx]))
        te = torch.from_numpy(np.array(E[sent_idx]))
        tl = torch.from_numpy(np.array(L[sent_idx]))      

        if use_cuda:
            tx = tx.cuda()
            tr = tr.cuda()
            te = te.cuda()
            tl = tl.cuda()

        tX.append(tx)
        tR.append(tr)
        tE.append(te)
        tL.append(tl)      

    return [(tX, tR, tE, tL)]



class InScriptDataLoader(object):
    def __init__(self, xml_dir, dict_pickle):
        self.name = os.path.basename(xml_dir)
        self.use_cuda = use_cuda

        # Use Predefined Dictionary 
        with open(dict_pickle,'rb') as fin:
            self.dictionary = pkl.load(fin)
        
        self.documents = []

        for xml_file in sorted(glob(os.path.join(xml_dir,"*.xml"))):
            doc_name = os.path.basename(xml_file)
            doc = self.parse_document(xml_file, self.dictionary)
            tensor_doc = to_tensor(doc)
            self.documents.append((doc_name,tensor_doc))

    def parse_document(self, xml_file, dictionary, debug=False):
        root = xml.etree.ElementTree.parse(xml_file).getroot()

        content = root[0][0].text
        sentences = content.split('\n')

        R = []
        E = [] 
        L = []

        participants = root[1][0]

        entity_table = {}

        for sent in sentences:
            R.append([0] * len(sent.split()))
            E.append([0] * len(sent.split())) 
            L.append([1] * len(sent.split())) 

        for label in participants:
            sentence_id, word_id = map(int, label.attrib['from'].split('-'))
            
            entity = label.attrib["name"]

            text = label.attrib["text"]
            tokens = text.split()

            start = word_id-1
            end = start+1
            if 'to' in label.attrib:
                _, end = map(int,label.attrib['to'].split('-'))

            if any(x == 0 for x in E[sentence_id-1][start:end]):
            
                if entity not in entity_table:
                    entity_table[entity] = len(entity_table)+1
                entity_id = entity_table[entity]

                # R : is entity?
                # E : entity index

                for idx in range(start,end,1):
                    E[sentence_id-1][idx] = entity_id
                    R[sentence_id-1][word_id-1] = 1

                # L : entity remaining length
                for l in range(len(tokens),0,-1):
                    idx = word_id-1 + len(tokens)-l
                    L[sentence_id-1][idx] = l

        X = []
        for sent in sentences:
            x = []
            for word in sent.split():
                xidx = self.dictionary.get(word,0)
                x.append(xidx)
            X.append(x)
        
        doc = []
        doc.append((X,R,E,L))
        return doc


class LetsGoCorpus(object):

    def __init__(self, data_path):
        train, valid, test = pkl.load(open(data_path, "rb"), encoding='bytes')
        self.train = self.load_data(train)
        self.valid = self.load_data(valid)
        self.test = self.load_data(test)
        print("Loaded %d train %d valid and %d test" % (len(self.train), len(self.valid), len(self.test)))

    def load_data(self, data):
        ret_dial = []
        for dial in data:
            ret_turn = []
            for turn in dial:
                sys = turn[0]
                usr = turn[1]
                try:
                    sys = sys.decode()
                except:
                    pass
                try:
                    usr = usr.decode()
                except:
                    pass
                ret_turn.append((sys, usr, turn[2], turn[3]))

            ret_dial.append(ret_turn)
        return ret_dial

    def get_train_sents(self):
        sys_sents = []
        usr_sents = []
        for dial in self.train:
            for turn in dial:
                sys_sents.append(turn[0])
                usr_sents.append(turn[1])

        return sys_sents + usr_sents, sys_sents

def read_corpus_vocab(corp, source):
    data = []
    for line in corp:
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data

def data_iter(data, batch_size, shuffle=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of source sentences in each batch is decreasing
    """ 
    buckets = defaultdict(list)
    for pair in data:
        src_sent = pair[0]
        buckets[len(src_sent)].append(pair)

    batched_data = []
    for src_len in buckets:
        tuples = buckets[src_len]
        if shuffle: np.random.shuffle(tuples)
        batched_data.extend(list(batch_slice(tuples, batch_size)))

    if shuffle:
        np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch

def batch_slice(data, batch_size, sort=True):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        src_sents = [data[i * batch_size + b][0] for b in range(cur_batch_size)]
        tgt_sents = [data[i * batch_size + b][1] for b in range(cur_batch_size)]

        if sort:
            src_ids = sorted(range(cur_batch_size), key=lambda src_id: len(src_sents[src_id]), reverse=True)
            src_sents = [src_sents[src_id] for src_id in src_ids]
            tgt_sents = [tgt_sents[src_id] for src_id in src_ids]

        yield src_sents, tgt_sents

"""
Data Loader for Lets Go Corpus
For EntityNLM, 
    I:  Whole Dialogue as document => 4k
    II: Previous context and target sentence => 55k
"""
class LetsGoDataLoader():
    def __init__(self, data, vocab):

        self.data = data
        self.vocab = vocab
        
        # For encoder decoder
        self.src = []
        self.tgt = []
        self.max_len = 30
        self.process()

        # For Entity NLM
        self.documents = []
        self.entities = []
        self.build_documents()
       
    def process(self):
        for dial in self.data:
            for i, turn in enumerate(dial):
                src_ctx = []
                if i == 0: continue

                for prev in dial[:i]:
                    sys = prev[0].strip().split(' ')[:self.max_len]
                    usr = prev[1].strip().split(' ')[:self.max_len]
                    src_ctx.append((sys, usr, prev[2], prev[3]))

                self.src.append(src_ctx)

                self.tgt.append(['<s>'] + turn[0].strip().split(' ') + ['</s>'])

    def build_documents(self):
        
        for dial_idx, dial in enumerate(self.data,1):
            # One Document
            entity_dict = {}
            X_s = []
            R_s = []
            E_s = []
            L_s = []
            for i, turn in enumerate(dial):
                sys = turn[0].strip().split(' ')
                usr = turn[1].strip().split(' ')
                
                sys = ['<s>'] + sys + ['</s>']
                usr = ['<s>'] + usr + ['</s>']

                sysR = self.get_R(sys)
                usrR = self.get_R(usr)
                
                sysL = self.get_L(sys)
                usrL = self.get_L(usr)
                
                sysE = self.get_E(sys, entity_dict)
                usrE = self.get_E(usr, entity_dict)
            
                sys = [ self.vocab[w] for w in sys ]
                usr = [ self.vocab[w] for w in usr ]
                
                X_s.append(sys)
                X_s.append(usr)
                R_s.append(sysR)
                R_s.append(usrR)
                E_s.append(sysE)
                E_s.append(usrE)
                L_s.append(sysL)
                L_s.append(usrL)
            

            assert len(X_s) % 2 == 0 
            doc = [ (X_s, R_s, E_s, L_s) ]

            # Convert to Tensor
            tensor_doc = to_tensor(doc)
            
            # Appent to self
            self.documents.append((str(dial_idx), tensor_doc))
            self.entities.append(entity_dict)

    def get_src(self):
        return self.src

    def get_tgt(self):
        return self.tgt

    def get_R(self, sent):
        ret = []
        for word in sent:
            if word.startswith('<') and word not in ['<s>','</s>']:
                ret.append(1)
            else:
                ret.append(0)
        return ret
    
    def get_E(self, sent, entity_dict):
        ret = []
        for word in sent:
            # Check if is entity
            if word.startswith('<') and word not in ['<s>','</s>']:
                if word not in entity_dict: 
                    entity_dict[ word ] = len(entity_dict)+1
                entity_idx = entity_dict.get(word)
            else:
                entity_idx = 0
            ret.append(entity_idx)
        
        return ret
    
    def get_L(self, sent):
        ret = [1 for _ in sent]
        return ret

    def to_tensor(self, doc):
        X, R, E, L = doc[0]
        
        tX = []
        tR = []
        tE = []
        tL = []

        for sent_idx in range(len(X)):
        
            tx = torch.from_numpy(np.array(X[sent_idx]))
            tr = torch.from_numpy(np.array(R[sent_idx]))
            te = torch.from_numpy(np.array(E[sent_idx]))
            tl = torch.from_numpy(np.array(L[sent_idx]))      

            if use_cuda:
                tx = tx.cuda()
                tr = tr.cuda()
                te = te.cuda()
                tl = tl.cuda()

            tX.append(tx)
            tR.append(tr)
            tE.append(te)
            tL.append(tl)      
        
        assert len(tX) % 2 == 0 

        return [(tX, tR, tE, tL)]

def load_corpus(args):
    dataset_name = args.dataset
    print("Loading dataset",dataset_name)
    if dataset_name == "inscript":
        data_dir = './data/modi'
        dict_pickle = os.path.join(data_dir,'train','dict.pickle')
        train_corpus = InScriptDataLoader(os.path.join(data_dir,'train'), dict_pickle)
        valid_corpus = InScriptDataLoader(os.path.join(data_dir,'valid'), dict_pickle)
        test_corpus = InScriptDataLoader(os.path.join(data_dir,'test'), dict_pickle)
        dictionary = train_corpus.dictionary
    elif dataset_name == "letsgo":
        vocab = torch.load('./data/vocab.bin')
        corpus = LetsGoCorpus('./data/union_data-1ab.p')
        train_corpus = LetsGoDataLoader(corpus.train, vocab.src)
        valid_corpus = LetsGoDataLoader(corpus.valid, vocab.src)
        test_corpus = LetsGoDataLoader(corpus.test, vocab.src)
        dictionary = train_corpus.vocab.word2id
    else:
        raise ValueError("Invalid dataset:",dataset_name)

    return train_corpus, valid_corpus, test_corpus, dictionary

if __name__ == "__main__":
    vocab = torch.load('./data/vocab.bin')
    corpus = LetsGoCorpus('./data/union_data-1ab.p')
    test_loader = LetsGoDataLoader(corpus.test, vocab.src)










