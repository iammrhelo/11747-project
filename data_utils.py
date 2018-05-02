from collections import defaultdict
import copy
import pickle as pkl
import string

import numpy as np
import torch

from vocab import Vocab, VocabEntry

np.set_printoptions(precision=3)
use_cuda = torch.cuda.is_available()

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

# Data feed
class DataLoader(object):
    batch_size = 0
    ptr = 0
    num_batch = None
    batch_indexes = None
    indexes = None
    data_size = None
    name = None
    equal_len_batch = None
    # if equal batch, the indexes sorted by data length
    # else indexes = range(data_size)

    def _shuffle_indexes(self):
        np.random.shuffle(self.indexes)

    def _shuffle_batch_indexes(self):
        np.random.shuffle(self.batch_indexes)

    def _prepare_batch(self, selected_indexes):
        raise NotImplementedError("Have to override prepare batch")

    def epoch_init(self, batch_size, shuffle=True):
        if self.name.upper() == "TEST":
            new_batch_size = None
            for i in range(3):
                factors = utils.factors(self.data_size-i)
                temp = min(factors, key=lambda x:abs(x-batch_size))
                if np.abs(temp-batch_size) < batch_size * 0.5:
                    new_batch_size = temp
                    break
            if new_batch_size is not None:
                batch_size = new_batch_size
                print("Adjust the batch size to %d" % batch_size)

        self.ptr = 0
        self.batch_size = batch_size
        self.num_batch = self.data_size // batch_size
        print("Number of left over sample %d" % (self.data_size-batch_size*self.num_batch))

        # if shuffle and we don't want to group lines, shuffle index
        if shuffle and not self.equal_len_batch:
            self._shuffle_indexes()

        self.batch_indexes = []
        for i in range(self.num_batch):
            self.batch_indexes.append(self.indexes[i * self.batch_size:(i + 1) * self.batch_size])

        # if shuffle and we want to group lines, shuffle batch indexes
        if shuffle and self.equal_len_batch:
            self._shuffle_batch_indexes()

        print("%s begins with %d batches" % (self.name, self.num_batch))

    def next_batch(self):
        if self.ptr < self.num_batch:
            selected_ids = self.batch_indexes[self.ptr]
            self.ptr += 1
            return self._prepare_batch(selected_indexes=selected_ids)
        else:
            return None

class LetsGoDataLoader():
    def __init__(self, data, vocab):

        self.data = data
        self.src = []
        self.tgt = []
        self.max_len = 30
        
        self.vocab = vocab
        self.process()

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
        self.documents = []
        self.entities = []
        for dial_idx, dial in enumerate(self.src,1):
            entity_dict = {}
            X_s = []
            R_s = []
            E_s = []
            L_s = []
            for i, turn in enumerate(dial):
           
                sys = turn[0]
                usr = turn[1]

                assert len(sys) and len(usr)

                if sys[-1] not in string.punctuation:
                    sys.append('.')
                if usr[-1] not in string.punctuation:
                    usr.append('.')

                assert len(sys) > 1 and len(usr) > 1

                sysR = self.get_R(sys)
                usrR = self.get_R(usr)
                
                sysL = self.get_L(sys)
                usrL = self.get_L(usr)
                
                sysE = self.get_E(sys, entity_dict)
                usrE = self.get_E(usr, entity_dict)
            
                sys = [self.vocab[w] for w in sys]
                usr = [self.vocab[w] for w in usr]
                
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

            tensor_doc = self.to_tensor(doc)

            self.documents.append((str(dial_idx), tensor_doc))

            self.entities.append(entity_dict)

    def get_src(self):
        return self.src

    def get_tgt(self):
        return self.tgt

    def get_R(self, sent):
        ret = [ int('<' in word) for word in sent]
        return ret
    
    def get_E(self, sent, entity_dict):
        ret = []
        for word in sent:
            # Check if is entity
            if '<' in word:
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

class LetsGoEntityDataLoader():
    def __init__(self, data, vocab, use_cuda=False):

        self.data = data
        self.dialogs = []
        self.R = []
        self.E = []
        self.L = []
        self.entity = [] # invalid entity
        self.vocab = vocab
        
        self.use_cuda = use_cuda

        self.documents, self.entities = self.process()
        
    def process(self):
        documents = []
        entities = []
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
                
                assert len(sys) and len(usr)

                if sys[-1] not in string.punctuation:
                    sys.append('.')
                if usr[-1] not in string.punctuation:
                    usr.append('.')

                assert len(sys) > 1 and len(usr) > 1

                sysR = self.get_R(sys)
                usrR = self.get_R(usr)
                
                sysL = self.get_L(sys)
                usrL = self.get_L(usr)
                
                sysE = self.get_E(sys, entity_dict)
                usrE = self.get_E(usr, entity_dict)
            
                sys = [self.vocab[w] for w in sys]
                usr = [self.vocab[w] for w in usr]
                
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

            tensor_doc = self.to_tensor(doc)

            documents.append((str(dial_idx), tensor_doc))

            entities.append(entity_dict)
        
        return documents, entities


    def get_R(self, sent):
        ret = [ int('<' in word) for word in sent]
        return ret
    
    def get_E(self, sent, entity_dict):
        ret = []
        for word in sent:
            # Check if is entity
            if '<' in word:
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

            if self.use_cuda:
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

    def display_stats(self):
        # Document 
        nwords = [] 
        nentities = []  
        nentities_distinct = []     

        # Sentence 
        nwords_sentence = []
        nentities_sentence = []
        for (doc_name, doc) in self.documents:

            X, R, E, L = doc[0]

            word_count = 0
            entity_count = 0
            for sent_x, sent_r in zip(X,R):
                word_count += sent_x.size()[0]
                entity_count += sent_r.sum()
                nwords_sentence.append(sent_x.size()[0])
                nentities_sentence.append(sent_r.sum())

            nwords.append(word_count)
            nentities.append(entity_count)
        
        nentities_distinct = [ len(entity_dict) for entity_dict in self.entities ]

        def print_mean_std(name, stats):
            mean = np.mean(stats)
            std = np.std(stats)
            maximum = np.max(stats)
            mininum = np.min(stats)

            msg = ""
            msg += "{0:.3f}".format(mean)
            msg += "(+/-{0:.3f})".format(std)
            msg += ", max {}".format(maximum)
            msg += ", min {}".format(mininum)
            print(name,msg)

        print_mean_std("Average number of words per document", nwords)
        print_mean_std("Average number of entities per document", nentities)
        print_mean_std("Average number of distinct entities per document", nentities_distinct)
        print_mean_std("Average number of words per sentences", nwords_sentence)
        print_mean_std("Average number of entites per sentence", nentities_sentence)

def load_corpus(args):
    dataset_name = args.dataset
    print("Loading dataset",dataset_name)
    if dataset_name == "letsgo":
        vocab = torch.load('./data/vocab.bin')
        corpus = LetsGoCorpus('./data/union_data-1ab.p')
        train_corpus = LetsGoEntityDataLoader(corpus.train, vocab.src, use_cuda=use_cuda)
        valid_corpus = LetsGoEntityDataLoader(corpus.valid, vocab.src, use_cuda=use_cuda)
        test_corpus = LetsGoEntityDataLoader(corpus.test, vocab.src, use_cuda=use_cuda)
        dictionary = train_corpus.vocab.word2id
    else:
        raise ValueError("Invalid dataset:",dataset_name)

    return train_corpus, valid_corpus, test_corpus, dictionary

if __name__ == "__main__":
    vocab = torch.load('./data/vocab.bin')
    corpus = LetsGoCorpus('./data/union_data-1ab.p')
    test_loader = LetsGoDataLoader(corpus.test, vocab.src)
    test_loader.build_documents()
    """
    train_loader = LetsGoEntityDataLoader(corpus.train, vocab.src)
    print("Train")
    train_loader.display_stats()
    valid_loader = LetsGoEntityDataLoader(corpus.valid, vocab.src)
    print("Valid")
    valid_loader.display_stats()

    test_loader = LetsGoEntityDataLoader(corpus.test, vocab.src)
    print("Test")
    test_loader.display_stats()
    """










