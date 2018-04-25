from vocab import *
from data_utils import *
import torch
import nltk
import string

class LetsGoEntityRawDataLoader():
    def __init__(self, raw_data, norm_data, vocab, use_cuda=False):

        self.raw_data = raw_data
        self.norm_data = norm_data
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
        for dial_idx, (raw_dial, norm_dial) in enumerate(zip(self.raw_data, self.norm_data),1):
            # One Document
            entity_dict = {}
            X_s = []
            R_s = []
            E_s = []
            L_s = []
            
            for i, turn in enumerate(raw_dial):
                norm_turn = norm_dial[i]
                
                if turn[0] in ['<d>', '<example>', '<hang_up>', '<restart>']:
                    sys = [turn[0]]
                else:
                    sys = nltk.word_tokenize(turn[0].replace(".", " .").replace(". .", ".."))
                    
                usr = nltk.word_tokenize(turn[1].replace("a m", "am")\
                                         .replace("p m", "pm")\
                                         .replace('oharamckees', 'oharamckees mckees')\
                                         .replace('pman', 'p man')\
                                         .replace('amain', 'a man')\
                                         .replace('51a man', '51a main')\
                                         .replace('amckeesport', 'a mckeesport')\
                                         .replace('88amoroeville', '88a moroeville')
                                        )
                #print(turn[1])
                #.replace("i'm", "i 'm").replace("it's", "it 's").strip().split(' ')
                
                sys_n = norm_turn[0].strip().split(' ')
                usr_n = norm_turn[1].strip().split(' ')
                
                assert len(sys) and len(usr)

                if sys[-1] not in string.punctuation:
                    sys.append('.')
                if usr[-1] not in string.punctuation:
                    usr.append('.')
                if sys_n[-1] not in string.punctuation:
                    sys_n.append('.')
                if usr_n[-1] not in string.punctuation:
                    usr_n.append('.')

                assert len(sys) > 1 and len(usr) > 1

                sysR = self.get_R(sys, sys_n)
                usrR = self.get_R(usr, usr_n)
                
                sysE = self.get_E(sys, sys_n, entity_dict)
                usrE = self.get_E(usr, usr_n, entity_dict)
                
                sysL = self.get_L(sysR)
                usrL = self.get_L(usrR)
                
                assert len(sysR) == len(sysE) == len(sysL)
                assert len(usrR) == len(usrE) == len(usrL)
            
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
            
            doc = [ (X_s, R_s, E_s, L_s) ]

            tensor_doc = self.to_tensor(doc)

            documents.append((str(dial_idx), tensor_doc))

            entities.append(entity_dict)
        
        return documents, entities
    
    def get_R(self, sent, sent_n):
        try:
            ret = []
            ptr = 0
            for i, word in enumerate(sent_n):
                if '<' in word and (word not in ['<d>', '<example>', '<hang_up>', '<restart>']):
                    if i+1 >= len(sent_n) or '<'in sent_n[i+1]: 
                        ret.append(1)
                        continue
                    while sent[ptr] != sent_n[i+1]:
                        ret.append(1)
                        ptr += 1
                    ptr -= 1
                else: 
                    ret.append(0)
                    ptr += 1

            return ret
        except:
            print(sent)
            print(sent_n)
    
    def get_E(self, sent, sent_n, entity_dict):
        ret = []
        ptr = 0
        for i, word in enumerate(sent_n):
            # Check if is entity
            if '<' in word and (word not in ['<d>', '<example>', '<hang_up>', '<restart>']):
                if word not in entity_dict: 
                    entity_dict[ word ] = len(entity_dict)+1
                entity_idx = entity_dict.get(word)
                
                if i+1 >= len(sent_n) or '<'in sent_n[i+1]: 
                    ret.append(1)
                    continue
                while sent[ptr] != sent_n[i+1]:
                    ret.append(entity_idx)
                    ptr += 1
                ptr -= 1
            else:
                ret.append(0)
                ptr += 1
                        
        return ret
    
    def get_L(self, R):
        R_rev = reversed(R)
        ret = []
        c = 1
        for i, each in enumerate(R_rev):
            if each == 0:
                ret.append(1)
                c = 1
            else:
                ret.append(c)
                c += 1  
        return list(reversed(ret))

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

class Entity():
    def __init__(self):

        self.entity2id = {}
        self.id2entity = {}
        
    def add_entity(self, ent):
        if ent not in self.entity2id:
            eid = self.entity2id[ent] = len(self.entity2id)
            self.id2entity[eid] = ent
            return eid
        else:
            return self.entity2id[ent]

if __name__ == '__main__':
    vocab = torch.load('./data/vocab-raw.bin')
    corpus_r = LetsGoCorpus('./data/raw_data-1ab.p')
    corpus_n = LetsGoCorpus('./data/norm_data-1ab.p')
    train_loader = LetsGoEntityRawDataLoader(corpus_r.train, corpus_n.train, vocab.src)


