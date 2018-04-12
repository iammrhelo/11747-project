from glob import glob
import os
import pickle
import random
import shutil
import xml.etree.ElementTree

from tqdm import tqdm
import numpy as np

import torch

class Corpus(object):
    def __init__(self, xml_dir, dict_pickle, use_cuda=False):
        self.name = os.path.basename(xml_dir)
        self.use_cuda = use_cuda

        # Use Predefined Dictionary 
        self.dictionary = self.load_dict(dict_pickle)
        self.documents = []


        for xml_file in sorted(glob(os.path.join(xml_dir,"*.xml"))):
            doc_name = os.path.basename(xml_file)
            doc = self.parse_document(xml_file, self.dictionary)
            tensor_doc = self.to_tensor(doc)
            self.documents.append((doc_name,tensor_doc))
    
        

    def __str__(self):
        return self.name
    
    def load_dict(self, dict_pickle):
        with open(dict_pickle,'rb') as fin:
            obj = pickle.load(fin)
        return obj
    
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
        

        ##############
        #   Check    #
        ##############

        # Check L
        for ls in L:
            for idx, l in enumerate(ls):
                assert l >= 1
                if l > 1:
                    assert ls[idx+1] == l-1

        doc = []
        doc.append((X,R,E,L))
        if debug:
            return doc, sentences
        else:
            return doc


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


if __name__ == "__main__":
    """
    train_dir = './data/InScript/train'
    train_corpus_pickle = os.path.join(train_dir, 'corpus.pickle')
    train_dict_pickle = os.path.join(train_dir,'dict.pickle')
    train_corpus = Corpus(train_dir, train_dict_pickle)
    """
    
    train_dir = './data/modi/train'
    xml_file = os.path.join(train_dir,'bath_001.xml')
    dict_pickle = os.path.join(train_dir, 'dict.pickle')

    output_file = 'debug_corpus.txt'

    corpus = Corpus(train_dir, dict_pickle, use_cuda=False)

    doc, sentences = corpus.parse_document(xml_file, corpus.dictionary, debug=True)

    X, R, E, L = doc[0]

    for sent, x, r, e, l in zip(sentences, X, R, E, L):
        print(sent)
        print(x)
        print(r)
        print(e)
        print(l)
        import pdb; pdb.set_trace()
