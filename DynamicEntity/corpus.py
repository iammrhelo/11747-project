import csv
from glob import glob
import os
import pickle
import random
import shutil
import xml.etree.ElementTree

from tqdm import tqdm

class Corpus(object):
    def __init__(self, xml_dir, dict_pickle):
        self.name = os.path.basename(xml_dir)

        # Use Predefined Dictionary 
        self.dictionary = self.load_dict(dict_pickle)
        self.documents = []

        for xml_file in glob(os.path.join(xml_dir,"*.xml")):
            doc_name = os.path.basename(xml_file)
            doc = self.parse_document(xml_file, self.dictionary)
            self.documents.append((doc_name,doc))
    
    def __str__(self):
        return self.name

    def parse_document(self, xml_file, dictionary):
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
            L.append([1] * (len(sent.split())-1) + [0] ) # "." has to be 0

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

    def load_dict(self, dict_pickle):
        with open(dict_pickle,'rb') as fin:
            obj = pickle.load(fin)
        return obj

if __name__ == "__main__":
    """
    train_dir = './data/InScript/train'
    train_corpus_pickle = os.path.join(train_dir, 'corpus.pickle')
    train_dict_pickle = os.path.join(train_dir,'dict.pickle')
    train_corpus = Corpus(train_dir, train_dict_pickle)
    """
    debug_dir = './data/InScript/debug_train'
    debug_corpus_pickle = os.path.join(debug_dir, 'corpus.pickle')
    debug_dict_pickle = os.path.join(debug_dir,'dict.pickle')
    debug_corpus = Corpus(debug_dir, debug_dict_pickle)

    import pdb; pdb.set_trace()
