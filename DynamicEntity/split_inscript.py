import argparse
from collections import Counter
from glob import glob
import json
import os
import pickle
import random
import shutil
import sys
import xml.etree.ElementTree

from tqdm import tqdm

def split_datasets_category(xml_dirs, output_dir):

    train_dir = os.path.join(output_dir,'train')
    valid_dir = os.path.join(output_dir,'valid')
    test_dir = os.path.join(output_dir,'test')

    test_idx = random.randint(0,9)
    valid_idx1 = random.randint(0,9)
    valid_idx2 = random.randint(0,9)
    for idx, xml_dir in enumerate(xml_dirs):
        xml_files = glob(os.path.join(xml_dir,"*.xml"))
        
        random.shuffle(xml_files)   

        test_end = 18 + ( idx == test_idx )
        valid_end = test_end + 9 + ( idx == valid_idx1 ) + ( idx == valid_idx2 )

        test_files = xml_files[:test_end]
        valid_files = xml_files[test_end:valid_end]
        train_files = xml_files[valid_end:]

        for copy_files, copy_dir in [(train_files,train_dir), (valid_files, valid_dir), (test_files, test_dir)]:
            for copy_file in copy_files:
                shutil.copy(copy_file, copy_dir)


def split_datasets_random(xml_dirs, output_dir):
    
    train_dir = os.path.join(output_dir,'train')
    valid_dir = os.path.join(output_dir,'valid')
    test_dir = os.path.join(output_dir,'test')
    
    # Train 
    xml_files = []
    for xml_dir in xml_dirs:
        xml_files += glob(os.path.join(xml_dir,"*.xml"))
    
    random.shuffle(xml_files)

    train_files = xml_files[:637]
    valid_files = xml_files[637:728]
    test_files = xml_files[728:]
    
    assert len(train_files) == 637 and len(valid_files) == 91 and len(test_files) == 182

    for copy_files, copy_dir in [(train_files,train_dir), (valid_files, valid_dir), (test_files, test_dir)]:
        for copy_file in copy_files:
            shutil.copy(copy_file, copy_dir)

def split_datasets_modi(xml_dirs, json_file, output_dir):
    # Train 
    train_dir = os.path.join(output_dir,'train')
    valid_dir = os.path.join(output_dir,'valid')
    test_dir = os.path.join(output_dir,'test')

    # Load json
    with open(json_file,'r') as fin:
        modi_split = json.loads(fin.read())

    for dataset, copy_dir in [('train', train_dir), ('dev', valid_dir), ('test', test_dir) ]:

        files = modi_split[ dataset ]

        for filename in files:
            category = filename.split('_')[0]
            filepath = os.path.join('./InScript/corpus',category, filename + '.xml')

            shutil.copy(filepath, copy_dir)

def build_dict(xml_dir, threshold=10):
    cnt = Counter()

    for xml_file in tqdm(glob(os.path.join(xml_dir,"*.xml"))):
        root = xml.etree.ElementTree.parse(xml_file).getroot()

        content = root[0][0].text
        sentences = content.split('\n')

        for sentence in sentences:
            for word in sentence.split():

                lowercased_word = word.lower()

                cnt[ lowercased_word ] += 1

    filtered_cnt = { k : v for k, v in cnt.items() if v >= threshold }

    d = dict()
    for word, count in filtered_cnt.items():
        if word not in d:
            d[word] = len(d) + 1

    print("threshold",threshold)
    print("Vocab size",len(d))
    
    with open(os.path.join(xml_dir,'dict.pickle'),'wb') as fout:
        pickle.dump(d,fout)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o','--output_dir',type=str, default='./data/modi')
    parser.add_argument('-t','--type',type=str,default='modi',help='category | random | modi')
    parser.add_argument('-j','--json',type=str,default='data/clean_data_split.json')
    parser.add_argument('--min_threshold',type=int,default=10)
    args = parser.parse_args()
    
    output_dir = args.output_dir
    split_type = args.type
    json_file = args.json
    min_threshold = args.min_threshold

    xml_dirs = list(filter(os.path.isdir,glob("InScript/corpus/*")))

    # Train 
    train_dir = os.path.join(output_dir,'train')
    valid_dir = os.path.join(output_dir,'valid')
    test_dir = os.path.join(output_dir,'test')

    for dirname in [ train_dir, valid_dir, test_dir ]:
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    if split_type == "category":
        split_datasets_category(xml_dirs, output_dir)
    elif split_type == "random":
        split_datasets_random(xml_dirs, output_dir)
    elif split_type == "modi":
        split_datasets_modi(xml_dirs, json_file, output_dir)

    train_xml_dirs = os.path.join(output_dir,'train')

    build_dict(train_xml_dirs, threshold=min_threshold)
