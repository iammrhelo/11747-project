from glob import glob
import os
import pickle
import random
import shutil
import xml.etree.ElementTree

from tqdm import tqdm

def split_datasets(xml_dirs, output_dir):
    # Train 
    train_dir = os.path.join(output_dir,'train')
    valid_dir = os.path.join(output_dir,'valid')
    test_dir = os.path.join(output_dir,'test')

    for dirname in [ train_dir, valid_dir, test_dir ]:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        else:
            print("Dataset already split!")
            return

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

def build_dict(xml_dir):
    d = dict()
    for xml_file in tqdm(glob(os.path.join(xml_dir,"*.xml"))):
        root = xml.etree.ElementTree.parse(xml_file).getroot()

        content = root[0][0].text
        sentences = content.split('\n')

        for sentence in sentences:
            for word in sentence.split():
                if word not in d:
                    d[ word ] = len(d)+1
    
    with open(os.path.join(xml_dir,'dict.pickle'),'wb') as fout:
        pickle.dump(d,fout)

if __name__ == "__main__":

    name = "InScript"
    output_dir = "data/InScript"
    xml_dirs = list(filter(os.path.isdir,glob("InScript/corpus/*")))

    split_datasets(xml_dirs, output_dir)

    train_xml_dirs = os.path.join(output_dir,'train')

    build_dict(train_xml_dirs)