from glob import glob
import os
import random
import shutil
import xml.etree.ElementTree

def parse_xml(xml_dir, output_file):
    
    with open(output_file,'w') as fout:
        for xml_file in glob(os.path.join(xml_dir,'*.xml')):
            print("Parsing {}...".format(xml_file))
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
                L.append([1] * (len(sent.split())-1) + [0] )

            for label in participants:
                sentence_id, word_id = map(int, label.attrib['from'].split('-'))
                
                entity = label.attrib["name"]

                if entity not in entity_table:
                    entity_table[entity] = len(entity_table)+1
                entity_id = entity_table[entity]

                text = label.attrib["text"]
                tokens = text.split()
                # R : is entity?
                R[sentence_id-1][word_id-1] = 1

                # E : entity index
                start = word_id-1
                end = start+1
                if 'to' in label.attrib:
                    _, end = map(int,label.attrib['to'].split('-'))
                for idx in range(start,end,1):
                    E[sentence_id-1][idx] = entity_id

                # L : entity remaining length
                for l in range(len(tokens),0,-1):
                    idx = word_id-1 + len(tokens)-l
                    L[sentence_id-1][idx] = l
                
            # Debug
            """
            for sent, r, e, l in zip(sentences, R, E, L):
                print(sent)
                print(r)
                print(e)
                print(l)
            """
            for sent, r, e, l in zip(sentences, R, E, L):
                row = sent + '\t' \
                    + ' '.join(map(str,r)) + '\t' \
                    + ' '.join(map(str,e)) + '\t' \
                    + ' '.join(map(str,l)) \
                    + '\n'
                fout.write(row)

def split_datasets(xml_dirs, output_dir):
    # Train 
    train_dir = os.path.join(output_dir,'train')
    valid_dir = os.path.join(output_dir,'valid')
    test_dir = os.path.join(output_dir,'test')

    for dirname in [ train_dir, valid_dir, test_dir ]:
        if not os.path.exists(dirname):
            os.makedirs(dirname)

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

if __name__ == "__main__":

    name = "InScript"
    output_dir = "data/InScript"
    xml_dirs = list(filter(os.path.isdir,glob("InScript/corpus/*")))

    split_datasets(xml_dirs, output_dir)

    train_xml_dirs = os.path.join(output_dir,'train')
    valid_xml_dirs = os.path.join(output_dir,'valid')
    test_xml_dirs = os.path.join(output_dir,'test')
    
    parse_xml(train_xml_dirs, os.path.join(output_dir,'inscript_train.tsv'))
    parse_xml(valid_xml_dirs, os.path.join(output_dir,'inscript_valid.tsv'))
    parse_xml(test_xml_dirs, os.path.join(output_dir,'inscript_test.tsv'))