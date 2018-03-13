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

            entity_locations = []

            participants = root[1][0]

            for sent in sentences:
                entity_locations.append([0] * len(sent.split()))

            for label in participants:
                sentence_id, loc = map(int, label.attrib['from'].split('-'))
                entity_locations[sentence_id-1][loc-1] = 1

            for sent, locations in zip(sentences, entity_locations):
                fout.write(sent + '\t' + ' '.join(map(str,locations)) + '\n' )


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