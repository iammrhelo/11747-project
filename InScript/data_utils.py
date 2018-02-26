from glob import glob
import os
import random
import xml.etree.ElementTree

def parse_xml(xml_dir, output_file):

    fout = open(output_file,'w')

    for xml_file in glob(os.path.join(xml_dir,"*.xml")):
        print("Parsing {}...".format(xml_file))
        scripts = xml.etree.ElementTree.parse(xml_file).getroot()

        for script in scripts:
            for item in script:
                original = item.attrib["original"]
                fout.write(original + '\n')

    fout.close()


def random_split(text_file, train_file, valid_file, test_file):
    
    with open(text_file,'r') as fin:
        lines = fin.read().split('\n')

    random.shuffle(lines)

    length = len(lines)

    train_end = int(length * 0.8)

    valid_begin = train_end
    valid_end = int(length * 0.9)

    test_begin = valid_end

    with open(train_file,'w') as fout:
        fout.write('\n'.join(lines[:train_end]))
    
    with open(valid_file,'w') as fout:
        fout.write('\n'.join(lines[valid_begin:valid_end]))
    
    with open(test_file,'w') as fout:
        fout.write('\n'.join(lines[test_begin:]))


if __name__ == "__main__":

    name = "second_esd"

    xml_dir = os.path.join("DeScript_LREC2016/esds/",name)

    parse_xml(xml_dir, name + '.txt')

    random_split(name + '.txt', name + "_train.txt", name + "_valid.txt", name + "_test.txt")