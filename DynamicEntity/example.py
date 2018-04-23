from vocab import *
from data_utils import *
import torch


def main():
    vocab = torch.load('./data/vocab.bin')
    corpus = LetsGoCorpus('./data/union_data-1ab.p')
    entity = Entity()
    train_loader = LetsGoEntityDataLoader(corpus.train, entity, vocab.src)
    entity = train_loader.entity
    dev_loader = LetsGoEntityDataLoader(corpus.valid, entity, vocab.src)
    entity = dev_loader.entity
    test_loader = LetsGoEntityDataLoader(corpus.test, entity, vocab.src)
    entity = test_loader.entity


    train_dialogs = train_loader.get_dialogs()
    train_R = train_loader.get_Rs()
    train_E = train_loader.get_Es()
    train_L = train_loader.get_Ls()


    print(train_dialogs[0])
    print(train_R[0])
    print(train_E[0])
    print(train_L[0])


if __name__ == '__main__':
    main()