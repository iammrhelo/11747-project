import argparse
import os

import numpy as np
import torch

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str,default='letsgo',help='debug | inscript | letsgo')
    parser.add_argument('--embed_dim',type=int,default=256)
    parser.add_argument('--hidden_size',type=int,default=256)
    parser.add_argument('--entity_size',type=int,default=256)
    parser.add_argument('--dropout',type=float,default=0.5)
    parser.add_argument('--num_epochs',type=int,default=40)
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--early_stop',type=int,default=3)
    parser.add_argument('--pretrained',action="store_true",default=False)
    parser.add_argument('--model_path',type=str,default=None)
    parser.add_argument('--exp_dir',type=str,default="exp")
    parser.add_argument('--tensorboard',type=str,default="runs")
    parser.add_argument('--skip_sentence',type=int,default=3)
    parser.add_argument('--max_entity',type=int,default=30)
    parser.add_argument('--ignore_x',action="store_true",default=False)
    parser.add_argument('--ignore_r',action="store_true",default=False)
    parser.add_argument('--ignore_e',action="store_true",default=False)
    parser.add_argument('--ignore_l',action="store_true",default=False)
    args = parser.parse_args()
    return args

def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=321, type=int, help='random seed')
    parser.add_argument('--cuda', action='store_true', default=False, help='use gpu')
    parser.add_argument('--vocab', default='./data/vocab.bin', type=str, help='path of the serialized vocabulary')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--beam_size', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--embed_size', default=100, type=int, help='size of word embeddings')
    parser.add_argument('--hidden_size', default=500, type=int, help='size of LSTM hidden states')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout rate')
    parser.add_argument('--uniform_init', default=0.1, type=float, help='if specified, use uniform initialization for all parameters')
    
    parser.add_argument('--decode_max_time_step', default=200, type=int, help='maximum number of time steps used '
                                                                              'in decoding and sampling')

    parser.add_argument('--valid_nepoch', default=1, type=int, help='every n iterations to perform validation')
    parser.add_argument('--valid_metric', default='bleu', choices=['bleu', 'ppl', 'word_acc', 'sent_acc'], help='metric used for validation')
    parser.add_argument('--load_model', default=None, type=str, help='load a pre-trained model')
    parser.add_argument('--save_to', default='./model/', type=str, help='save trained model to')
    parser.add_argument('--save_model_after', default=5, help='save the model only after n validation iterations')
    parser.add_argument('--patience', default=100, type=int, help='training patience')
    parser.add_argument('--clip_grad', default=5., type=float, help='clip gradients')
    parser.add_argument('--max_niter', default=-1, type=int, help='maximum number of training iterations')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=0.8, type=float, help='decay learning rate if the validation performance drops')

    parser.add_argument('--load_model_path', default='./model/best.bin', type=str, help='load trained model from')

    args = parser.parse_args()

    # seed the RNG
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    return args


def build_model_name(args):
    ignore_list = ["early_stop","model_path","tensorboard", "exp", "skip_sentence", "max_entity",\
                    "pretrained",\
                    "ignore_x","ignore_r","ignore_e","ignore_l"]

    attributes = []
    for k, v in sorted(vars(args).items()):
        if k not in ignore_list:
            attrib = "{}_{}".format(k,v)
            attributes.append(attrib)

    model_name = "_".join(attributes)

    return model_name

def build_model_path(args):
    if not os.path.exists(args.exp_dir): 
        os.makedirs(args.exp_dir)
    if args.model_path is None:
        model_name = build_model_name(args)
        model_path = os.path.join(args.exp_dir, model_name + '.pt')
    return model_path


if __name__ == "__main__":
    args = parse_arguments()
    model_name = build_model_name(args)
    print(model_name)
