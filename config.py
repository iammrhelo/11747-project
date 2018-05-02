import argparse
import torch
import numpy as np
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
