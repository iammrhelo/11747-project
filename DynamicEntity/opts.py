import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str,default='inscript',help='debug | inscript | letsgo')
    parser.add_argument('--embed_dim',type=int,default=256)
    parser.add_argument('--hidden_size',type=int,default=256)
    parser.add_argument('--entity_size',type=int,default=256)
    parser.add_argument('--dropout',type=float,default=0.5)
    parser.add_argument('--num_epochs',type=int,default=40)
    parser.add_argument('--optim',type=str,default="adam")
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--early_stop',type=int,default=3)
    parser.add_argument('--pretrained',action="store_true",default=False)
    parser.add_argument('--model_path',type=str,default=None)
    parser.add_argument('--exp',type=str,default="exp")
    parser.add_argument('--tensorboard',type=str,default="runs")
    parser.add_argument('--skip_sentence',type=int,default=3)
    parser.add_argument('--max_entity',type=int,default=30)
    parser.add_argument('--ignore_x',action="store_true",default=False)
    parser.add_argument('--ignore_r',action="store_true",default=False)
    parser.add_argument('--ignore_e',action="store_true",default=False)
    parser.add_argument('--ignore_l',action="store_true",default=False)
    args = parser.parse_args()
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

def build_model_path(exp_dir, model_name):
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    model_path = os.path.join(exp_dir,model_name + '.pt')
    return model_path

if __name__ == "__main__":
    args = parse_arguments()
    model_name = build_model_name(args)
    print(model_name)
