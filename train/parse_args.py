import os
import time
import argparse
import datetime
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(add_help=False, fromfile_prefix_chars='@')
parser.add_argument('-lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('-b', type=int, default=16, help='batch size')
parser.add_argument('-nepoch', type=int, default=200, help='number of epochs')
parser.add_argument('-nblock', type=int, default=4, help='number of stages')
parser.add_argument('-nscale', type=int, default=12, help='number of scales')
parser.add_argument('-att', type=str, default="pa", help='attention mode')
parser.add_argument('-K', type=int, default=3, help='kernel size')
parser.add_argument('-alpha', type=float, default=2.0, help='coefficient for the softmax')
parser.add_argument('-tau', type=float, default=10.0, help='parameter for SoftGambel')

parser.add_argument('-zmemo', type=str, default="x", help='memo')
parser.add_argument('-losstype', type=int, default=0, help='type of loss')
parser.add_argument('-userefl', type=int, default=0, help='use reflectivity')
parser.add_argument('-datamode', type=int, default=2, help='training data mode')

args_key, unparsed = parser.parse_known_args()
if len(unparsed) > 0:
    print("unparsed", unparsed)
    # quit()

def get_fresult(dic_):
    #fresult_ = time.strftime('%m%d_')
    fresult_ = ''
    key_params = dic_.keys()
    for key in sorted(key_params):
        value_str = str(dic_[key])
        if value_str.find('_') >= 0:
            value_str = value_str.replace("_", "-")

        if key == 'data':
            continue
            
        fresult_ += str(key) + '=' + value_str + '_'
    return fresult_[:-1]

fresult = get_fresult(vars(args_key))

####################################
## Parsing secondary arguments
####################################
parser2 = argparse.ArgumentParser(parents=[parser],  fromfile_prefix_chars='@')

parser2.add_argument('-data', type=str, default='mpi', help='folder naming: data/[nmaterials]dataset')
parser2.add_argument('-verbose', '-v', type=int, default=1, help='control verbose mode')
parser2.add_argument('-cuda', type=int, default=-1, help='the number of GPU device')
parser2.add_argument('-T', type=int, default=1024)
parser2.add_argument('-ppp', type=float, default=64.0)
parser2.add_argument('-sbr', type=float, default=64.0)
parser2.add_argument('-pnorm', type=int, default=0, help='patch normalization')

parser2.add_argument('-dataroot', type=str, default='../data/', help='data root')
parser2.add_argument('-resroot', type=str, default='../result/', help='result root')
parser2.add_argument('-resume', type=int, default=0, help='resume ')
parser2.add_argument('-perm', type=int, default=0, help='perm')

args, unparsed = parser2.parse_known_args(namespace=args_key) # for jupyter

if args.cuda >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

def update_args(args):
    # for jupyter lab
    args.ddata = os.path.join(args.dataroot, args.data) + '/'
    args.fdata = f"T={args.T}_ppp={args.ppp}_sbr={args.sbr}"
    args.fresult = fresult
    args.dresult = os.path.join(args.resroot, args.data) + f"/{args.fdata}/ours_{fresult}/"
    
    os.makedirs(args.dresult, exist_ok=True)
    print(args)

update_args(args)

# plt.rcParams['image.axis'] = 'off'
# plt.rcParams['axes.grid'] = False
# rc = {"axes.spines.left" : False,
#       "axes.spines.right" : False,
#       "axes.spines.bottom" : False,
#       "axes.spines.top" : False,
#       "xtick.bottom" : False,
#       "xtick.labelbottom" : False,
#       "ytick.labelleft" : False,
#       "ytick.left" : False}
# plt.rcParams.update(rc)
