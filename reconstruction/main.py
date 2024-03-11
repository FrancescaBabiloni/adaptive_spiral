import pickle
import argparse
import yaml
import shutil
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch_geometric.transforms as T
from psbody.mesh import Mesh

import conv
from reconstruction import AE, run
from datasets import MeshData
from utils import utils, writer, DataLoader, mesh_sampling

config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE', help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='mesh autoencoder')
parser.add_argument('--exp_name', type=str, default='interpolation_exp')
parser.add_argument('--data_fp', type=str, default='./data/', help='folder holding template.obj and transforms.pkl')
parser.add_argument('--dataroot', type=str, default='./data/', help='data_path: dataroot + dataset')

parser.add_argument('--outroot', type=str, default='./out', help = 'build the out_path as "root + reconstruction + dataset +exp_name"')
# dataset hyperparameters
parser.add_argument('--dataset', type=str, default='CoMA', help= "dset tag")

parser.add_argument('--split', type=str, default='interpolation')
parser.add_argument('--test_exp', type=str, default='bareteeth')
parser.add_argument('--n_threads', type=int, default=4)
parser.add_argument('--device_idx', type=int, default=0)

# network hyperparameters
parser.add_argument('--layer', default='SpiralConv')
parser.add_argument('--out_channels', nargs='+',default=[32, 32, 32, 64],type=int)
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--seq_length', type=int, default=[9, 9, 9, 9], nargs='+', help="rf size for spiralconv")
parser.add_argument('--dynamic_seq_length', type=int, default=[1, 1, 1, 40], nargs='+', help="max rf for adaptive spiral")
parser.add_argument('--dilation', type=int, default=[1, 1, 1, 1], nargs='+')
parser.add_argument('--ds_factors', type=int, default=[4, 4, 4, 4], nargs='+')

# optimizer hyperparmeters
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay', type=float, default=0.99)
parser.add_argument('--decay_step', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=0)

# training hyperparameters
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=300)

# others
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--description', type=str, default='')
parser.add_argument('--fp16', type=bool, default=False)
parser.add_argument('--print_params_epoch', type=int, default=20)

def _getattr(_lib, name):
    #allow access to nn.Modules as default
    if 'nn' in name: 
        return getattr(nn, name.replace('nn.', '')) 
    else:
        return getattr(_lib, name)

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    return args

args = _parse_args()

args.out_dir = osp.join(args.outroot, 'reconstruction', args.dataset, args.exp_name)
args.checkpoints_dir = osp.join(args.out_dir, 'checkpoints')
print(args)

utils.makedirs(args.out_dir)
utils.makedirs(args.checkpoints_dir)

#shutil.copy('./reconstruction/network.py', args.out_dir) 
#shutil.copy('./conv/adaptive_spiralconv.py', args.out_dir) 

if args.fp16:
    scaler = torch.cuda.amp.GradScaler()
else:
    scaler = None

writer = writer.Writer(args)
device = torch.device('cuda', args.device_idx)
torch.set_num_threads(args.n_threads)

# deterministic
torch.manual_seed(args.seed)
cudnn.benchmark = False
cudnn.deterministic = True

# load dataset
template_fp = osp.join(args.data_fp, args.dataset, 'template', 'template.obj')
meshdata = MeshData(osp.join(args.dataroot, args.dataset),
                    template_fp,
                    dset= args.dataset,
                    split=args.split,
                    test_exp=args.test_exp)

train_loader = DataLoader(meshdata.train_dataset,
                          batch_size=args.batch_size,
                          shuffle=True)
test_loader = DataLoader(meshdata.test_dataset, 
                        shuffle=False, 
                        batch_size=args.batch_size)
print(f'TRAIN:{len(meshdata.train_dataset)}')
print(f'TEST:{len(meshdata.test_dataset)}')
eval_loader = None

# generate/load transform matrices
transform_fp = osp.join(args.data_fp, args.dataset, 'transform', 'transform_{}.pkl'.format(args.ds_factors[-1]))

if not osp.exists(transform_fp):
    print('Generating transform matrices...')
    mesh = Mesh(filename=template_fp)
    _, A, D, U, F, V = mesh_sampling.generate_transform_matrices(
        mesh, args.ds_factors)
    tmp = {
        'vertices': V,
        'face': F,
        'adj': A,
        'down_transform': D,
        'up_transform': U
    }

    with open(transform_fp, 'wb') as fp:
        pickle.dump(tmp, fp)
    print('Done!')
    print('Transform matrices are saved in \'{}\''.format(transform_fp))
else:
    with open(transform_fp, 'rb') as f:
        tmp = pickle.load(f, encoding='latin1')

# generate spirals 
spiral_indices_list = [
    utils.preprocess_spiral(tmp['face'][idx], args.seq_length[idx],
                            tmp['vertices'][idx],
                            args.dilation[idx]).to(device)
    for idx in range(len(tmp['face']) - 1)
]

dynamic_spiral_indices_list = [
    utils.preprocess_spiral(tmp['face'][idx], args.dynamic_seq_length[idx],
                            tmp['vertices'][idx],
                            args.dilation[idx]).to(device)
    for idx in range(len(tmp['face']) - 1)
]

# generate down/up sampling 
down_transform_list = [
    utils.to_sparse(down_transform).to(device)
    for down_transform in tmp['down_transform']
]
up_transform_list = [
    utils.to_sparse(up_transform).to(device)
    for up_transform in tmp['up_transform']
]

# get module from name
layer = [_getattr(conv,layer) for layer in  args.layer]

model = AE(args.in_channels, args.out_channels, args.latent_channels,
           spiral_indices_list, down_transform_list, up_transform_list, 
           layer=layer, dynamic_spiral_indices=dynamic_spiral_indices_list).to(device)

writer.print_additional_info(str(utils.count_parameters(model)))
writer.print_additional_info(str(model))
print(model)
print('Number of parameters: {}'.format(utils.count_parameters(model)))

#log config
writer.save_config(args)

# train
optimizer = torch.optim.Adam(model.parameters(),
                             lr=args.lr,
                             weight_decay=args.weight_decay)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            args.decay_step,
                                            gamma=args.lr_decay)
    
run(model, train_loader, test_loader, eval_loader, args.epochs, optimizer, scheduler, writer, scaler, device, 
print_params_epoch=args.print_params_epoch, meshdata=meshdata, out_dir=args.out_dir)
