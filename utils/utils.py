import torch
import os
import numpy as np
from glob import glob
import openmesh as om
import pickle


def makedirs(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_sparse(spmat):
    return torch.sparse.FloatTensor(
        torch.LongTensor(np.array([spmat.tocoo().row,spmat.tocoo().col])),
        torch.FloatTensor(spmat.tocoo().data), torch.Size(spmat.tocoo().shape))


def to_edge_index(mat):
    return torch.LongTensor(np.vstack(mat.nonzero()))


def preprocess_spiral(face, seq_length, vertices=None, dilation=1):
    from .generate_spiral_seq import extract_spirals
    assert face.shape[1] == 3

    if vertices is not None:
        mesh = om.TriMesh(np.array(vertices), np.array(face))
    else:
        try:
            n_vertices = face.max() + 1
        except:
            import pdb
            pdb.set_trace()
        mesh = om.TriMesh(np.ones([n_vertices, 3]), np.array(face))
    spirals = torch.tensor(
        extract_spirals(mesh, seq_length=seq_length, dilation=dilation))
    return spirals

def preprocess_template(transform_fp, seq_length, dilation, rank, dynamic_seq_length=None):
    '''
    wrapper to preprocess template mesh. 
    '''
    #assert os.path.exists(transform_fp), f'{transform_fp}'
    # load template remeshed 
    with open(transform_fp, 'rb') as f:
        tmp = pickle.load(f, encoding='latin1')
    # compute spiral index for every vertex
    spiral_indices_list = [
        preprocess_spiral(tmp['face'][idx], seq_length[idx],
                                tmp['vertices'][idx],
                                dilation[idx]).to(rank)
        for idx in range(len(tmp['face']) - 1)
    ]
    # compute longer spiral index for every vertex
    if dynamic_seq_length:
        dynamic_spiral_indices_list = [
            preprocess_spiral(tmp['face'][idx], dynamic_seq_length[idx],
                                    tmp['vertices'][idx],
                                    dilation[idx]).to(rank)
            for idx in range(len(tmp['face']) - 1)
        ]
    else:
        dynamic_spiral_indices_list = None
    # compute downsampling transformation
    down_transform_list = [
        to_sparse(down_transform).to(rank)
        for down_transform in tmp['down_transform']
    ]
    # compute upsampling transformation
    up_transform_list = [
        to_sparse(up_transform).to(rank)
        for up_transform in tmp['up_transform']
    ]    
    return spiral_indices_list, dynamic_spiral_indices_list, up_transform_list, down_transform_list
