import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

from conv import *

def Pool(x, trans, dim=1):
    row, col = trans._indices()
    value = trans._values().unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value
    out = scatter_add(out, row, dim, dim_size=trans.size(0))
    return out


class SpiralEnblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices=None, dynamic_indices=None,layer=SpiralConv, **kwargs):
        super(SpiralEnblock, self).__init__()
        self.conv = layer(in_channels, out_channels, indices=indices, dynamic_indices=dynamic_indices)
        self.act = nn.ELU()#nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, down_transform):
        out = self.act(self.conv(x))
        out = Pool(out, down_transform)
        return out


class SpiralDeblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices=None,dynamic_indices=None, layer=SpiralConv, **kwargs):
        super(SpiralDeblock, self).__init__()
        self.conv = layer(in_channels, out_channels, indices=indices, dynamic_indices=dynamic_indices)
        self.act = nn.ELU()#nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, up_transform):
        out = Pool(x, up_transform)
        out = self.act(self.conv(out))
        return out

class AE(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels,
                 spiral_indices, down_transform, up_transform, layer=[], 
                 dynamic_spiral_indices=[],**kwargs):
        super(AE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.latent_channels = latent_channels
        self.spiral_indices = spiral_indices
        self.dynamic_spiral_indices = dynamic_spiral_indices
        self.down_transform = down_transform
        self.up_transform = up_transform
        self.num_vert = self.down_transform[-1].size(0)

        # encoder
        self.en_layers = nn.ModuleList()
        for idx in range(len(out_channels)):
            if idx == 0:
                self.en_layers.append(
                    SpiralEnblock(in_channels, out_channels[idx],
                                  self.spiral_indices[idx] ,
                                  dynamic_indices=self.dynamic_spiral_indices[idx], layer=layer[idx]))
            else:
                self.en_layers.append(
                    SpiralEnblock(out_channels[idx - 1], out_channels[idx],
                                  self.spiral_indices[idx], 
                                  dynamic_indices=self.dynamic_spiral_indices[idx], layer=layer[idx]))
        self.en_layers.append(
            nn.Linear(self.num_vert * out_channels[-1], latent_channels))
        
        # decoder
        self.de_layers = nn.ModuleList()
        self.de_layers.append(
            nn.Linear(latent_channels, self.num_vert * out_channels[-1]))
        for idx in range(len(out_channels)):
            if idx == 0:
                self.de_layers.append(
                    SpiralDeblock(out_channels[-idx - 1],
                                  out_channels[-idx - 1],
                                  self.spiral_indices[-idx - 1], 
                                  dynamic_indices=self.dynamic_spiral_indices[-idx - 1], layer=layer[-idx - 1]))
            else:
                self.de_layers.append(
                    SpiralDeblock(out_channels[-idx], out_channels[-idx - 1],
                                  self.spiral_indices[-idx - 1], 
                                  self.dynamic_spiral_indices[-idx - 1], layer=layer[-idx - 1]))
        self.de_layers.append(
            SpiralConv(out_channels[0], in_channels, self.spiral_indices[0]))

        self.apply(self._init_weights)
    
    def _init_weights(self, m): 
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def encoder(self, x):
        for i, layer in enumerate(self.en_layers):
            if i != len(self.en_layers) - 1:
                x = layer(x, self.down_transform[i])
            else:
                x = x.view(-1, layer.weight.size(1))
                x = layer(x)
        return x

    def decoder(self, x):
        num_layers = len(self.de_layers)
        num_features = num_layers - 2
        for i, layer in enumerate(self.de_layers):
            if i == 0:
                x = layer(x)
                x = x.view(-1, self.num_vert, self.out_channels[-1])
            elif i != num_layers - 1:
                x = layer(x, self.up_transform[num_features - i])
            else:
                x = layer(x)
        return x

    def forward(self, x, *indices):
        z = self.encoder(x)
        out = self.decoder(z)
        return out