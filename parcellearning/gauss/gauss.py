from parcellearning.conv.gausconv import GAUSConv
import numpy as np

import dgl
from dgl import data
from dgl.data import DGLDataset
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax

import torch
import torch.nn as nn
import torch.nn.functional as F


class GAUSS(nn.Module):

    """
    Instantiate a Graph Attention Network model.
    
    Parameters:
    - - - - -
    num_layers: int
        number of layers in network
    in_dim: int
        input feature dimension
    num_hidden: int
        number of nodes per hidden layer
    num_classes: int
        number of output classes
    num_kernels: list of length (2)
        number of independent kernels per layer (multi-kernel mechanisms)
        num_kernels[0] = hidden kernels
        num_kernels[1] = output output kernels
    feat_drop: float
        layer-wise dropout rate [0,1]
    krnl_drop: float
        mechanism-wise kernel dropout rate [0,1]
    negative_slope:
        negative slope of leaky ReLU
    """

    def __init__(self,
                 in_dim,
                 num_layers,
                 num_hidden,
                 num_classes,
                 num_kernels,
                 activation,
                 feat_drop,
                 krnl_drop,
                 negative_slope,
                 allow_zero_in_degree=True):

        super(GAUSS, self).__init__()

        if feat_drop:
            assert min(
                0, feat_drop) >= 0, 'Dropout rate must be greater or equal to 0'
            assert max(1, feat_drop) == 1, 'Dropout rate must be less than 1'

        if krnl_drop:
            assert min(
                0, krnl_drop) >= 0, 'Dropout rate must be greater or equal to 0'
            assert max(1, krnl_drop) == 1, 'Dropout rate must be less than 1'

        self.num_layers = num_layers
        self.num_kernels = num_kernels[0]
        self.num_out_kernels = num_kernels[-1]
        self.gauss_layers = nn.ModuleList()
        self.activation = activation

        # input layer
        self.gauss_layers.append(GAUSConv(in_feats=in_dim,
                                          out_feats=num_hidden,
                                          num_kernels=self.num_kernels,
                                          feat_drop=feat_drop, 
                                          krnl_drop=krnl_drop,
                                          negative_slope=negative_slope,
                                          activation=self.activation,
                                          allow_zero_in_degree=allow_zero_in_degree))

        # hidden layers
        for l in range(1, num_layers):

            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gauss_layers.append(GAUSConv(in_feats=num_hidden*self.num_kernels,
                                              out_feats=num_hidden,
                                              num_kernels=self.num_kernels,
                                              feat_drop=feat_drop, 
                                              krnl_drop=krnl_drop,
                                              negative_slope=negative_slope, 
                                              activation=self.activation,
                                              allow_zero_in_degree=allow_zero_in_degree))

        # output projection
        self.gauss_layers.append(GAUSConv(in_feats=num_hidden*self.num_kernels,
                                          out_feats=num_classes, 
                                          num_kernels=self.num_out_kernels,
                                          feat_drop=feat_drop, 
                                          krnl_drop=krnl_drop, 
                                          negative_slope=negative_slope,
                                          activation=None,
                                          allow_zero_in_degree=allow_zero_in_degree))

    
    def forward(self, g, inputs):
        
        """
        Parameters:
        - - - - -
        g: DGL Graph
            the graph
        inputs: tensor
            node features
            
        Returns:
        - - - - -
        logits: tensor
            output layer 
        """

        h = inputs
        for l in range(self.num_layers):
            h = self.gauss_layers[l](g, h)
            h = h.flatten(1)

        # output projection
        logits = self.gauss_layers[-1](g,h).mean(1)

        return logits

    def save(self, filename):
        
        """

        """

        torch.save(self.state_dict(), filename)
