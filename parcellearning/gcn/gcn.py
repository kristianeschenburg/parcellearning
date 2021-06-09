import dgl
from dgl.data import DGLDataset
from dgl import data

import torch as th
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GraphConv

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

 
class GCN(nn.Module):

    """
    Instantiate a graph convolutional neural network.  Fits
    an inductive model that generalizes to new graphs of different
    topology than the training data.
    
    Parameters:
    - - - - -
    in_dim: torch tensor
        array of node features
    
    num_hidden: list, int
        number of hidden nodes per layer
    
    num_layers: int
        number of layers in network
    
    num_classes: int
        number of classes to predict
    
    activation: function
        activation function to apply
        default: relu
    
    feat_drop: float
        dropout rate, [0,1]

    residual:: bool
        apply residual skip connections
    """

    def __init__(self,
                 in_dim,
                 num_classes,
                 num_hidden,
                 num_layers,
                 feat_drop=0.1,
                 activation=F.relu,
                 negative_slope=0.2,
                 residual=False,
                 allow_zero_in_degree=True):

        super(GCN, self).__init__()

        if feat_drop:
            assert min(
                0, feat_drop) >= 0, 'Dropout rate must be greater or equal to 0'
            assert max(1, feat_drop) == 1, 'Dropout rate must be less than 1'

        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(
            GraphConv(in_dim, num_hidden, activation=activation))

        # hidden layers
        for i in range(self.num_layers - 2):
            self.layers.append(
                GraphConv(num_hidden, num_hidden, activation=activation))

        # output layer
        self.layers.append(GraphConv(num_hidden, num_classes))

        self.dropout = nn.Dropout(p=feat_drop)


    def forward(self, g=None, inputs=None, **kwds):
        """
        Apply a forward pass through the network, given the current weights.
        
        Parameters:
        - - - - -
        G: dgl.graph structure 
            contains the adjacency information of our data
    
        features: torch tensor
            node features
        """

        h=inputs
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)

        return h

    def save(self, filename):

        """

        """

        th.save(self.state_dict(), filename)
