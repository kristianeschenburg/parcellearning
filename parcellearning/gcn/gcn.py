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
    in_feats: torch tensor
        array of node features
    
    n_hidden: list, int
        number of hidden nodes per layer
    
    n_layers: int
        number of layers in network
    
    n_classes: int
        number of classes to predict
    
    activation: function
        activation function to apply
        default: relu
    
    dropout: float
        dropout rate, [0,1]
    """

    def __init__(self,
                 in_feats,
                 n_classes,
                 n_hidden,
                 activation=F.relu,
                 dropout=0.5,
                 random_seed=None):

        super(GCN, self).__init__()

        if random_seed:
            th.manual_seed(random_seed)

        if dropout:
            assert min(
                0, dropout) >= 0, 'Dropout rate must be greater or equal to 0'
            assert max(1, dropout) == 1, 'Dropout rate must be less than 1'

        self.n_layers = len(n_hidden)
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(
            GraphConv(in_feats, n_hidden[0], activation=activation))

        # hidden layers
        for i in range(self.n_layers - 1):
            self.layers.append(
                GraphConv(n_hidden[i], n_hidden[i+1], activation=activation))

        # output layer
        self.layers.append(GraphConv(n_hidden[-1], n_classes))

        self.dropout = nn.Dropout(p=dropout)


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

        h = features

        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)

        return h

    def save(self, filename):

        """

        """

        th.save(self.state_dict(), filename)
