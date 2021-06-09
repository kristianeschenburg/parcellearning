import numpy as np

import dgl
from dgl import data
from dgl.data import DGLDataset
import dgl.function as fn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

class MLP(nn.Module):

    """
    Instantiate a simple linear feed-forward network.

    Parameters:
    - - - - -
    num_layers: int
        number of layers in the network
    in_dim: int
        number of input features
    num_hidden: int
        number of nodes per hidden layer
    num_classes: int
        number of output classes
    activation:
        activation function to use after each layer
    feat_drop:
        dropout rate on features
    negative_slope:
        negative slope on LeakyReLU
    """
    
    def __init__(self, in_dim,
                 num_layers, 
                 num_hidden, 
                 num_classes, 
                 activation=F.leaky_relu, 
                 feat_drop=0.1,
                 negative_slope=0.2):

        super(MLP, self).__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.activation = activation

        self.feat_drop = nn.Dropout(feat_drop)

        # input projection (no residual)
        self.layers.append(Linear(in_dim, num_hidden, bias=True))

        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.layers.append(Linear(num_hidden, num_hidden, bias=True))

        self.layers.append(Linear(num_hidden, num_classes, bias=True))

        # initialize model weights
        self.reset_parameters()

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.
        """

        for layer in self.layers:
            layer.reset_parameters()


    def forward(self, inputs=None, **kwds):
        
        """
        Forward pass of network.  

        Parameters:
        - - - - -
        inputs: tensor
            node features

        Returns:
        - - - - -
        logits: tensor
            output layer 
        """

        h = inputs
        h = self.feat_drop(h)

        for l in range(self.num_layers-1):

            h = self.layers[l](h)
            h = self.activation(h)
        
        logits = self.layers[-1](h)

        return logits

    def save(self, filename):

        """
        Save learned model to disk.
        
        Parameters:
        - - - - -
        filename: str
            model name
        """

        torch.save(self.state_dict(), filename)
