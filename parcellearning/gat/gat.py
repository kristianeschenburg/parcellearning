import numpy as np

import dgl
from dgl import data
from dgl.data import DGLDataset
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv

import torch
import torch.nn as nn
import torch.nn.functional as F


class GAT(nn.Module):
    
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
    num_heads: list of length (2)
        number of independent heads per layer (multi-head attention mechanisms)
        num_heads[0] = hidden heads
        num_heads[1] = output heads
    activation: 
    feat_drop: float
        layer-wise dropout rate [0,1]
    attn_drop: float
        mechanism-wise dropout rate [0,1]
    negative_slope:
        negative slope of leaky ReLU
    residual:
        use residual connection
    """
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 num_heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope=0.2,
                 residual=False,
                 allow_zero_in_degree=True):
        
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads[0]
        self.num_out_heads = num_heads[-1]
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, self.num_heads,
            feat_drop, attn_drop, negative_slope, False, self.activation, allow_zero_in_degree))
        
        # hidden layers
        for l in range(1, num_layers-1):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * self.num_heads, num_hidden, self.num_heads,
                feat_drop, attn_drop, negative_slope, residual, self.activation, allow_zero_in_degree))
            
        # output layer
        self.gat_layers.append(GATConv(
            num_hidden * self.num_heads, num_classes, self.num_out_heads,
            feat_drop, attn_drop, negative_slope, residual, None, allow_zero_in_degree))

    def forward(self, g=None, inputs=None, **kwds):
        
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
        for l in range(self.num_layers-1):
            h = self.gat_layers[l](g, h).flatten(1)
            h = h.flatten(1)

        # output projection
        logits = self.gat_layers[-1](g,h).mean(1)
        
        return logits

    def save(self, filename):

        """

        """

        torch.save(self.state_dict(), filename)
