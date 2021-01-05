import sys
sys.path.append('../conv/')
from cgatconv import CGATConv

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
from dgl.nn.pytorch import edge_softmax

class CGAT(nn.Module):
    
    """
    Instantiate a Graph Attention Network model.
    
    Parameters:
    - - - - -
    in_dim: int
        input feature dimension
    num_classes: int
        number of output classes
    num_heads: list of length (2)
        number of independent attention heads
        num_heads[0] = hidden heads
        num_heads[1] = output heads
    num_hidden: int
        number of nodes per hidden layer
    num_layers: int
        number of layers in network
    feat_drop: float
        layer-wise feature dropout rate [0,1]
    graph_margin: float
        slack variable controlling margin of graph-structure loss
    class_margin: float
        slack variable controlling margin of class-boundary loss
    top_k: int
        number of adjacent nodes to aggregate over in message passing step
    activation: torch nn functional 
        activation function to apply after each layer
    negative_slope:
        negative slope of leaky ReLU
    residual:
        use residual connection
    """
    def __init__(self,
                 in_dim,
                 num_classes,
                 num_heads,
                 num_hidden,
                 num_layers,
                 feat_drop,
                 graph_margin,
                 class_margin,
                 top_k,
                 activation=F.leaky_relu,
                 negative_slope=0.2,
                 residual=False,
                 allow_zero_in_degree=True):
        
        super(CGAT, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads[0]
        self.num_out_heads = num_heads[-1]
        self.cgat_layers = nn.ModuleList()
        self.activation = activation
        
        # input projection (no residual)
        self.cgat_layers.append(CGATConv(in_dim,
                                         num_hidden,
                                         self.num_heads,
                                         feat_drop=feat_drop,
                                         graph_margin=graph_margin, 
                                         class_margin=class_margin,
                                         top_k=top_k,
                                         negative_slope=0.2,
                                         residual=False,
                                         activation=activation,
                                         allow_zero_in_degree=allow_zero_in_degree))
        
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.cgat_layers.append(CGATConv(num_hidden*self.num_heads,
                                             num_hidden,
                                             self.num_heads,
                                             feat_drop=feat_drop,
                                             graph_margin=graph_margin, 
                                             class_margin=class_margin,
                                             top_k=top_k,
                                             negative_slope=0.2,
                                             residual=False,
                                             activation=activation,
                                             allow_zero_in_degree=allow_zero_in_degree))
            
        # output projection
        self.cgat_layers.append(CGATConv(num_hidden*self.num_heads,
                                         num_classes,
                                         self.num_out_heads,
                                         feat_drop=feat_drop,
                                         graph_margin=graph_margin, 
                                         class_margin=class_margin,
                                         top_k=top_k,
                                         negative_slope=0.2,
                                         residual=False,
                                         activation=activation,
                                         allow_zero_in_degree=allow_zero_in_degree))

    def forward(self, g, inputs, label):
        
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
        Lg = 0
        Lb = 0
        for l in range(self.num_layers):
            
            h = self.cgat_layers[l](g, h, label)
            
            Lg += self.cgat_layers[l].Lg
            Lb += self.cgat_layers[l].Lb

            h = h.flatten(1)

        # output projection
        logits = self.cgat_layers[-1](g,h,label)
        logits = logits.mean(1)
        
        Lg += self.cgat_layers[-1].Lg
        Lb += self.cgat_layers[-1].Lb

        return logits, Lg, Lb

    def save(self, filename):

        """

        """

        torch.save(self.state_dict(), filename)
