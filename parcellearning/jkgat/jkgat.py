from parcellearning.conv.gatconv import GATConv
import numpy as np

import dgl
from dgl import data
from dgl.data import DGLDataset
import dgl.function as fn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear


class JKGAT(nn.Module):
    
    """
    Instantiate a Graph Attention Network using Jumping Knowledge learning represenations.
    
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
    aggregation: str
        aggregation strategy for jumping knowledge learning
        options: ['pool', 'concat']
    learn_structure: bool
        learn network structure of jumping-knowledge layer
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
                 allow_zero_in_degree=True,
                 aggregation='concat',
                 return_attention=False):
        
        super(JKGAT, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads[0]
        self.num_out_heads = num_heads[-1]
        self.jkgat_layers = nn.ModuleList()
        self.activation = activation
        self.aggregation = aggregation
        self.return_attention = return_attention
        
        # input projection (no residual)
        self.jkgat_layers.append(GATConv(
            in_dim, num_hidden, self.num_heads,
            feat_drop, attn_drop, negative_slope, False, self.activation, allow_zero_in_degree, return_attention))
        
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.jkgat_layers.append(GATConv(
                num_hidden * self.num_heads, num_hidden, self.num_heads,
                feat_drop, attn_drop, negative_slope, residual, self.activation, allow_zero_in_degree, return_attention))
            
        # Jumping Knowledge Layer
        self.fc_proj = torch.nn.Linear(num_hidden*self.num_heads, num_classes, bias=False)
        self.jkgat_layers.append(self.fc_proj)

        # initialize model weights
        self.reset_parameters()

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """

        gain = nn.init.calculate_gain('relu')
        # initialize fully connected weights 
        if hasattr(self, 'fc_proj'):
            nn.init.uniform_(self.fc_proj.weight)

    def forward(self, g=None, inputs=None, **kwds):
        
        """
        Forward pass of network.  We perform jumping knowledge learning by aggregating over
        the embeddings of each layer.  Options include max-pooling and concatenation.

        Max-pooling learns a unique aggreation for each node, while concatenation learns
        a unique aggregation for the entire graph.

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
        embeddings = []
        A = []
        for l in range(self.num_layers):

            if self.return_attention:
                h,attn = self.jkgat_layers[l](g, h)
                A.append(attn)
            else:
                h = self.jkgat_layers[l](g,h)
            
            embeddings.append(h.sum(1))
            h = h.flatten(1)

        # jumping knowledge using concatenation
        if self.aggregation == 'concat':
            embeddings = torch.cat(embeddings, dim=1)
            
        # jumping knowledge using maxpooling
        elif self.aggregation == 'pool':
            embeddings = torch.stack(embeddings, dim=0)
            embeddings = torch.max(embeddings, dim=0)[0]

        # output projection followed by activation
        h = self.jkgat_layers[-1](embeddings)
        logits = torch.sigmoid(h)
        
        if self.return_attention:
            return logits, A
        else:
            return logits, 

    def save(self, filename):

        """

        """

        torch.save(self.state_dict(), filename)
