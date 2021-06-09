from parcellearning.conv.gatconv import GATConv
import numpy as np

import dgl
from dgl import data
from dgl.data import DGLDataset
import dgl.function as fn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, LSTM
from dgl.nn.pytorch import GraphConv


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
        options: ['lstm', 'concat']
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
                 lstm_layers=2,
                 aggregation='cat'):
        
        super(JKGAT, self).__init__()
        
        assert aggregation in ['cat', 'lstm'], 'Aggregation must be ```cat``` or ```lstm```'
        
        self.num_layers = num_layers
        self.num_heads = num_heads[0]
        self.num_out_heads = num_heads[-1]
        self.layers = nn.ModuleList()
        self.activation = activation
        self.lstm_layers=lstm_layers
        self.aggregation = aggregation

        print('Number of LSTM layers: %s' % (self.lstm_layers))
        
        # input projection (no residual)
        self.layers.append(GATConv(
            in_dim, num_hidden, self.num_heads,
            feat_drop, attn_drop, negative_slope, False, self.activation, allow_zero_in_degree))
        
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.layers.append(GATConv(
                num_hidden * self.num_heads, num_hidden, self.num_heads,
                feat_drop, attn_drop, negative_slope, residual, self.activation, allow_zero_in_degree))
            
        # Jumping Knowledge Layer
        if aggregation == 'cat':
            self.output = Linear(self.num_heads * num_hidden * num_layers, num_classes, bias=False)
            self.layers.append(self.output)

        elif aggregation == 'lstm':
            # bidirectional LSTM concats the forward and backward embeddings
            # so final output will be of size 2 * `hidden_size`
            lstm_layers = self.lstm_layers

            self.lstm = LSTM(input_size=num_hidden * self.num_heads, 
                             hidden_size=num_hidden, 
                             num_layers = lstm_layers,
                             batch_first=True, 
                             bidirectional=True)

            self.attn = Linear(2*num_hidden, 1)
            self.output = Linear(num_hidden*self.num_heads, num_classes, bias=False)

            self.layers.append(self.lstm)
            self.layers.append(self.attn)
            self.layers.append(self.output)   

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

        if hasattr(self, 'lstm'):
            self.lstm.reset_parameters()
        if hasattr(self, 'att'):
            self.att.reset_parameters()
        if hasattr(self, 'linear'):
            self.output.reset_parameters()

    def forward(self, g=None, inputs=None, return_alpha=False, **kwds):
        
        """
        Forward pass of network.  

        Parameters:
        - - - - -
        g: DGL Graph
            the graph
        inputs: tensor
            node features
        return_alpha: bool
            only applies if aggregation == 'lstm'
            returns learned attentions for each node
            
        Returns:
        - - - - -
        logits: tensor
            output layer 
        """

        h = inputs
        xs = []
        for l in range(self.num_layers):

            h = self.layers[l](g,h).flatten(1)
            xs.append(h.unsqueeze(-1))

        # LSTM aggregator
        if self.aggregation == 'lstm':

            # input to lstm
            # xs shape will be shape (nodes x seq_length x features)
            xs = torch.cat(xs, dim=-1).transpose(1,2)

            # compute attentions
            alpha,_ = self.lstm(xs)
            alpha = self.attn(alpha).squeeze(-1)
            alpha = torch.softmax(alpha, dim=-1)

            # compute final embeddings
            h = (xs * alpha.unsqueeze(-1)).sum(1)
            h = self.output(h)

        # CONCAT aggregator
        elif self.aggregation == 'cat':
            h = torch.cat(xs, dim=1).squeeze()
            h = self.output(h)

        # apply sigmoid activation to jumping-knowledge output
        # logits = torch.sigmoid(h)

        if return_alpha:
            return h, alpha
        else:
            return h

    def save(self, filename):

        """
        Save learned model to disk.
        
        Parameters:
        - - - - -
        filename: str
            model name
        """

        torch.save(self.state_dict(), filename)
