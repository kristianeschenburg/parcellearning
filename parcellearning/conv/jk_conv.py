"""
Code from: https://github.com/mori97/JKNet-dgl/blob/master/modules/graph_conv_layer.py
"""

import dgl.function as fn
import torch

AGGREGATIONS = {
    'sum': torch.sum,
    'mean': torch.mean,
    'max': torch.max,
}


class GraphConvLayer(torch.nn.Module):

    """Graph convolution layer.
    Args:
        in_features (int): Size of each input node.
        out_features (int): Size of each output node.
        aggregation (str): 'sum', 'mean' or 'max'.
                           Specify the way to aggregate the neighbourhoods.
    """


    def __init__(self, in_features, out_features, aggregation='sum'):
        
        """
        Parameters:
        - - - - - 
        in_features: torch, int
            number of input features
        out_features: torch, int
            number of output features
        aggregation: str
            method for aggregating neighborhood signals
        """

        super(GraphConvLayer, self).__init__()

        if aggregation not in AGGREGATIONS.keys():
            raise ValueError("'aggregation' argument has to be one of "
                             "'sum', 'mean' or 'max'.")
        self.aggregate = lambda nodes: AGGREGATIONS[aggregation](nodes, dim=1)

        self.linear = torch.nn.Linear(in_features, out_features)
        self.self_loop_w = torch.nn.Linear(in_features, out_features)
        self.bias = torch.nn.Parameter(torch.zeros(out_features))

    def forward(self, graph, x):

        """
        Forward pass of layer.

        Parameters:
        - - - - -
        graph: DGL Graph
            input graph
        x: torch, tensor
            input features
        """

        graph.ndata['h'] = x
        graph.update_all(
            fn.copy_src(src='h', out='msg'),
            lambda nodes: {'h': self.aggregate(nodes.mailbox['msg'])})
        h = graph.ndata.pop('h')

        # projecting output data AFTER aggregation
        h = self.linear(h)

        # output embedding is the sum of projected aggregated embedding + projected input signal + bias
        return h + self.self_loop_w(x) + self.bias