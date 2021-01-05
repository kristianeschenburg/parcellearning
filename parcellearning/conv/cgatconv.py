"""Torch modules for contrained graph attention networks(GAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import numpy as np

import torch as th
from torch import nn

import dgl

import dgl.function as fn
from dgl.ops import edge_softmax
from dgl.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair

# pylint: enable=W0235
class CGATConv(nn.Module):
    r"""

    Description
    -----------
    Apply `Constrained Graph Attention Network <https://arxiv.org/abs/1910.11945>`__
    over an input signal.

    Parameters
    ----------
    in_feats : int, or pair of ints
        Input feature size
    out_feats : int
        Output feature size
    num_heads : int
        Number of heads in Multi-Head Attention.
    feat_drop : float, optional
        Dropout rate on feature. Defaults: ``0``.
    graph_margin : float, optional
        margin between distant and adjacent attention values
    class_margin : float, optional
        margin betwen same vs. differently label destination node attention values
    top_k : int, optional
        number of attention-weights messages to aggregate over
        aggregates messages of desination nodes with top K attention values
    negative_slope : float, optional
        LeakyReLU angle of negative slope. Defaults: ``0.2``.
    residual : bool, optional
        If True, use residual connection. Defaults: ``False``.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Defaults: ``False``.

    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 graph_margin=0.1, 
                 class_margin=0.1,
                 top_k=3,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=True):
                 
        super(CGATConv, self).__init__()
        
        # define parameters for layer-wise losses
        self._graph_margin = graph_margin
        self._class_margin = class_margin
        self._top_k = top_k
        
        # number of layer heads
        self._num_heads = num_heads
        # number of source features
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        # number of output features
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree

        # fully connected projection layer
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)

        # learned weight vector to compute attentions
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))

        # feature dropout layer
        self.feat_drop = nn.Dropout(feat_drop)

        # non-linearity
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        
        # skip connections layer
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)

        self.reset_parameters()

        # activation function
        self.activation = activation

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
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        
        # initialize attention L/R weights
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)

        # intialize residual weights
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, label):
        r"""

        Description
        -----------
        Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        
        # define functions for compute graph and class losses
        # we internalize these to make use of the graph/class margin attributes of the layer
        def graph_loss(nodes):
            
            """
            Loss function on graph structure.
            
            Enforces high attention to adjacent nodes and 
            lower attention to distant nodes via negative sampling.
            """

            msg = nodes.mailbox['m']

            pw = msg[:, :, :, 0, :].unsqueeze(1)
            nw = msg[:, :, :, 1, :].unsqueeze(2)

            loss = (nw + self._graph_margin - pw).clamp(0)
            loss = loss.sum(1).sum(1)

            return {'graph_loss': loss}

        def adjacency_message(edges):
            
            """
            Compute binary message on edges.

            Compares whether source and destination nodes
            have the same or different labels.
            """

            l_src = edges.src['l']
            l_dst = edges.dst['l']

            if l_src.ndim > 1:
                adj = th.all(l_src == l_dst, dim=1)
            else:
                adj = (l_src == l_dst)

            return {'adj': adj.detach()}


        def class_loss(nodes):
            
            """
            Loss function on class boundaries.
            
            Enforces high attention to adjacent nodes with the same label
            and lower attention to adjacent nodes with different labels.
            """

            m = nodes.mailbox['m']

            w = m[:, :, :-1]
            adj = m[:, :, -1].unsqueeze(-1).bool()

            same_class = w.masked_fill(adj == 0, np.nan).unsqueeze(2)
            diff_class = w.masked_fill(adj == 1, np.nan).unsqueeze(1)

            difference = (diff_class + self._class_margin - same_class).clamp(0)
            loss = th.nansum(th.nansum(difference, 1), 1)

            return {'boundary_loss': loss}
        
        def topk_reduce_func(nodes):
    
            """
            Aggregate attention-weighted messages over the top-K 
            attention-valued destination nodes
            """

            K = self._top_k

            m = nodes.mailbox['m']
            [m,_] = th.sort(m, dim=1, descending=True)
            m = m[:,:K,:,:].sum(1)

            return {'ft': m}

        with graph.local_scope():
        
            # check for zero-degree nodes
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                'output for those nodes will be invalid. '
                                'This is harmful for some applications, '
                                'causing silent performance regression. '
                                'Adding self-loop on the input graph by '
                                'calling `g = dgl.add_self_loop(g)` will resolve '
                                'the issue. Setting ``allow_zero_in_degree`` '
                                'to be `True` when constructing this module will '
                                'suppress the check and let the code run.')

            # if analyzing bipartite data
            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])

                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc

                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)

            else:
                h_src = h_dst = self.feat_drop(feat)

                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)

                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]

            # inner product between learned attention weights and node features
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)

            # compute unnormalized attention values for each source node with destination nodes
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))

            # compute graph structure loss
            Lg = th.tensor(0)
            if self._graph_margin is not None:
                neg_graph = [construct_negative_graph(i, k=1) for i in dgl.unbatch(graph)]
                neg_graph = dgl.batch(neg_graph)

                neg_graph.srcdata.update({'ft': feat_src, 'el': el})
                neg_graph.dstdata.update({'er': er})
                neg_graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
                ne = self.leaky_relu(neg_graph.edata.pop('e'))

                combined = th.stack([e, ne]).transpose(0, 1).transpose(1, 2)
                graph.edata['combined'] = combined
                graph.update_all(fn.copy_e('combined', 'm'), graph_loss)
                Lg = graph.ndata['graph_loss'].sum() / (graph.num_nodes() * self._num_heads)
            
            # compute class boundary loss
            Lb = th.tensor(0)
            if self._class_margin is not None:
                graph.ndata['l'] = label
                graph.apply_edges(adjacency_message)
                adj = graph.edata.pop('adj').float()

                combined = th.cat([e.squeeze(), adj.unsqueeze(-1)], dim=1)
                graph.edata['combined'] = combined
                graph.update_all(fn.copy_e('combined', 'm'), class_loss)
                Lb = graph.ndata['boundary_loss'].sum() / (graph.num_nodes() * self._num_heads)

                # remove edge data to release memory
                graph.edata.pop('combined');
            
            # apply non-linearity to un-normalized attention weights
            graph.edata['a'] = edge_softmax(graph, e)

            # message passing
            if self._top_k is not None:
                graph.update_all(fn.u_mul_e('ft', 'a', 'm'), 
                                topk_reduce_func)
            else:
                graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                                fn.sum('m', 'ft'))
                
            rst = graph.dstdata['ft']

            # apply skip connections
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval

            # apply activation to updated node features
            if self.activation:
                rst = self.activation(rst)

            # retain layer-wise losses
            self.Lg = Lg
            self.Lb = Lb

            # return the updated node features
            return rst 


def construct_negative_graph(graph, k=1):
    """
    Reshuffle the edges of a graph.

    Parameters:
    - - - - -
    graph: DGL graph object
        input graph to reshuffle
    k: int
        number of edge pairs to generate
        if original graph has E edges, new graph will 
        have <= k*E edges
    
    Returns:
    - - - -
    neg_graph: DGL graph object
        reshuffled graph
    """

    src, dst = graph.edges()

    neg_src = src.repeat_interleave(k)
    neg_dst = th.stack([dst[th.randperm(graph.num_edges())]
                        for i in range(k)], dim=0)
    neg_dst = neg_dst.view(-1)

    neg_graph = dgl.graph((neg_src, neg_dst),
                          num_nodes=graph.number_of_nodes())

    return neg_graph
