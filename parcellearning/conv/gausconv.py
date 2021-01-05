"""Torch modules for gaussian kernel graph convolutional networks."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn

from dgl import function as fn
from dgl.ops import edge_softmax
from dgl.base import DGLError
from dgl.utils import expand_as_pair

# pylint: enable=W0235
class GAUSConv(nn.Module):

    r"""

    Description
    -----------

    Apply a `Gaussian Kernel Graph Convolutional Network <http://arxiv.org/abs/1803.10336>`__
    over an input signal distributed over a graph.

    .. math::

        z^{l}_{v,p} = \sum_{j \in N_{v}} \sum^{M_{l}}_{q=1}\sum_{k=1}^{K_{l}} w_{p,q,k}^{(l)} \times y_{j,q}^{(l)} \times \phi(\mu_{v}, \mu_{u} ; \hat{\mu}_{k}, \hat{\sigma}_{k}) + b_{p}^{(l)}

    where :math:`\phi(\mu_{v}, \mu_{u} ; \hat{\mu}_{k}, \hat{\sigma}_{k})` is the kernel weight

    .. math::
        \phi(\mu_{v}, \mu_{u} ; \hat{\mu}_{k}, \hat{\sigma}_{k}) = exp^{-\hat{\sigma}_{k} ||(\mu_{v} - \mu_{u})-\hat{\mu}_{k}||^{2})}
    
    Parameters
    ----------
    in_feats : int, or pair of ints
        Input feature size; i.e, the number of dimensions of :math:`z_i^{(l)}`.
    out_feats : int
        Output feature size; i.e, the number of dimensions of :math:`z_i^{(l+1)}`.
    num_kernels : int
        Number of kernels to convolve over
    feat_drop : float, optional
        Dropout rate on feature. Defaults: ``0``.
    krnl_drop : float, optional
        Dropout rate on kernel weight. Defaults: ``0``.
    negative_slope : float, optional
        LeakyReLU angle of negative slope. Defaults: ``0.2``.
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
                 num_kernels,
                 feat_drop=0.,
                 krnl_drop=0.,
                 negative_slope=0.2,
                 activation=None,
                 random_seed=None,
                 allow_zero_in_degree=False):

        # set random seed
        if random_seed:
            th.manual_seed(random_seed)
                 
        super(GAUSConv, self).__init__()

        self._num_kernels = num_kernels
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree

        # instantiate typical layer parameters
        # feature weights and bias vector
        self.bias = nn.Parameter(
            th.Tensor(1, num_kernels, out_feats), requires_grad=True)

        # instantiate kernel perameters
        # mean vectors and isotropic sigmas
        self.mu = nn.Parameter(
            th.Tensor(1, num_kernels, out_feats), requires_grad=True)
        self.sigma = nn.Parameter(
            th.Tensor(num_kernels, 1), requires_grad=True)

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats*num_kernels, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats*num_kernels, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats*num_kernels, bias=False)

        # Dropout layers
        # apply dropout to both features and kernels
        self.feat_drop = nn.Dropout(feat_drop)
        self.krnl_drop = nn.Dropout(krnl_drop)

        # instantiate non-linearity function
        self.leaky_relu = nn.LeakyReLU(negative_slope)

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

        nn.init.zeros_(self.bias)
        nn.init.uniform_(self.sigma, a=0.0, b=1.0, )
        nn.init.xavier_uniform_(self.mu)
        

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

    def forward(self, graph, feat):
        r"""

        Description
        -----------
        Compute gaussian kernel network layer.

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

        def gaussian_message(edges):

            K = edges.src['ft'] - edges.dst['ft']
            K = th.norm((K-self.mu), dim=2, p='fro', keepdim=True)
            K = (-1*K*self.sigma).exp()

            return {'kernel': K}

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

                feat_src = self.fc_src(
                    h_src).view(-1, self._num_kernels, self._out_feats)
                feat_dst = self.fc_dst(
                    h_dst).view(-1, self._num_kernels, self._out_feats)

            else:
                h_src = h_dst = self.feat_drop(feat)

                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_kernels, self._out_feats)

                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]

            graph.srcdata.update({'ft': feat_src})

            # apply gaussian kernel to pairwise edges
            graph.apply_edges(gaussian_message)
            kernel = graph.edata['kernel']

            # apply softmax normalization to kernel weights
            # apply kernel dropout to kernel weights
            graph.edata['a'] = self.krnl_drop(edge_softmax(graph, kernel))

            # message passing step
            # aggregate features
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                                fn.sum('m', 'ft'))
                
            rst = graph.dstdata['ft']

            # add bias to layer output
            rst = rst + self.bias

            # apply activation function if desired
            if self.activation:
                rst = self.activation(rst)

            # return the updated source node features
            return rst
