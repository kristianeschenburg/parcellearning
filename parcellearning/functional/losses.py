import numpy as np

import dgl
import dgl.function as fn
import torch.nn.functional as F

import torch as th

class StructuredCrossEntropy(object):

    """
    Compute a structured cross-entropy loss.
    
    Loss penalizes high logit probabilities assigned to labels
    that are not directly adjacent to the true label.
    """

    def __init__(self):

        """
        """
    
    def __call__(self, graph, logits, target):

        """
        
        Parameters:
        - - - - -
        graph: DGL graph
            input graph structure
        input: torch tensor
            logits from model
        target: torch tensor
            true node labeling
        structured_weight: torch tensor, float
            weight to assign to incorrect labels
        Returns:
        - - - -
        loss: torch tensor
            structured cross-entropy loss
        """
        
        # compute one-hot encoding
        hot_encoding = F.one_hot(target).float()
        
        # identify adjacent labels
        weight = th.matmul(hot_encoding.t(), 
                                th.matmul(graph.adjacency_matrix(), hot_encoding))
        weight = (1-(weight>0).float())


        # compute inverted encoding (non-adjacent labels receive value of 1)
        inv_encoding = weight[target]
         # weight by 1/(# non adjacent)
        sums = inv_encoding.sum(1)
        print('# of zero-sums in inverse encoding: %i' % (sums == 0).sum())
        inv_encoding = inv_encoding / sums.unsqueeze(1)
        print('# infs in scaled inv-encoding: %i' % th.isinf(inv_encoding).sum())
        print('# nans in inv-encoding: %i' % th.isnan(inv_encoding).sum())
        loss = th.sum(inv_encoding*th.log(1-F.softmax(logits, dim=1)), 1)

        return -loss.mean()

class NormalizedCut(object):

    """
    Parameters:
    - - - - - - - - -
    graph: DGL graph
        graph structure of data
    logits: torch tensor, float
        output of network
    relaxed: bool
        use a discrete or continuous (relaxed) formulation of normalized cuts
        If continuous, use softmax probabilities instead of hot encoding to 
        compute loss

    Returns:
    - - - - 
    loss: torch float tensor
        ncut loss value 
    """

    def __init__(self, relaxed=False):

        """

        """

        self.relaxed = relaxed
    
    def __call__(self, graph, logits):

        """
        
        """

        A = graph.adjacency_matrix()
        d = th.sparse.sum(A, dim=0)

        if not self.relaxed:
            # compute maximum-probability class for each node
            max_idx = th.argmax(F.softmax(logits), 1, keepdim=True)

            # initialize one-hot encoding matrix
            encoding = th.FloatTensor(logits.shape)
            encoding.zero_()
            encoding.scatter_(0, max_idx, 1)
        else:
            encoding = F.softmax(logits)

        assoc = th.matmul(encoding.t(), th.matmul(A, encoding))
        degree = th.matmul(d, encoding)

        loss = th.nansum(assoc.diag() / degree)

        return -loss

class GraphStructure(object):

    """
    Enforce the constraint that neighboring nodes have more similar 
    embedding than nodes that are far apart from one another.  This can 
    be viewed as a similar loss that used in `<https://arxiv.org/abs/1706.02216>`__

    .. math:

        \mathit{L_{g}} = \sum_{i \in V} \sum_{j \in \mathcal{N_{i}}} \sum_{k \in V \setminus \mathcal{N_{i}}} max(0, \phi(v_{i},v_{k}) + \\xi_{g} - \phi(v_{i},v_{j}))


    Loss computes a single negative sample (reshuffles edges only once).

    """

    def __init__(self, margin=0.1):

        """
        Parameters:
        - - - - -
        margin: float
            slack variable controlling margin between positive and negative samples
        """

        self.margin = margin
    
    def __call__(self, G, embedding):

        """
        Parameters:
        - - - - -
        G: DGL graph
            graph on which loss is computed
        embedding: torch tensor
            node features to use for loss
        """

        with G.local_scope():
            embedding = embedding / th.norm(embedding, dim=1, p=2).unsqueeze(1)

            G.ndata['h'] = embedding
            G.apply_edges(fn.u_dot_v('h', 'h', 'pos'))

            neg_graph = construct_negative_graph(G, k=1)
            neg_graph.ndata['h'] = embedding
            neg_graph.apply_edges(fn.u_dot_v('h', 'h', 'neg'))

            D = neg_graph.edata['neg'] + self.margin - G.edata['pos']
            Z = th.repeat_interleave(th.tensor(0), G.num_edges()).unsqueeze(1)

            # only those vertices whose embedding is more similar to distant embeddings
            # that to local embeddings contribute to the loss function
            input_max,_ = th.max(th.hstack([Z, D]), dim=1)

            return input_max.sum() / G.num_nodes()


class ClassBoundary(object):
    
    """
    Enforce the constraint that neighboring nodes that have the same label 
    have more similar embedding than neighboring nodes that have different labels.  

    .. math:

        \mathit{L_{b}} = \sum_{i \in V} \sum_{j \in \mathcal{N_{i}^{+}}} \sum_{k \in\mathcal{N_{i}^{-}}} max (0, \phi(v_{i},v_{k})) + \\xi_{b} - \phi(v_{i},v_{j})

    Loss computes a single negative sample (reshuffles edges only once).

    """

    def __init__(self, margin=0.1):
        """
        Parameters:
        - - - - -
        margin: float
            slack variable controlling margin between positive and negative samples
        """

        self.margin = margin
    
    def __call__(self, G, embedding, label):

        """
        """

        with G.local_scope():

            embedding = embedding / th.norm(embedding, dim=1, p=2).unsqueeze(1)
            G.ndata['h'] = embedding
            G.ndata['l'] = label

            G.apply_edges(fn.u_sub_v('l', 'l', 'class'))
            G.apply_edges(fn.u_dot_v('h', 'h', 'score'))  

            same_class = (G.edata['class'] == 0).float().unsqueeze(1)
            diff_class = (G.edata['class'] != 0).float().unsqueeze(1)
            
            # isolate edge weights between nodes w/ same class
            G.edata['within_class'] = G.edata['score']*same_class
            # isolate edge weights between nodes w/ different classes
            G.edata['between_class'] = G.edata['score']*diff_class

            # reduce to within-class vs. between-class similarities
            G.update_all(fn.copy_e('within_class', 'wc'), fn.sum('wc', 'wc'))
            G.update_all(fn.copy_e('between_class', 'bc'), fn.sum('bc', 'bc'))

            D = G.ndata['bc'] + self.margin - G.ndata['wc']
            Z = th.repeat_interleave(th.tensor(0), G.num_nodes()).unsqueeze(1)

            input_max, _ = th.max(th.hstack([Z, D]), dim=1)
            
            return input_max.sum() / G.num_nodes()


class AdjacentLabelLoss(object):

    """
    Penalize edges between nodes if their labels assigned to those nodes
    are not adjacent in the training dataset.

    Requires that the graph 
    """

    def __init__(self):
        """
        """

    def __call__(self, G, embedding, label, adjacency):

        with G.local_scope():

            embedding = embedding / th.norm(embedding, dim=1, p=2).unsqueeze(1)
            
            G.ndata['l_adj'] = adjacency[label.long()]
            G.ndata['h'] = embedding
            G.ndata['l'] = label.float()

            G.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            G.apply_edges(msg_func)
            
            adj = G.edata['adj'].squeeze()
            score = G.edata['score'].squeeze()

            loss = th.dot(adj, score)

            return loss / G.num_nodes()

def msg_func(edges):

    h_src = edges.src['h']
    h_dst = edges.dst['h']

    l_dst = edges.dst['l']

    l_asrc = edges.src['l_adj']
    
    # adj is a vector, indicating if the edge is between two nodes with the same label
    # index = 1 if labels are the same, index = 0 if labels are different
    adj = 1-th.gather(l_asrc, dim=1, index=l_dst.long().unsqueeze(1))
    
    return {'adj': adj.float()}
    

class Sage(object):

    """
    Compute the loss function as described in `<https://arxiv.org/abs/1706.02216>`__
    for a given embedding computed on a graph.

    .. math:

        \mathit{J}_{\mathcal{G}}(z_{u}) = -log(\sigma(z_{u}^{T}z_{v})) - Q \cdot \mathbb{E}_{v_{n}} \sim \mathit{P_{n}}(v) log(\sigma(-z_{u}^{T}z_{v_{n}}))

    Parameters:
    - - - - - 
    graph: DGL graph object
        input data graph
    embedding: torch tensor
        embedding on graph
    node_samples: int
        number of nodes to sample to compute the positive sampling score
    length: int
        random walk length for positive sampling score
    size_neg_graph: int
        number of random permutations of graph to generate
    
    Returns:
    - - - - 
    loss: float tensor 
        current loss of embedding
    """

    def __init__(self, length=3, size_neg_graph=1):

        """
        Parameters:
        - - - - - 
        length: int
            size of random walk from source node
            larger number will explore a larger neighborhood around the source node
            a value of length=1 only explores the adjacent and self nodes
        size_neg_graph: int
            number of edge repeats to include
            a value of 1 simply shuffles the existing edges
            a value of 2 repeats the edges and shuffles them 2...
        """

        self.length = length
        self.size_neg_graph = size_neg_graph

    def __call__(self, graph, embeddings, node_samples=1):

        """
        Parameters:
        - - - - -
        graph: DGL graph 
            graph to generate loss from
        embedding: torch tensor
            node features to use to compute the loss value
        node_samples: float [0,1]
            fraction of source nodes to use for computing positive loss
        """

        length = self.length
        size_neg_graph = self.size_neg_graph

        if node_samples is not None:
            # compute positive sampling scores
            node_samples = int(np.floor(graph.num_nodes*node_samples))
            node_ids = np.random.choice(
                graph.nodes(), size=node_samples, replace=False)
        else:
            node_ids = graph.nodes()

        pos_ids = dgl.sampling.random_walk(
            graph, node_ids, length=length)[0][0, -1]
        pos_score = th.sum(embeddings[node_ids] * embeddings[pos_ids], dim=1)
        pos_score = F.logsigmoid(pos_score)  # log sigmoid

        # compute negative sampling score
        neg_graph = construct_negative_graph(graph, size_neg_graph)
        neg_graph = dgl.to_homogeneous(neg_graph)
        neg_graph.ndata['h'] = embeddings
        neg_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))

        # return loss
        loss = -th.sum(pos_score) - th.sum(F.logsigmoid(-neg_graph.edata['score']))

        return loss / graph.number_of_nodes()

def construct_negative_graph(graph, k=1, ndata=None, edata=None):

    src, dst = graph.edges()

    neg_src = src.repeat_interleave(k)
    neg_dst = th.randint(0, graph.number_of_nodes(), (len(src) * k,))
    neg_graph = dgl.graph((neg_src, neg_dst), num_nodes=graph.number_of_nodes())
    
    if ndata is not None:
        neg_graph.ndata[ndata] = graph.ndata[ndata]
    
    if edata is not None:
        neg_graph.edata[edata] = graph.edata[edata]

    return neg_graph
