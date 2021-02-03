import torch
import dgl


def cosine(nodes):
    """
    Compute the cosine distance between all pairs of nodes adjacent on a source node.
    """

    # ```m``` is a matrix of N nodes x E edges x F features
    # representing the messages incident on source nodes with E edges
    m = nodes.mailbox['m']

    N = m.shape[1]
    N = (N*(N-1))/2

    m = m.transpose(1,2)

    e = torch.matmul(m, m.transpose(2, 3))
    e = torch.triu(e, diagonal=1).sum(-1).sum(-1)
    e = e/N

    return {'cos': e}
