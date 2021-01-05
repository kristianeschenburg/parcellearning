import torch
import numpy as np

class GraphSampler(object):

    """
    Downsample a dataset to include only certain labels, as mentioned in
    <https://openreview.net/pdf?id=rkKvBAiiz>`__ where the authors examine
    only the parcellations for Brodmann's Area 44 and 45.

    Parameters:
    - - - - - 
    foreground: list, int
        list of labels to include
    background: list, int
        list of labels to act as a buffer between inclusion and exclusion
    """

    def __init__(self, foreground, background):

        self.fg = foreground
        self.bg = background

    def sample(self, G):

        """
        Apply downsampling to list of graphs.

        Parameters:
        - - - - -
        G: list, DGLObject
            list of graphs to downsample
        """

        inclusion = self.fg + self.bg

        for graph in G:

            idx_exclude = torch.where(~torch.tensor(np.isin(graph.ndata['label'], inclusion)))[0]
            graph.remove_nodes(idx_exclude)

            idx_background = torch.tensor(np.isin(graph.ndata['label'], self.bg))

            graph.ndata['label'][idx_background] = 0

        return G


    