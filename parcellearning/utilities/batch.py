import dgl
import numpy as np
import random

def partition_graphs(in_graphs, n=1):

        """
        Function to partition the training data into batches.

        Parameters:
        - - - - -
        in_graphs: list, DGLGraphs
            list of subject-level training graphs
        n: int
            number of batches to return
        """

        if n is not None:
            assert n >= 1
    
        random.shuffle(in_graphs)
        batches = [dgl.batch(in_graphs[i::n]) for i in range(n)]

        return batches