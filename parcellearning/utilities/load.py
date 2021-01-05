import sys
sys.path.append('../cgat/')
sys.path.append('../gat/')
sys.path.append('../gcn/')
sys.path.append('../gauss/')

import argparse
import json
import torch
import torch.nn.functional as F

from gat import gat
from cgat import cgat
from gcn import gcn
from gauss import gauss

def load_model(schema, model_file):

    """
    Load a model, given the scheme file of model parameters and the model state dictionary
    saved after the model fitting.

    Parameters:
    - - - - -
    schema: dict
        dictionary of parameter values for a given architecture
    model_file: str
        path to learned model parameters

    Returns:
    - - - -
    model: nn.Module
        loaded model
    """

    model_function = {'CGAT': cgat.CGAT,
                      'GAT': gat.GAT,
                      'GAUSS': gauss.GAUSS,
                      'GCN': gcn.GCN}

    model = model_function[schema['model']](**schema['model_parameters'])

    model.load_state_dict(torch.load(model_file)['model_state_dict'])

    return model
    