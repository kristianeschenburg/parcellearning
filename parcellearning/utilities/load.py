from ..cgat import cgat
from ..gat import gat
from ..gauss import gauss
from ..gcn import gcn

import argparse
import json
import torch
import torch.nn.functional as F

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


def load_schema(schema_file):
    
    """
    
    """

    with open(schema_file, 'r') as f:
        parameters = json.load(f)
    
    for param in ['model_parameters', 'loss_parameters', 'optimizer_parameters']:
        if param in parameters:
            for k,v in parameters[param].items():
                if v == 'True':
                    parameters[param][k] = True
                elif v == 'False':
                    parameters[param][k] = False
                if k == 'activation':
                    parameters[param][k] = eval(v)

    return parameters
    