import argparse, json, os
import dgl
import torch
import networkx as nx
import numpy as np
import pandas as pd

from niio import loaded
import pysurface

def main(args):

    with open(args.feature_file, 'r') as f:
        schema = json.load(f)

    # assert not os.path.exists(schema['output']), '%s already exists.' % (schema['output'])

    [verts, faces] = loaded.loadSurf(schema['surface'])
    Surface = pysurface.adjacency.SurfaceAdjacency(faces)
    Surface.generate()

    # generate surface adjacency list
    graph = Surface.adj
    # convert list to networkx graph
    graph = nx.from_dict_of_lists(graph)
    # convert networkx graph to dgl graph
    graph = dgl.from_networkx(graph)

    wall = loaded.load(schema['medial_wall'])
    wall = np.where(wall)[0]

    if args.dType in ['training', 'validation']:

        assert 'label' in schema.keys(), 'If training / validation data, must provide label file: %s' % (schema['label'])

        label = loaded.load(schema['label'])
        label = torch.tensor(label).long()
        graph.ndata['label'] = label

    features = []
    for feature, feature_file in schema['features'].items():

        assert os.path.exists(feature_file), '%s does not exist.' % (feature_file)

        if os.path.splitext(feature_file)[1] == '.csv':
            feature_data = pd.read_csv(feature_file, index_col=[0])
        elif os.path.splitext(feature_file)[1] in ['.gii', '.mat']:
            feature_data = loaded.load(feature_file)
        
        feature_data = np.asarray(feature_data)
        feature_data = torch.tensor(feature_data)
        graph.ndata[feature] = feature_data
    
    graph.ndata['idx'] = torch.tensor(np.arange(32492))

    medial_wall = loaded.load(schema['medial_wall'])
    medial_wall = np.where(medial_wall)[0]
    graph.remove_nodes(medial_wall)

    dgl.save_graphs(filename=schema['output'], g_list=graph)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-f',
                        '--feature-file',
                        help='JSON file with paths to feature files, surface file, label file, and output name.',
                        required=True,
                        type=str)
    
    parser.add_argument('--dType',
                        help='Indicate whether graph is training, validation, or testing.',
                        default='training',
                        required=False,
                        choices=['training', 'validation', 'testing'])
    
    args = parser.parse_args()

    main(args)