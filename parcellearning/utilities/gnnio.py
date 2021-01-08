import dgl
from dgl.data import DGLDataset
from dgl import data
import torch

import os
import nibabel as nb
import numpy as np
import pandas as pd

import fragmenter
import pysurface

import argparse
import json


class GCNData(DGLDataset):

    """
    Class for generating DGL dataset

    Parameters
    ----------
    subject_list: str
        list of individuals to aggregate data over
    data_name: str
        output dataset net
    url : str
        URL to download the raw dataset
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    save_dir : str
        Directory to save the processed dataset.
        Default: the value of `raw_dir`
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information
    """

    def __init__(self,
                 subject_list,
                 data_name,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False,
                 labels = {'dir': '/projects3/parcellation/data/labels/',
                          'extension': '.L.CorticalAreas.fixed.32k_fs_LR.label.gii'},
                 graphs = {'dir': '/projects3/parcellation/data/surfaces/',
                           'extension': '.L.midthickness.32k_fs_LR.acpc_dc.surf.gii'},
                 features={'regionalized': {'dir': '/projects3/parcellation/data/regionalization/Destrieux/',
                                            'extension': '.L.aparc.a2009s.Mean.CrossCorr.csv'}}):

        self.data_name = data_name
        self.feature_map = features
        self.graph_map = graphs
        self.label_map = labels

        # load provided list of training subjects
        with open(subject_list, 'r') as f:
            subjects = f.read().split()

        self.subjects = subjects

        super(GCNData, self).__init__(name='GCNData',
                                       url=url,
                                       raw_dir=raw_dir,
                                       save_dir=save_dir,
                                       force_reload=force_reload,
                                       verbose=verbose)

    def download(self):

        # download raw data to local disk
        pass

    def process(self):

        features = list(self.feature_map.keys())
        features.sort()

        # check which files have all of the training features and surface files
        qc = []
        for subject in self.subjects:

            existence = 0
            # check with features exists
            for f in features:
                filename='%s%s%s' % (self.feature_map[f]['dir'], subject, self.feature_map[f]['extension'])
                if os.path.exists(filename):
                    existence += 1
            
            # check surface file exists
            surface_file = '%s%s%s' % (self.graph_map['dir'], subject, self.graph_map['extension'])
            if os.path.exists(surface_file):
                existence += 1
            
            # check label file exists
            response_file = '%s%s%s' % (self.label_map['dir'], subject, self.label_map['extension'])
            if os.path.exists(response_file):
                existence += 1

            # append passing subject
            if existence == (len(features)+2):
                qc.append(subject)
        
        # reassign list of subjects that passed
        if len(qc) < len(self.subjects):
            print('Warning: provided list of subjects has missing data.')
            print('Only %i of %i subjects were loaded.' % (len(qc), len(self.subjects)))
        else:
            print('Data for all subjects provided exists')

        self.subjects = qc

        graphs = []
        # iterate over subjects
        # load training data, response data, and graph structure
        for subject in self.subjects:
            
            # load node features -- independent variables
            sfeats = {}
            for f in features:
                filename='%s%s%s' % (self.feature_map[f]['dir'], subject, self.feature_map[f]['extension'])

                ext=filename.split('.')[-1]
                if ext == 'csv':
                    df = pd.read_csv(filename, index_col=[0])
                    df = np.asarray(df)
                elif ext == 'gii':
                    df = nb.load(filename)
                    df = df.darrays[0].data

                df = torch.tensor(df)
                if df.ndim == 1:
                    df = df.unsqueeze(-1)

                sfeats[f] = df

            # load surface file -- to generate graph structure
            surface_file = '%s%s%s' % (self.graph_map['dir'], subject, self.graph_map['extension'])
            surface = nb.load(surface_file)
            vertices = surface.darrays[0].data
            faces = surface.darrays[1].data

            # generate graph adjacency structure
            adjacency = pysurface.matrix.adjacency(F=faces)

            # load label file -- dependent variable
            response_file = '%s%s%s' % (self.label_map['dir'], subject, self.label_map['extension'])
            label = nb.load(response_file)
            label = label.darrays[0].data
            label = torch.tensor(label).long()

            gidx = torch.where(label >= 1)[0]

            adjacency = adjacency[gidx,:][:,gidx]

            graph = dgl.from_scipy(adjacency)
            for feature, data in sfeats.items():
                graph.ndata[feature] = data[gidx]
            
            graph.ndata['label'] = label[gidx]
            graph.ndata['idx'] = gidx

            graphs.append(graph)

        self.graph = graphs

    def __getitem__(self, idx):

        # get one example by index
        pass

    def __len__(self):

        # number of data examples
        pass

    def save(self):

        # save processed data to directory `self.save_dir`

        data.save_graphs(g_list=self.graph, filename=self.data_name)

    def load(self):

        # load processed data from directory `self.save_path`

        return data.load_graphs(filename=self.data_name)[0]

    def has_cache(self):

        # check if preprocessed data already exists

        return os.path.exists(self.data_name)

def standardize(dataset):

    """
    Standardize the columns of the feature matrix.

    Parameters:
    - - - - -
    dataset: torch tensor
        array of features to standardize
    """

    dataset = dataset.detach().numpy()

    mean = np.nanmean(dataset, axis=0)
    std = np.nanstd(dataset, axis=0)

    dataset = (dataset - mean) / std
    dataset = torch.tensor(dataset).float()

    return dataset

def dataset(features=None,
            dir='/projects3/parcellation/',
            dSet=None,
            dType='training',
            norm=True,
            clean=True):

    """
    Load datasets that can be plugged in directly to GNN models.

    Parameters:
    - - - - -
    features: list
        list of features to include in the dataset
    dir: str
        base path to data
    dSet: set
        path to previously computed dataset
    norm: bool
        standardize the columnso of the features
    clean: bool
        after aggregation of all features into single matrix
        remove single feature columns
    """

    dName = '%sdata/%s.bin' % (dir, dType)
    subject_list = '%ssubjects/%s.txt' % (dir, dType)

    features_map={'regionalized': {'dir': '%sdata/regionalization/Destrieux/' % (dir),
                               'extension': '.L.aparc.a2009s.Mean.CrossCorr.csv'},
                  'sulcal': {'dir': '%sdata/sulcal/' % (dir), 
                                'extension': '.L.sulc.32k_fs_LR.shape.gii'},
                  'myelin': {'dir': '%sdata/myelin/' % (dir), 
                                'extension': '.L.MyelinMap.32k_fs_LR.func.gii'},
                  'spectral': {'dir': '%sdata/spectra/' % (dir), 
                                'extension': '.L.midthickness.spectrum.LSA.CPD.csv'},
                  'curv': {'dir': '%sdata/curvature/' % (dir),
                                'extension': '.L.curvature.32k_fs_LR.shape.gii'}}

    if features is None:
        features = list(features_map.keys())
        
    features.sort()
    fmap = {k: features_map[k] for k in features}

    # by default, load the whole dataset
    # it will be created if it does not exist 
    if not dSet:
        data_set = GCNData(subject_list=subject_list,
                           data_name=dName,
                           features=fmap)
        data_set = data_set.load()

    # otherwise, load the specified path
    # assumes the feature names are the same as those in the full dataset
    else:
        data_set = dgl.load_graphs(dSet)[0]

    # standardize features
    if norm:
        for graph in data_set:

            indices = graph.ndata['idx'].bool()

            # standardize the specified features
            for feature in fmap.keys():

                temp = graph.ndata[feature]
                # normalize each feature column
                temp = standardize(temp)
                graph.ndata[feature] = temp

    # aggregate the features of interest
    for graph in data_set:
        temp = torch.cat([graph.ndata[f] for f in features], dim=1)
        graph.ndata['features'] = temp

    # remove all individual features apart from the aggregation
    if clean:
        exfeats = [l for l in graph.ndata.keys() if l not in ['features', 'label', 'idx']]
        for graph in data_set:
            for exfeat in exfeats:
                graph.ndata.pop(exfeat)

    # exclude self loop connections
    for graph in data_set:
            graph = dgl.remove_self_loop(graph)

    return data_set
