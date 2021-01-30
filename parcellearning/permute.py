import argparse, json, os, random, subprocess, sys, time

from parcellearning.utilities import gnnio
from parcellearning.utilities.load import load_schema, load_model

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import torch

from niio import write

def main(args):

    schema = load_schema(args.schema_file)

    if args.no_background:
        print('Excluding background in accuracy calculations')
        pred_dir = '%s/predictions/no_background/' % (schema['data']['out'])
    else:
        pred_dir = '%s/predictions/' % (schema['data']['out'])

    Path(pred_dir).mkdir(parents=True, exist_ok=True)

    features = schema['features']
    features.sort()

    assess_subjects = '%ssubjects/%s.txt' % (schema['data']['in'], args.data)
    with open(assess_subjects, 'r') as f:
        subjects = f.read().split()

    label_table = '%sL.labeltable' % (schema['data']['in'])

    graphs = gnnio.dataset(dType=args.data,
                           features=features,
                           dSet=schema['data'][args.data],
                           norm=True,
                           aggregate=True,
                           clean=True)

    # get model file
    model_parameters = '%s%s.earlystop.Loss.pt' % (schema['data']['out'], schema['model'])
    model = load_model(schema, model_parameters)

    A = []
    F = []
    S = []

    for i, graph in enumerate(graphs):
        model.eval()

        with torch.no_grad():

            test_X = graph.ndata['features']
            test_Y = graph.ndata['label']
            idx = graph.ndata['idx']

            if args.no_background:

                    background = (test_Y == 0)
                    
                    test_Y = test_Y[~background]
                    test_X = test_X[~background]
                    idx = idx[~background]

            for perm in range(args.permutations):

                test_logits = model(**{'g': graph, 
                                       'inputs': test_X[torch.randperm(test_X.size()[0])], 
                                       'label': test_Y})
                _,indices = torch.max(test_logits, dim=1)

                correct = torch.sum(indices == test_Y)

                # compute accuracy and f1-score
                acc = correct.item() * 1.0 / len(test_Y)
                f = f1_score(test_Y, indices, labels=np.arange(1,181), average='micro')
                
                # store accuracy and F1 metrics
                A.append(acc)
                F.append(f)
                S.append(subjects[i])

    print('Mean accuracy: %.3f' % np.nanmean(accuracies))

    # plot test performance metrics
    df = pd.DataFrame(np.column_stack([A, F, S]), columns=['acc', 'f1', 'subject'])
    df_file = '%s%s.metrics.%s.permutations.csv' % (pred_dir, schema['model'], args.data)
    df.to_csv(df_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Perform permutation model testing on input data.  For each graph in the input data, we reshuffle the feature rows ```permutations``` time and compute the model accuracy.',
                                     usage='use "%(prog)s --help" for more information')
                                     
    parser.add_argument('--schema-file', 
                        type=str,
                        help='JSON file with parameters for model, training, and output.')

    parser.add_argument('--data', 
                        type=str, 
                        help='Assess model performance on test or validation set.',
                        default='testing', 
                        choices=['testing', 'validation'],
                        required=False)
    
    parser.add_argument('--permutations', 
                        type=int, 
                        help='Number of permutations to generate random data.',
                        default=1,
                        required=False)
                        
    parser.add_argument('-no_background', 
                        help='Exclude background voxels in accuracy calculation.',
                        action='store_true',
                        required=False)
    args = parser.parse_args()

    main(args)
