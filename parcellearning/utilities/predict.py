import argparse, dgl, json, os, random, subprocess, sys, time

from parcellearning.utilities import gnnio
from parcellearning.utilities.load import load_schema, load_model

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, jaccard_score

import torch

from niio import write
 
def main(args):

    schema = load_schema(args.schema_file)

    ##### GET PARAMETERS FROM SCHEMA FILE #####
    # - - - - - - - - - - - - - - - - - - - - #
    # - - - - - - - - - - - - - - - - - - - - #

    DATA_PARAMS = schema['data']

    MODEL_PARAMS = schema['model_parameters']
    OPT_PARAMS = schema['optimizer_parameters']
    TRAIN_PARAMS = schema['training_parameters']
    STOP_PARAMS = schema['stopping_parameters']

    VAR_PARAMS = schema['variable_parameters']
    
    # - - - - - - - - - - - - - - - - - - - - #
    # - - - - - - - - - - - - - - - - - - - - #

    features = VAR_PARAMS['features']
    features.sort()

    if args.compute_accuracy:
        atlas = VAR_PARAMS['response']
    else:
        atlas = None

    label_table = DATA_PARAMS['label_table']

    if args.masked:
        mask_file='/projects3/parcellation/data/L.Glasser.Mask.csv'
        mask = pd.read_csv(mask_file, index_col=[0])
        mask = np.asarray(mask)

    # - - - - - - - - - - - - - - - - - - - - #
    # - - - - - - - - - - - - - - - - - - - - #

    graphs = gnnio.dataset(features=features,
                           dSet=args.graph,
                           atlas=atlas,
                           norm=True,
                           clean=True)

    # get model file
    model_parameters = '%s%s.earlystop.Loss.pt' % (DATA_PARAMS['out'], schema['model'])
    model = load_model(schema, model_parameters)
 
    for i, graph in enumerate(graphs):
        model.eval()

        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)

        prediction = np.zeros((32492,))

        with torch.no_grad():

            test_X = graph.ndata['features']
            idx = graph.ndata['idx']

            if args.compute_accuracy:
                test_Y = graph.ndata['label']
                background = (test_Y<=0)

            if schema['model'] in ['JKGAT']:

                return_alpha=True
                if schema['model_parameters']['aggregation'] in ['cat']:
                    return_alpha=False
                
                # return attention coefficients
                if return_alpha:
                    test_logits, alpha = model(**{'g': graph, 
                                                'inputs': test_X, 
                                                'return_alpha': return_alpha})
                else:
                    test_logits = model(**{'g': graph,
                                           'inputs': test_X,
                                           'return_alpha': return_alpha})


            elif schema['model'] in ['GAT', 'GCN']:
                test_logits = model(**{'g': graph, 'inputs': test_X})

            elif schema['model'] in ['MLP']:
                test_logits = model(**{'inputs': test_X})

            if args.masked:
                test_logits = test_logits - mask[idx]
            
            _,indices = torch.max(test_logits, dim=1)

            prediction[idx] = indices

            probabilities = np.zeros((prediction.shape[0], test_logits.shape[1]))
            probabilities[idx,:] = torch.softmax(test_logits, axis=1)
            output_prob_file = '%s.%s.Probabilities.func.gii' % (args.output, schema['model'])
            write.save(probabilities, output_prob_file, 'L')

            if args.compute_accuracy:

                correct = torch.nansum(indices[~background] == test_Y[~background])
                acc = correct.item() * 1.0 / len(test_Y[~background])

                f1 = f1_score(indices[~background], 
                              test_Y[~background], 
                              labels=np.arange(1,181), 
                              average='micro')

                jaccard = jaccard_score(indices[~background], 
                                        test_Y[~background], 
                                        labels=np.arange(1,180), 
                                        average='micro')

                df = pd.DataFrame(np.asarray([acc, f1, jaccard])[None,:], columns=['acc', 'dice', 'jaccard'])
                df_file = '%s.%s.metrics.csv' % (args.output, schema['model'])
                df.to_csv(df_file)

                print('Accuracy: %.2f' % (100*acc))

            # save output predictions as metric file
            out_func_file = '%s.%s.prediction.func.gii' % (
                args.output, schema['model'])
            write.save(prediction, out_func_file, 'L')

            # convert metric file to label file
            out_label_file = '%s.%s.label.gii' % (
                args.output, schema['model'])

            bashCommand = "wb_command -metric-label-import  %s %s %s" % (
                            out_func_file, label_table, out_label_file)
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()

            # remove func file
            os.remove(out_func_file)

            if schema['model'] in ['JKGAT']:
                if return_alpha:
                    attention = np.zeros((32492,alpha.shape[1]))
                    attention[idx,:] = alpha
                    out_func_file = '%s.%s.attention.func.gii' % (
                        args.output, schema['model'])
                    write.save(attention, out_func_file, 'L')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Perform model testing.')
    parser.add_argument('--schema-file', 
                        type=str,
                        help='JSON file with parameters for model, training, and output.')

    parser.add_argument('--graph', 
                        type=str, 
                        help='Input graph file.',
                        required=True)

    parser.add_argument('--output', 
                        type=str, 
                        help='Output base name.',
                        required=True)
    
    parser.add_argument('--masked',
                        action='store_true',
                        help='Mask output logits using probability assignments of cortex.',
                        required=False)

    parser.add_argument('--compute-accuracy',
                        action='store_true',
                        help='Compute accuracy of prediction against known map.',
                        required=False,
                        default=False)

    args = parser.parse_args()

    main(args)
