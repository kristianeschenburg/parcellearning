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

    accuracies = np.zeros((len(graphs),))
    F = np.zeros((len(graphs),))

    predictions = np.zeros((32492, len(graphs)))

    for i, graph in enumerate(graphs):
        model.eval()

        with torch.no_grad():

            test_X = graph.ndata['features']
            test_Y = graph.ndata['label']
            idx = graph.ndata['idx']

            P = np.zeros((32492, 2))*np.nan

            if args.alpha:
                test_logits, alpha = model(**{'g': graph, 'inputs': test_X, 'label': test_Y, 'return_alpha': args.alpha})
                A = np.zeros((32492, alpha.shape[1]))

            else:
                test_logits = model(**{'g': graph, 'inputs': test_X, 'label': test_Y})

            _,indices = torch.max(test_logits, dim=1)

            if args.no_background:

                background = (test_Y == 0)
                indices = indices[~background]
                test_Y = test_Y[~background]
                idx = idx[~background]

                if args.alpha:
                    A[indices] = alpha[~background]
            else:
                A[indices] = alpha
            
            correct = torch.sum(indices == test_Y)

            # compute accuracy and f1-score
            acc = correct.item() * 1.0 / len(test_Y)
            f = f1_score(test_Y, indices, labels=np.arange(1,181), average='micro')
            
            # store accuracy and F1 metrics
            accuracies[i] = acc
            F[i] = f

            # update buffer label to have value of -1
            indices[indices == 0] = -1
            test_Y[test_Y == 0] = -1
            # save prediction and ground truth to single file
            P[idx, 0] = indices
            P[idx, 1] = test_Y
            
            # save output predictions as metric file
            out_func_file = '%s%s.L.%s.func.gii' % (
                pred_dir, subjects[i], schema['model'])
            write.save(P, out_func_file, 'L')

            # convert metric file to label file
            out_label_file = '%s%s.L.%s.label.gii' % (
                pred_dir, subjects[i], schema['model'])

            bashCommand = "wb_command -metric-label-import  %s %s %s" % (
                            out_func_file, label_table, out_label_file)
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()

            # save learned layer-wise attentions as metric file
            out_attn_file = '%s%s.L.%s.Attention.func.gii' % (
                pred_dir, subjects[i], schema['model'])
            write.save(A, out_attn_file, 'L')

            # remove func file
            os.remove(out_func_file)

            # store predicted maps
            predictions[idx, i] = indices

    print('Mean accuracy: %.3f' % accuracies.mean())

    # save all predicted maps into single file
    out_func_file = '%s%s.predictions.%s.func.gii' % (pred_dir, schema['model'], args.data)
    write.save(predictions, out_func_file, 'L')

    out_label_file = '%s%s.predictions.%s.label.gii' % (pred_dir, schema['model'], args.data)
    bashCommand = "wb_command -metric-label-import  %s %s %s" % (
                    out_func_file, label_table, out_label_file)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    os.remove(out_func_file)


    # save consensus map
    axis = 1
    u, idx = np.unique(predictions, return_inverse=True)
    consensus = u[np.argmax(np.apply_along_axis(np.bincount, axis, idx.reshape(predictions.shape),
                                                None, np.max(idx) + 1), axis=axis)]
    
    out_func_file = '%s%s.predictions.consensus.%s.func.gii' % (pred_dir, schema['model'], args.data)
    write.save(consensus, out_func_file, 'L')

    out_label_file = '%s%s.predictions.consensus.%s.label.gii' % (pred_dir, schema['model'], args.data)
    bashCommand = "wb_command -metric-label-import  %s %s %s" % (
                    out_func_file, label_table, out_label_file)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    os.remove(out_func_file)


    # plot test performance metrics
    df = pd.DataFrame(np.column_stack([accuracies, F, subjects]), columns=['acc', 'f1', 'subject'])
    df_file = '%s%s.metrics.%s.csv' % (pred_dir, schema['model'], args.data)
    df.to_csv(df_file)

    fig, [ax1, ax2] = plt.subplots(1,2, figsize=(12,5))

    # test accuracy distribution
    ax1.hist(accuracies, density=True)
    ax1.tick_params(labelsize=15)
    ax1.set_xlabel('Accuracy', fontsize=15)
    ax1.set_ylabel('Density', fontsize=15)
    ax1.set_xlim([0,1])
    ax1.set_title('Model %s Performance: Accuracy' % (args.data.capitalize()), fontsize=20)

    # test F1 distribution
    ax2.hist(F, density=True)
    ax2.tick_params(labelsize=15)
    ax2.set_xlabel('F1-Score', fontsize=15)
    ax2.set_ylabel('Density', fontsize=15)
    ax2.set_xlim([0,1])
    ax2.set_title('Model %s Performance: Dice' % (args.data.capitalize()), fontsize=20)

    plt.tight_layout()
    fig_file = '%s%s.accuracies.%s.png' % (pred_dir, schema['model'], args.data)
    plt.savefig(fig_file, bbox_inches='tight', transparent=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Perform model testing on JK-GAT models.')

    parser.add_argument('--schema-file', 
                        type=str,
                        help='JSON file with parameters for model, training, and output.')
    parser.add_argument('--data', 
                        type=str, 
                        help='Assess model performance on test or validation set.',
                        default='testing', 
                        choices=['testing', 'train', 'validation'],
                        required=False)
    parser.add_argument('-no_background', 
                        help='Exclude background voxels in accuracy calculation.',
                        action='store_true',
                        required=False)
    parser.add_argument('-alpha', 
                        help='Return layerwise attention values.', 
                        action='store_true', 
                        required=False)

    args = parser.parse_args()

    main(args)
