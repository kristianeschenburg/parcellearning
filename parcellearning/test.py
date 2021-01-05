import argparse, json, os, random, subprocess, sys, time
from parcellearning.utilities import gnnio, load

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import torch

from niio import write

def main(args):

    schema = gnnio.load_schema(args.schema_file)

    pred_dir = '%s/predictions/' % (schema['data']['out'])
    Path(pred_dir).mkdir(parents=True, exist_ok=True)

    features = schema['features']
    features.sort()

    testing_subjects = '%ssubjects/testing.txt' % (schema['data']['in'])
    with open(testing_subjects, 'r') as f:
        subjects = f.read().split()

    label_table = '%sL.labeltable' % (schema['data']['in'])

    testing = gnnio.loaddata(dType='testing',
                             features=features,
                             data_path=schema['data']['testing'])

    # get model file
    model_parameters = '%s%s.earlystop.Loss.pt' % (schema['data']['out'], schema['model'])
    model = load.load_model(schema, model_parameters)

    accuracies = np.zeros((len(testing),))
    F = np.zeros((len(testing),))

    predictions = np.zeros((32492, len(testing)))

    for i, graph in enumerate(testing):
        model.eval()

        with torch.no_grad():

            test_X = graph.ndata['features']
            test_Y = graph.ndata['label']

            P = np.zeros((32492, 2))

            test_logits,_,_ = model(graph, test_X, test_Y)
            _,indices = torch.max(test_logits, dim=1)
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
            P[graph.ndata['idx'], 0] = indices
            P[graph.ndata['idx'], 1] = test_Y
            
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

            # remove func file
            os.remove(out_func_file)

            # store predicted maps
            predictions[graph.ndata['idx'], i] = indices

    print('Mean test accuracy: %.3f' % accuracies.mean())

    # save all predicted maps into single file
    out_func_file = '%s%s.predictions.func.gii' % (pred_dir, schema['model'])
    write.save(predictions, out_func_file, 'L')

    out_label_file = '%s%s.predictions.label.gii' % (pred_dir, schema['model'])
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
    
    out_func_file = '%s%s.predictions.consensus.func.gii' % (pred_dir, schema['model'])
    write.save(consensus, out_func_file, 'L')

    out_label_file = '%s%s.predictions.consensus.label.gii' % (pred_dir, schema['model'])
    bashCommand = "wb_command -metric-label-import  %s %s %s" % (
                    out_func_file, label_table, out_label_file)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    os.remove(out_func_file)


    # plot test performance metrics
    df = pd.DataFrame(np.column_stack([accuracies, F, subjects]), columns=['acc', 'f1', 'subject'])
    df_file = '%s%s.metrics.csv' % (pred_dir, schema['model'])
    df.to_csv(df_file)

    fig, [ax1, ax2] = plt.subplots(1,2, figsize=(12,5))

    # test accuracy distribution
    ax1.hist(accuracies, density=True)
    ax1.tick_params(labelsize=15)
    ax1.set_xlabel('Accuracy', fontsize=15)
    ax1.set_ylabel('Density', fontsize=15)
    ax1.set_xlim([0,1])
    ax1.set_title('Model Test Performance: Accuracy', fontsize=20)

    # test F1 distribution
    ax2.hist(F, density=True)
    ax2.tick_params(labelsize=15)
    ax2.set_xlabel('F1-Score', fontsize=15)
    ax2.set_ylabel('Density', fontsize=15)
    ax2.set_xlim([0,1])
    ax2.set_title('Model Test Performance: Dice', fontsize=20)

    plt.tight_layout()
    fig_file = '%s%s.accuracies.png' % (pred_dir, schema['model'])
    plt.savefig(fig_file, bbox_inches='tight', transparent=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Perform model testing.')
    parser.add_argument('--schema-file', 
                        type=str,
                        help='JSON file with parameters for model, training, and output.')

    args = parser.parse_args()

    main(args)