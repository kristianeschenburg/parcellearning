import argparse, json, os, time

from parcellearning import jkgat
from parcellearning.utilities import gnnio
from parcellearning.utilities.early_stop import EarlyStopping
from parcellearning.utilities.batch import partition_graphs
from parcellearning.utilities.load import load_schema, load_model

from shutil import copyfile
from pathlib import Path

import numpy as np
import pandas as pd

import dgl
from dgl.data import register_data_args
import dgl.function as fn

import torch
import torch.nn.functional as F


def main(args):

    schema = load_schema(args.schema_file)
    out_dir = schema['data']['out']
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # copy schema file to output directory
    copy_schema = ''.join([out_dir, args.schema_file.split('/')[-1]])
    if not os.path.exists(copy_schema):
        copyfile(args.schema_file, copy_schema)

    ##### GET PARAMETERS FROM SCHEMA FILE #####
    # - - - - - - - - - - - - - - - - - - - - #
    # - - - - - - - - - - - - - - - - - - - - #
  
    MODEL_PARAMS = schema['model_parameters']
    OPT_PARAMS = schema['optimizer_parameters']
    TRAIN_PARAMS = schema['training_parameters']
    STOP_PARAMS = schema['stopping_parameters']

    DATA_PARAMS = schema['variable_parameters']

    # - - - - - - - - - - - - - - - - - - - - #
    # - - - - - - - - - - - - - - - - - - - - #
 
    features = DATA_PARAMS['features']
    features.sort()
 
    # load training and validation data
    training = gnnio.dataset(features=features,
                             dSet=schema['data']['training'],
                             atlas=DATA_PARAMS['response'],
                             norm=True,
                             clean=True)

    validation = gnnio.dataset(features=features,
                               dSet=schema['data']['validation'],
                               atlas=DATA_PARAMS['response'],
                               norm=True,
                               clean=True)

    validation = dgl.batch(validation)
    val_X = validation.ndata['features']
    val_Y = validation.ndata['label']

    ##### MODEL INITIALIZATION #####
    # - - - - - - - - - - - - #
    # - - - - - - - - - - - - #

    # instantiate model using schema parameters
    if args.existing:

        print('Loading preexisting model')
        model_parameters = '%s%s.earlystop.Loss.pt' % (schema['data']['out'], schema['model'])
        model = load_model(schema, model_parameters)

        model_progress = '%sperformance.%s.json' % (schema['data']['out'], schema['model'])
        with open(model_progress, 'r') as f:
            progress = json.load(f)

    else:

        print('Training new model')
        model = jkgat.JKGAT(**MODEL_PARAMS)

        progress = {k: [] for k in ['Epoch',
                                    'Duration',
                                    'Train Loss',
                                    'Train Acc',
                                    'Val Loss',
                                    'Val Acc']}

    print(model)

    # instantiate Adam optimizer using scheme parameters
    optimizer = torch.optim.Adam(model.parameters(), **OPT_PARAMS)

    # initialize early stopper
    stopped_model_output='%s%s.earlystop.Loss.pt' % (out_dir, schema['model'])
    stopper = EarlyStopping(filename=stopped_model_output, **STOP_PARAMS)

    cross_entropy = torch.nn.CrossEntropyLoss()

    dur = []

    ##### MODEL TRAINING #####
    # - - - - - - - - - - - - #
    # - - - - - - - - - - - - #


    # if model has already been trained for some time and not finished,
    # we start at the last completed epoch.  Otherwise we start at 0.
    starting_epoch = len(progress['Epoch'])

    print('\nTraining model\n')
    for epoch in range(starting_epoch, TRAIN_PARAMS['epochs']):

        # learn model on training data
        batches = partition_graphs(training, TRAIN_PARAMS['n_batch'])

        model.train()
        t0 = time.time()

        # zero the gradients for this epoch
        optimizer.zero_grad()
        
        # aggregate training batch losses
        train_loss = 0

        # aggregate training batch accuracies
        train_acc = 0

        for iteration, batch in enumerate(batches):

            # get training features for this batch
            batch_X = batch.ndata['features']
            batch_Y = batch.ndata['label']

            # push batch through network and compute loss
            batch_logits = model(batch, batch_X)
            batch_loss = cross_entropy(batch_logits, batch_Y)

            # accuracy
            batch_softmax = F.softmax(batch_logits, dim=1)
            _, batch_indices = torch.max(batch_softmax, dim=1)
            batch_acc = (batch_indices == batch_Y).sum() / batch_Y.shape[0]

            # apply backward parameter update pass
            batch_loss.backward()

            print('Batch: %i | Batch Acc: %.3f | Batch Loss: %.3f ' % (iteration+1, batch_acc.item(), batch_loss.item()))

            # update training performance
            train_loss += batch_loss
            train_acc += batch_acc

            # accumulate the gradients from each batch
            if (iteration+1) % TRAIN_PARAMS['n_batch'] == 0:
                optimizer.step()
                optimizer.zero_grad()

        dur.append(time.time() - t0)

        # switch model into evaluation mode 
        # so we don't update the gradients using the validation data
        model.eval()
        with torch.no_grad():

            # push validation through network
            val_logits = model(validation, val_X)
            val_loss = cross_entropy(val_logits, val_Y)

            # accuracy
            val_softmax = F.softmax(val_logits, dim=1)
            _, val_indices = torch.max(val_softmax, dim=1)
            val_acc = (val_indices == val_Y).sum() / val_Y.shape[0]
        
        train_loss /= TRAIN_PARAMS['n_batch']
        train_acc /= TRAIN_PARAMS['n_batch']

        # Show current performance
        print("Epoch {:05d} | Time(s) {:.4f} | Train Loss {:.4f} | Train Acc {:.4f} | Val Loss {:.4f} | Val Acc {:.4f}".format(
            epoch, np.mean(dur),
            train_loss.item(), train_acc.item(),
            val_loss.item(), val_acc.item()))

        progress['Epoch'].append(epoch)
        if epoch > 3:
            progress['Duration'].append(time.time() - t0)
        else:
            progress['Duration'].append(0)

        # update training performance
        progress['Train Loss'].append(train_loss.item())
        progress['Train Acc'].append(train_acc.item())
        
        # update validation performance
        progress['Val Loss'].append(val_loss.item())
        progress['Val Acc'].append(val_acc.item())

        # set up early stopping criteria on validation loss
        early_stop = stopper.step(val_loss.detach().data, model)
        if early_stop:
            break

    ##### MODEL SAVING #####
    # - - - - - - - - - - - - #
    # - - - - - - - - - - - - #

    model_output = '%s%s.pt' % (out_dir, schema['model'])
    model.save(filename=model_output)

    # save performance to json
    performance_output = '%sperformance.%s.json' % (out_dir, schema['model'])
    with open(performance_output, "w") as outparams:  
        json.dump(progress, outparams, ensure_ascii=True, indent=4, sort_keys=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='JKGAT')
    parser.add_argument('--schema-file', 
                        type=str,
                        help='JSON file with parameters for model, training, and output.')
                        
    parser.add_argument('--existing', 
                        help='Load pre-existing model to continue training.', 
                        action='store_true', 
                        required=False)

    args = parser.parse_args()

    main(args)
