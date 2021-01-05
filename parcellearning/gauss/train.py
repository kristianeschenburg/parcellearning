import argparse, copy, json, random, sys, time
sys.path.append("../utilities/")
sys.path.append("../functional/")

from shutil import copyfile
from pathlib import Path

import numpy as np

import dgl
from dgl.data import register_data_args

import torch
import torch.nn.functional as F
import dgl.function as fn

import gauss, gnnio
from early_stop import EarlyStopping
from batch import partition_graphs

def main(args):


    schema = gnnio.load_schema(args.schema_file)
    in_dir = schema['data']['in']
    out_dir = schema['data']['out']
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # copy schema file to output directory
    copy_schema = ''.join([out_dir, args.schema_file.split('/')[-1]])
    copyfile(args.schema_file, copy_schema)

    # get features
    features = schema['features']
    features.sort()

    ##### GET PARAMETERS FROM SCHEMA FILE #####
    # - - - - - - - - - - - - - - - - - - - - #
    # - - - - - - - - - - - - - - - - - - - - #

    MODEL_PARAMS = schema['model_parameters']
    OPT_PARAMS = schema['optimizer_parameters']
    TRAIN_PARAMS = schema['training_parameters']
    STOP_PARAMS = schema['stopping_parameters']

    # - - - - - - - - - - - - - - - - - - - - #
    # - - - - - - - - - - - - - - - - - - - - #

    # load training and validation data
    training = gnnio.loaddata(dType='training', 
                              features=features, 
                              data_path=schema['data']['training'])

    validation = gnnio.loaddata(dType='validation', 
                                features=features, 
                                data_path=schema['data']['validation'])
    validation = dgl.batch(validation)
    val_X = validation.ndata['features']
    val_Y = validation.ndata['label']


    ##### MODEL TRAINING #####
    # - - - - - - - - - - - - #
    # - - - - - - - - - - - - #

    # instantiate model using schema parameters
    model = gauss.GAUSS(**MODEL_PARAMS)

    # instantiate Adam optimizer using scheme parameters
    optimizer = torch.optim.Adam(model.parameters(), **OPT_PARAMS)

    # initialize early stopper
    stopped_model_output='%s%s.earlystop.Loss.pt' % (out_dir, schema['model'])
    stopper = EarlyStopping(filename=stopped_model_output, **STOP_PARAMS)

    progress = {k: [] for k in ['Epoch',
                                'Duration',
                                'Train Loss',
                                'Train Acc',
                                'Val Acc',
                                'Val Loss']}

    cross_entropy = torch.nn.CrossEntropyLoss()

    dur = []
    print('Training model')
    for epoch in range(TRAIN_PARAMS['epochs']):

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

            batch_logits = model(batch, batch_X)
            batch_loss = cross_entropy(batch_logits, batch_Y)

            _, batch_indices = torch.max(F.softmax(batch_logits, dim=1), dim=1)
            batch_acc = (batch_indices == batch_Y).sum() / batch_Y.shape[0]

            # apply backward parameter update pass
            batch_loss.backward()

            # update training performance
            train_loss += batch_loss
            train_acc += batch_acc

            if TRAIN_PARAMS['verbose']:
                print("Batch Loss {:.4f} | Batch Acc {:.4f}".format(
                    batch_loss.item(), batch_acc.item()))

            # accumulate the gradients from each batch
            if (iteration+1) % TRAIN_PARAMS['n_batch'] == 0:
                optimizer.step()
                optimizer.zero_grad()

        # switch model into evaluation mode 
        # so we don't update the gradients using the validation data
        dur.append(time.time() - t0)
        model.eval()
        with torch.no_grad():
            # push validation through network
            val_logits = model(validation, val_X)

            # compute validation performance
            # loss
            val_loss = cross_entropy(val_logits, val_Y)

            # accuracy
            _, val_indices = torch.max(F.softmax(val_logits, dim=1), dim=1)
            val_acc = (val_indices == val_Y).sum() / val_Y.shape[0]

        train_loss /= TRAIN_PARAMS['n_batch']
        train_acc /= TRAIN_PARAMS['n_batch']

        # Show current performance
        print("Epoch {:05d} | Time(s) {:.4f} | " \
            "Train Loss {:.4f} | Train Acc {:.4f} | Val Loss {:.4f} | Val Acc {:.4f}".format(
            epoch, np.mean(dur), train_loss.item(), train_acc.item(), val_loss.item(), val_acc.item()))

        progress['Epoch'].append(epoch)
        if epoch > 3:
            progress['Duration'].append(time.time() - t0)
        else:
            progress['Duration'].append(0)

        # update training performance
        progress['Train Loss'].append(train_loss.item())
        progress['Train Acc'].append(train_acc.item())
        progress['Val Loss'].append(val_loss.item())
        progress['Val Acc'].append(val_acc.item())

        # set up early stopping criteria on validation loss 
        # if validation loss does not decree for patience=10
        # save best model in last 10 and break the training
        early_stop = stopper.step(val_loss.detach().data, model)
        if early_stop:
            break

    model_output = '%s%s.pt' % (out_dir, schema['model'])
    model.save(filename=model_output)

    # save performance to json
    performance_output = '%sperformance.%s.json' % (out_dir, schema['model'])
    with open(performance_output, "w") as outparams:  
        json.dump(progress, outparams, ensure_ascii=True, indent=4, sort_keys=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GConv')
    parser.add_argument('--schema-file', 
                        type=str,
                        help='JSON file with parameters for model, training, and output.')

    args = parser.parse_args()

    main(args)
