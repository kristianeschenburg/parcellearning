import argparse
from pathlib import Path
import sys
sys.path.append("../utilities/")

import dgl
import torch
import torch.nn.functional as F
import dgl.function as fn

import gcn
import gnnio, performance

import numpy as np
import time

import json

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


def main(args):

    data_dir = '/hpcdrive/parcellation/'
    model_dir = '%smodels/gcn/' % (data_dir)
    out_dir = '%s%s/' % (model_dir, args.out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    features = args.features
    features.sort()
    dSuffix = '.'.join(features)

    training_list = '%straining.txt' % (data_dir)
    training = gnnio.loaddata(args.features, training_list, dType='training')

    validation_list = '%svalidation.txt' % (data_dir)
    validation = gnnio.loaddata(args.features, validation_list, dType='validation')

    # set variables for GCN model
    n_features = training.ndata['X'].shape[1]
    n_classes = len(np.unique(training.ndata['Y']))+1
    n_hidden = [args.n_hidden] * args.n_layers

    # save script arguments to file
    args.__dict__['n_features'] = n_features
    args.__dict__['n_classes'] = n_classes

    argfile='%s%s.GCN.arguments.json' % (out_dir, dSuffix)
    gnnio.saveargs(args, argfile)

    progress = {k: [] for k in ['epoch',
                                'duration',
                                'train_loss',
                                'train_acc',
                                'val_acc',
                                'val_loss']}

    model = gcn.GCN(activation=F.relu,
                dropout=args.dropout,
                in_feats=n_features,
                n_classes=n_classes,
                n_hidden=n_hidden)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    loss_fcn = torch.nn.CrossEntropyLoss()

    dur = []
    for epoch in range(args.n_epochs):
        model.train()

        if epoch >= 3:
            t0 = time.time()

        logits = model(training, training.ndata['X'])
        loss = loss_fcn(logits, training.ndata['Y'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = performance.accuracy(logits, training.ndata['Y'])

        if epoch >= 3:
            dur.append(time.time() - t0)

        [val_acc, val_loss] = performance.evaluate(model, 
                                                   validation, 
                                                   validation.ndata['X'], 
                                                   validation.ndata['Y'], 
                                                   loss_fcn)

        print("Epoch {:05d} | Time(s) {:.4f} | Train Loss {:.4f} | Train Acc {:.4f} |"
            " Val Loss {:.4f} | Val Acc {:.4f}".format(epoch,
                                    np.mean(dur),
                                    loss.item(), train_acc,
                                    val_loss, val_acc))

        progress['epoch'].append(epoch)

        if epoch > 3:
            progress['duration'].append(time.time() - t0)
        else:
            progress['duration'].append(0)

        progress['train_loss'].append(loss.item())
        progress['train_acc'].append(train_acc)
        progress['val_acc'].append(val_acc)
        progress['val_loss'].append(val_loss)
    

    model_output = '%s%s.GCN.pt' % (out_dir, dSuffix)
    model.save(filename=model_output)

    performance_output = '%s%s.performance.GCN.json' % (out_dir, dSuffix)
    with open(performance_output, "w") as outparams:  
        json.dump(progress, outparams, ensure_ascii=True, indent=4, sort_keys=True)
    
    performance_output = '%s%s.performance.GCN.png' % (out_dir, dSuffix)
    performance.plot(progress, performance_output)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument('--features', 
                        type=str, 
                        nargs='+')
    parser.add_argument('--out-dir', 
                        type=str, 
                        help='Output directory.')
    parser.add_argument("--dropout", 
                        type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--lr", 
                        type=float, 
                        default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", 
                        type=int, 
                        default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", 
                        type=int, 
                        default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", 
                        type=int, 
                        default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", 
                        type=float, 
                        default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--self-loop", 
                        action='store_true',
                        help="graph self-loop (default=False)")
    parser.add_argument("--standardize", 
                        action='store_true',
                        help="standardize features of training data (default=True)")
    parser.set_defaults(self_loop=True)
    parser.set_defaults(standardize=True)
    
    args = parser.parse_args()
    print(args)

    main(args)