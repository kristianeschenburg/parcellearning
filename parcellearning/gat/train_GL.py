import argparse, copy, json, os, sys, time
sys.path.append("../utilities/")
sys.path.append("../functional/")

import argparse
from pathlib import Path

import numpy as np

import dgl
from dgl.data import register_data_args

import torch as th
import torch.nn.functional as F
import dgl.function as fn

import gat
import gnnio, performance
import losses



def main(args):

    data_dir = '/hpcdrive/parcellation/'
    model_dir = '%smodels/gat/' % (data_dir)
    out_dir = '%s%s/' % (model_dir, args.out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    features = args.features
    features.sort()
    dSuffix = '.'.join(features)

    training_list = '%straining.txt' % (data_dir)
    training = gnnio.loaddata(args.features, training_list, dType='training')
    X_train = training.ndata['X']
    Y_train = training.ndata['Y']

    # compute training label adjacency matrix
    # used for penalizing label assignments to nodes
    # when those labels arent' adjacent
    hot_encoding = F.one_hot(Y_train).float()
    weight = th.matmul(hot_encoding.t(), th.matmul(
        training.adjacency_matrix(), hot_encoding))
    weight = (weight > 0).float()

    validation_list = '%svalidation.txt' % (data_dir)
    validation = gnnio.loaddata(args.features, validation_list, dType='validation')
    X_val = validation.ndata['X']
    Y_val = validation.ndata['Y']

    # set variables for GAT model and add them to arguments
    heads = [args.num_heads, args.num_out_heads]
    n_features = training.ndata['X'].shape[1]
    n_classes = len(np.unique(training.ndata['Y']))+1

    # save script arguments to file
    args.__dict__['n_features'] = n_features
    args.__dict__['n_classes'] = n_classes

    argfile = '%s%s.GAT.arguments.json' % (out_dir, dSuffix)
    gnnio.saveargs(args, argfile)

    progress = {k: [] for k in ['epoch', 'duration',
                                'train_loss', 'train_entropy_loss', 'train_graph_loss', 'train_acc',
                                'val_loss', 'val_entropy_loss', 'val_graph_loss', 'val_acc']}

    model = gat.GAT(num_layers=args.num_layers,
                        in_dim=n_features,
                        num_hidden=args.num_hidden,
                        num_classes=n_classes,
                        num_heads=heads,
                        activation=F.leaky_relu,
                        feat_drop=args.in_drop,
                        attn_drop=args.attn_drop,
                        negative_slope=args.negative_slope,
                        residual=args.residual)

    optimizer = th.optim.Adam(model.parameters(), 
                                 lr=args.lr, 
                                 weight_decay=args.weight_decay)

    entropy_loss = th.nn.CrossEntropyLoss()  
    graph_loss = losses.GraphStructure(margin=args.margin)

    # store duration of each epoch
    # we'll save the two best models in terms of minimized val loss and maximized val accuracy
    dur = []

    best_val_acc = 0
    acc_epoch=None

    best_val_loss = 100
    loss_epoch=None

    for epoch in range(args.num_epochs):
        
        copy_model_file = '%s%s.GAT.epoch.%i.pt' % (out_dir, dSuffix, epoch)
        model.save(filename=copy_model_file)

        # we'll also save the last three models
        # this is in the event of training model failure
        # we want to see what's going in just before the model fails
        temp_file = '%s%s.GAT.epoch.%i.pt' % (out_dir, dSuffix, epoch-3)
        if os.path.exists(temp_file):
            os.remove(temp_file)

        model.train()

        if epoch >= 3:
            t0 = time.time()
        
        # compute logits of training data
        train_logits = model(training, X_train)
        train_sf = F.softmax(train_logits, dim=1)

        # compute training accuracy
        _, train_indices = th.max(train_sf, dim=1)
        train_correct = th.sum(train_indices == Y_train)
        train_acc = train_correct.item() * 1.0 / len(Y_train)

        # compute training losses
        train_graph_loss = args.nu * graph_loss(training, train_sf)
        train_entropy_loss = entropy_loss(train_logits, Y_train)
        train_loss = train_graph_loss + train_entropy_loss

        if th.isinf(train_loss):
            print('Epoch: %i' % (epoch))
            print('Training loss is infinity')
            break

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)
        
        model.eval()
        with th.no_grad():
            val_logits = model(validation, X_val)
            val_sf = F.softmax(val_logits, dim=1)

            # compute validation accuracy
            _, val_indices = th.max(val_sf, dim=1)
            val_correct = th.sum(val_indices == Y_val)
            val_acc = val_correct.item() * 1.0 / len(Y_val)

            # compute validation losses
            val_graph_loss = args.nu * graph_loss(validation, val_sf)
            val_entropy_loss = entropy_loss(val_logits, Y_val)
            val_loss = val_entropy_loss + val_graph_loss
        
        # copy best loss model
        if val_loss < best_val_loss:
            loss_epoch = epoch
            best_val_loss = val_loss
            best_loss_model = copy.deepcopy(model)

        # copy best accuracy model
        if val_acc > best_val_acc:
            acc_epoch = epoch
            best_val_acc = val_acc
            best_acc_model = copy.deepcopy(model)

        print("Epoch {:05d} | Time(s) {:.4f} | Train Loss {:.4f} | Train CE Loss {:.4f} | Train Graph Loss {:.4f} | Train Acc {:.4f} |"
              " Val Loss {:.4f} | Val CE Loss {:.4f} | Val Graph Loss {:.4f} | Val Acc {:.4f}".format(
                  epoch, np.mean(dur),
                  train_loss.item(), train_entropy_loss.item(), train_graph_loss.item(), train_acc,
                  val_loss.item(), val_entropy_loss.item(), val_graph_loss.item(), val_acc))

        progress['epoch'].append(epoch)

        if epoch > 3:
            progress['duration'].append(time.time() - t0)
        else:
            progress['duration'].append(0)

        # append training performance
        progress['train_loss'].append(train_loss.item())
        progress['train_graph_loss'].append(train_graph_loss.item())
        progress['train_entropy_loss'].append(train_entropy_loss.item())
        progress['train_acc'].append(train_acc)

        # append validation performance
        progress['val_loss'].append(val_loss.item())
        progress['val_graph_loss'].append(val_graph_loss.item())
        progress['val_entropy_loss'].append(val_entropy_loss.item())
        progress['val_acc'].append(val_acc)


    model_output = '%s%s.GAT.acc.pt' % (out_dir, dSuffix)
    best_acc_model.save(filename=model_output)

    model_output = '%s%s.GAT.loss.pt' % (out_dir, dSuffix)
    best_loss_model.save(filename=model_output)

    model_output = '%s%s.GAT.pt' % (out_dir, dSuffix)
    model.save(filename=model_output)

    performance_output = '%s%s.performance.GAT.json' % (out_dir, dSuffix)
    with open(performance_output, "w") as outparams:  
        json.dump(progress, outparams, ensure_ascii=True, indent=4, sort_keys=True)
    
    performance_output = '%s%s.performance.GAT.png' % (out_dir, dSuffix)
    performance.plot(progress, performance_output, loss_epoch, acc_epoch)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument('--features', 
                        type=str, 
                        nargs='+')
    parser.add_argument('--out-dir', 
                        type=str, 
                        help='Output directory.')
    parser.add_argument("--num-epochs", 
                        type=int, 
                        default=200,
                        help="number of training epochs")
    parser.add_argument("--num-heads", 
                        type=int, 
                        default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", 
                        type=int, 
                        default=8,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", 
                        type=int, 
                        default=3,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", 
                        type=int, 
                        default=32,
                        help="number of hidden units")
    parser.add_argument("--residual", 
                        action="store_true", 
                        default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", 
                        type=float, 
                        default=.2,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", 
                        type=float, 
                        default=.2,
                        help="attention dropout")
    parser.add_argument("--lr", 
                        type=float, 
                        default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', 
                        type=float, 
                        default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', 
                        type=float, 
                        default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--margin',
                        type=float,
                        default=0.1,
                        help='Slack variable controlling margin between negative and positive samples.')
    parser.add_argument('--nu',
                        type=float,
                        default=10,
                        help='Regularizing weight on graph loss.')

    parser.add_argument("--standardize", 
                        action='store_true',
                        help="standardize features of training data (default=True)")
    parser.set_defaults(standardize=True)

    args = parser.parse_args()
    print(args)

    main(args)
