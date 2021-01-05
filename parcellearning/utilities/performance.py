import torch
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt


def evaluate(model, in_graph, features, labels, loss_function):
    
    """
    Evaluate loss of current model.
    
    Parameters:
    - - - - -
    model: torch nn module
        current fitted network
    in_graph: DGL Graph
        input graph (train, test, or validation)
    features: torch tensor
        node features of ```in_graph```
    labels: torch tensor
        true node classes of ```in_graph```
    """
    
    model.eval()
    with torch.no_grad():
        logits = model(in_graph, features)
        loss = loss_function(logits, labels)

        return [accuracy(logits, labels), loss.item()]

def predict(model, in_graph, features, labels):
    
    """
    Compute predicted labels for given input graph and features.

    Parameters:
    - - - - -
    model: torch nn module
        current fitted network
    in_graph: DGL Graph
        input graph (train, test, or validation)
    features: torch tensor
        node features of ```in_graph```
    labels: torch tensor
        true node classes of ```in_graph```
    
    Returns:
    - - - -
    idx: torch tensor
        predicted node classes of ```in_graph```
    """
    
    model.eval()
    with torch.no_grad():
        logits = model(in_graph, features)
        prediction = F.log_softmax(logits, 1)
        [_,idx] = torch.max(prediction, 1)
    
    return idx

def accuracy(logits, labels):

    """
    Compute accuracy of given forward pass through network.

    Parameters:
    - - - - -
    logits: torch tensor
        output of single network forward pass
    labels: torch tensor
        ground truth class of each sample

    Returns:
    - - - -
    float scalar of network prediction accuracy

    """

    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    
    return correct.item() * 1.0 / len(labels)


def plot(metrics, filename, loss_epoch=None, acc_epoch=None):

    """
    Plots the performance metrics of a model, recorded over training.

    Parameters:
    - - - - -
    metrics: dictionary
        assumes the following keys and values exist:
        train_loss, train_acc, val_loss, val_acc
    """

    fig, [ax1, ax2] = plt.subplots(1,2, figsize=(12,5))

    ax1.plot(metrics['epoch'], metrics['train_acc'], linewidth=3, label='Train Acc')
    ax1.plot(metrics['epoch'], metrics['val_acc'], linewidth=3, label='Val Acc')
    if acc_epoch:
        ax1.axvline(acc_epoch, linewidth=3, c='r', linestyle='--', 
                    label='Optimal Validation Accuracy')

    ax1.legend(loc='lower right', fontsize=15)
    ax1.tick_params(labelsize=15)
    ax1.set_xlabel('Epoch', fontsize=15)
    ax1.set_ylabel('Accuracy', fontsize=15)
    ax1.set_title('Model Accuracy', fontsize=20)

    ax2.plot(metrics['epoch'], metrics['train_loss'], linewidth=3, label='Train Loss')
    ax2.plot(metrics['epoch'], metrics['val_loss'], linewidth=3, label='Val Loss')
    if loss_epoch:
        ax2.axvline(loss_epoch, linewidth=3, c='r', linestyle='--',
                    label='Optimal Validation Loss')

    ax2.legend(loc='upper right', fontsize=15)
    ax2.tick_params(labelsize=15)
    ax2.set_xlabel('Epoch', fontsize=15)
    ax2.set_ylabel('Loss (Cross Entropy)', fontsize=15)
    ax2.set_title('Model Loss', fontsize=20)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', transparent=True)
