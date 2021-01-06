==============
parcellearning
==============

Python project for learning cortical maps using graph neural networks.  The module are based on the (Deep Graph Library)[https://www.dgl.ai/].

* Free software: 3-clause BSD license

Features
--------

This package has modules for training the following algorithms:
       * Graph Convolution Networks
       * Graph Attention Networks
       * Gaussian Graph Attention Networks
       * Constrained Graph Attention Networks

The package contains the following modules:
       * ```utilities```: contains various submodules for loading training / validation / testing data, early stopping of models based on some metric and a patience parameter, methods for downsampling graphs by subsetting predicted labels, and generating randomly sampled batches of training data
       * ```gcn```, ```gat```, ```gauss```, ```cgat```: main modules contained network architecture code for each of the above-mentioned network structures
       * ```functional```: contains modules for various loss functions that can be incorporated into the training
       * ```conv```: contains modules for implementing the Constrained Graph Attention Network layer, and Gaussian Graph Attention Network layer, on top of which are built the actual algorithms
