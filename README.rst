==============
parcellearning
==============

* Free software: 3-clause BSD license

Python project for learning cortical maps using graph neural networks.  The modules are based on the (Deep Graph Library)[https://www.dgl.ai/].

This package has modules for training the following algorithms:
**********************

* Graph Convolution Networks
* Graph Attention Networks
* Gaussian Graph Attention Networks
* Constrained Graph Attention Networks

The package contains the following submodules:
**********************


* ``utilities``: submodules for loading training / validation / testing data, class for early stopping of models based on some metric and a patience parameter, methods for downsampling graphs, and method for generating randomly sampled batches of training data
* ``gcn``, ``gat``, ``gauss``, ``cgat``: main modules containing network architecture code for each of the above-mentioned algorithms
* ``functional``: modules for various loss functions that can be incorporated into the training
* ``conv``: contains modules for implementing the Constrained Graph Attention Network layer, and Gaussian Graph Attention Network layer, on top of which are built the actual algorithms
