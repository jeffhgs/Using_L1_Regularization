
# Synopsis

This repository is sample code for the article [Building Sparse Models for Deep Learning Using L1 Regularization](http://www.groovescale.com/posts/Building_Sparse_Models_Using_L1_Regularization/).

# Background

We will use Python 3.7.9 and Tensorflow 2.3.1 as our framework.

The Tensorflow 2 User Guide section called [Writing a training loop from scratch](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch) provided a base model for our analysis.  It is a simple feedforward model on the mnist hand written digit data.

The sample code trains mnist handwritten digit classification using a simple feed forward network.

The article discusses multiple variations of execution.  Command line interface arguments are provided so that all the variations can be run through command line arguments.

# Instructions

Install tensorflow with GPU:

    pip install tensorflow-gpu==2.3.1

Or without GPU:

    pip install tensorflow==2.3.1

Use GPU support if you have it, but don't worry if you don't.  Due to the small data set of the base model, GPU acceleration is not a big win.

The article gives command line switches for running the program.
