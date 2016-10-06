# rbm

Simple Restricted Boltzmann Machine implementation. Learn a single-layer binary RBM using Contrastive Divergence. For testing I applied this to the task of MNIST digit recognition. This is done by augmenting the pixel feature vector with a 1-hot encoding of the digit class, and choosing the class that results in the hidden vector with the lowest energy.

Detailed writeups is available here: http://thevoid.ghost.io/restricted-boltzmann-machines/
