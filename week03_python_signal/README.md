# Week 03: Python and Communication Signal Basics

This week I learned:
- how to create, inspect, index, reshape, and compute with PyTorch tensors, as well as conversion between NumPy arrays and tensors
- how to use requires_grad, computational graphs, backward(), gradients, and optimizers to update model parameters
- how to define neural network models with nn.Module, __init__(), and forward()
- how to load, transform, split, and batch MNIST data using Dataset and DataLoader
- how to train a model with CrossEntropyLoss, Adam, backpropagation, and parameter updates

## Files
- tensor_basics.py:demonstrates tensor creation, tensor properties, indexing, slicing, reshaping, matrix multiplication, device selection, and NumPy–Tensor conversion
- autograd_basics.py: demonstrates automatic differentiation, gradient inspection, gradient clearing, and parameter updates with SGD
- model.py: defines a simple neural network for MNIST classification
- dataset.py: downloads, transforms, splits, and batches the MNIST dataset
- train.py: trains and validates the model while reporting loss and accuracy