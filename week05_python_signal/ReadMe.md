# Week 05: Python and Communication Signal Basics

This week I learned:
- how to use NumPy to generate, split, and visualize three-class 2D data and train a perceptron for linear classification.
- how to implement Linear layers and ReLU, Sigmoid, and Tanh activations with forward and backward propagation.
- how to convert logits into class probabilities with Softmax, evaluate predictions with cross-entropy loss, and compute stable gradients.

## Files
- data.py and perceptron_demo.py: generate the data, implement and train the perceptron, evaluate accuracy, and plot the classification results.
- layers.py: implement Linear forward/backward propagation, calculate parameter gradients, and implement activation functions.
- losses.py: implement numerically stable Softmax, cross-entropy loss, and the combined Softmax-cross-entropy gradient.