# Week 03: Python and Communication Signal Basics

This week I learned:
- how I/Q signals represent digital modulation and how to generate, normalize, label, add noise to, and visualize BPSK, QPSK, 8PSK, and 16QAM I/Q signals in Python.
- how to build a PyTorch Dataset and DataLoader by generating labeled noisy modulation signals, converting I/Q NumPy arrays into tensors, batching samples, and splitting data into training and validation sets.
- how to define a 1D CNN with convolution, activation, pooling, and fully connected layers for modulation classification.
- how to train and validate the CNN using CrossEntropyLoss and Adam, monitor performance, and save the best model checkpoint.
- how to evaluate the trained CNN at different SNR levels, analyze accuracy trends, plot training curves, and organize the complete project.

## Files
- signal_generation.py:generate BPSK, QPSK, 8PSK, and 16QAM I/Q signals, adds AWGN noise, normalizes signal power, assigns integer labels, and plots constellation diagrams
- dataset.py: generates labeled noisy modulation samples, converts I/Q data into PyTorch tensors, implements Dataset and DataLoader, and creates training and validation sets.
- model.py: defines the 1D CNN architecture for classifying four modulation types.
- train.py: trains and validates the CNN, calculates loss and accuracy, and saves the best model.
- evaluate_snr.py: loads the best model, evaluates accuracy at different SNR levels, and generates performance plots.