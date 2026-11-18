# Week 06: Review Data and build the baseline of CNN

This week I learned:
- how to reuse four modulation signal generators, build separate training and validation datasets across multiple SNR levels, and verify the input and output shapes of a baseline CNN.
- how Conv1d uses channels, kernel size, stride, and padding to extract local I/Q features, and how pooling and normalization affect the shapes of intermediate tensors.


## Files
- dataset.py and model_cnn.py and run_batch.py:build labeled I/Q datasets, define a four-class CNN baseline, and verify the shapes of one batch and its output.
- model_cnn_Conv1d.py and check_shapes.py:add convolution, batch normalization, and pooling layers, and print the tensor shape after each layer.
