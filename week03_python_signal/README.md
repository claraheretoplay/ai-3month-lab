# Week 03: Python and Communication Signal Basics

This week I learned:
- how to create, inspect, index, reshape, and compute with PyTorch tensors, as well as conversion between NumPy arrays and tensors
- how to use requires_grad, computational graphs, backward(), gradients, and optimizers to update model parameters
- how to define neural network models with nn.Module, __init__(), and forward()
- how to load, transform, split, and batch MNIST data using Dataset and DataLoader
- how to train a model with CrossEntropyLoss, Adam, backpropagation, and parameter updates
- how to evaluate model accuracy, switch between training and evaluation modes, and save and load model parameters
- how to create .gitignore

## Files
- tensor_basics.py:demonstrates tensor creation, tensor properties, indexing, slicing, reshaping, matrix multiplication, device selection, and NumPy–Tensor conversion
- autograd_basics.py: demonstrates automatic differentiation, gradient inspection, gradient clearing, and parameter updates with SGD
- model.py: defines a simple neural network for MNIST classification
- dataset.py: downloads, transforms, splits, and batches the MNIST dataset
- train.py: trains and validates the model while reporting loss and accuracy
- evaluate.py: loads the trained model and evaluates its accuracy on the MNIST test set
- .gitignore: specifies files and directories that Git should exclude from version control
  -- data/:MNIST数据
  -- __pycache__/:Python缓存
  -- *.py[code]:code可为c、o、d、e,*.pyc：编译字节码；.pyo:旧版本Python的优化字节码；.pyd:Windows下的Python扩展模块；.pye:较少见的扩展名
  -- *.pth:模型参数文件
  -- checkpoints/:模型保存目录
  -- .venv/:忽略虚拟环境
