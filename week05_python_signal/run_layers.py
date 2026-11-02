import numpy as np
from layers import Linear,ReLU

x = np.array([
    [1.0,-2.0],
    [0.5,3.0]
])

layer1 = Linear(2,3)
relu = ReLU()
layer2 = Linear(3,1)

#前向传播
z1 = layer1.forward(x)
a1 = relu.forward(z1)
output = layer2.forward(a1)

print("output.shape =",output.shape)

#假设损失函数传回的梯度
grad_output = np.ones_like(output)

#反向传播，顺序必须与前向传播相反
grad = layer2.backward(grad_output)
grad = relu.backward(grad)
grad = layer1.backward(grad)

print("dx.shape =",grad.shape)
print("layer1.dW.shape =",layer1.dW.shape)
print("layer1.db.shape =",layer1.db.shape)