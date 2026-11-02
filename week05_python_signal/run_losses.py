import numpy as np
from losses import(
    softmax,
    cross_entropy_loss,
    softmax_cross_entropy_backward
)

logits = np.array([
    [2.0,1.0,0.1],
    [0.2,2.5,0.3]
])

y_true = np.array([0,1])

probabilities = softmax(logits)
loss = cross_entropy_loss(logits,y_true)
grad_logits = softmax_cross_entropy_backward(logits,y_true)

print("概率:")
print(probabilities)

print("每行概率和:")
print(np.sum(probabilities,axis=1))

print("loss =",loss)

print("logits的梯度:")
print(grad_logits)