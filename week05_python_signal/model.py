import numpy as np
from layers import Linear,ReLU
from losses import cross_entropy_loss
from losses import softmax_cross_entropy_backward

class MLP:
    def __init__(
        self,
        in_features,
        hidden_features,
        num_classes
    ):
        self.linear1=Linear(
            in_features,
            hidden_features
        )

        self.relu = ReLU()

        self.linear2 = Linear(
            hidden_features,
            num_classes
        )

    def forward(self,X):
        """
        X.shape:(batch_size,in_features)
        """

        #第一层Linear
        z1 = self.linear1.forward(X)

        #ReLU激活
        a1 = self.relu.forward(z1)

        #第二层Linear
        logits = self.linear2.forward(a1)

        return logits

    def backward(self,grad_logits):
        """
        grad_logits是损失对logits的梯度
        """

        #forward的逆序进行backward
        grad_a1 = self.linear2.backward(grad_logits)
        grad_z1 = self.relu.backward(grad_a1)
        grad_X = self.linear1.backward(grad_z1)

        return grad_X

    def update(self,learning_rate=0.01):
        self.linear1.W -= learning_rate * self.linear1.dW
        self.linear1.b -= learning_rate * self.linear1.db

        self.linear2.W -= learning_rate * self.linear2.dW
        self.linear2.b -= learning_rate * self.linear2.db