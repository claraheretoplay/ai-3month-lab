import numpy as np

#定义线性层，数学形式为y=xW+b
class Linear:
    def __init__(self,in_features,out_features):
        #权重和偏置
        self.W = np.random.randn(in_features,out_features) * 0.1    #初始化权重，生成随机权重矩阵，*0.1使初始权重较小，避免刚开始计算时数值过大
        self.b = np.zeros(out_features)     #初始化偏置，生成全0偏置

        #保存反向传播需要的数据
        self.x = None       #用于保存前向传播时的输入x，供反向传播计算权重梯度

        #参数梯度
        self.dW = None      #用于反向传播时保存损失函数对权重W的梯度
        self.db = None      #用于反向传播时保存损失函数对偏置b的梯度

    def forward(self,x):
        """
        x.shape:(batch_size,in_features)
        输出.shape:(batch_size,out_features)
        """
        self.x = x
        return x @ self.W + self.b          #一批有batch_size个样本，每个样本有in_features个输入特征，每个样本经过线性层后得到out_features个输出特征

    def backward(self, grad_output):
        """
        grad_output是上游梯度
        grad_output.shape:(batch_size,out_features)

        返回：
        grad_input.shape:(batch_size,in_features)
        """
        #对输入 x 的梯度
        grad_input = grad_output @ self.W.T

        #对权重 W 的梯度
        self.dW = self.x.T @ grad_output

        #对偏置 b 的梯度
        self.db = np.sum(grad_output, axis=0)

        return grad_input

class ReLU:
    def __init__(self):
        self.x = None

    def forward(self,x):
        self.x = x
        return np.maximum(0,x)

    def backward(self,grad_output):
        # x > 0时梯度为1，否则为0
        grad_input = grad_output * (self.x > 0)
        return grad_input

class Sigmoid:
    def __init__(self):
        self.output = None

    def forward(self,x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output
    
    def backward(self,grad_output):
        # sigmoid'(x) = sigmoid(x) *(1 - sigmoid(x))
        grad_input = (
            grad_output
            * self.output
            * (1 - self.output)
        )

class Tanh:
    def __init__(self):
        self.output = None

    def forward(self,x):
        self.output = np.tanh(x)
        return self.output

    def backward(self,grad_output):
        #tanh'(x) = 1 - tanh(x)^2
        grad_input = grad_output * (1 - self.output ** 2)
        return grad_input


