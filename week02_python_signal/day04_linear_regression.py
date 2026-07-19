# linear_regression：
# y_hat = X @ w + b
# 损失函数：
# L = 1/n × Σ(y_hat - y)²
# 求导：
# error = y_hat - y
# dw = (2 / n_samples) * (X.T @ error)
# db = (2 / n_samples) * np.sum(error)

import numpy as np
import matplotlib.pyplot as plt

#固定随机种子，保证结果可复现
np.random.seed(42)

#生成训练数据：y=3x+2+noise
n_samples=100
X=np.random.randn(n_samples,1)
noise=0.2*np.random.randn(n_samples,1)
y=3*X+2+noise

#初始化模型参数
w=np.array([[0.0]])#权重
b=0.0            #偏置

learning_rate=0.05#每次参数更新的步长，步长大大训练不稳定，步长太小训练速度慢
epochs=1000#定义训练轮数
loss_history=[]#保存每轮的损失值

for epoch in range(epochs):
    #预测
    y_hat=X@w+b#模型

    #计算均方误差MSE#MSE=平均（（预测值-真实值）^2）
    error=y_hat-y
    loss=np.mean(error**2)#将每个误差平方，计算所有平方误差的平均值
    loss_history.append(loss)#将当前一轮的损失值保存到loss_history列表中，以便绘制Loss曲线，观察模型是否逐渐学好
    #通常训练过程中，loss逐渐下降->模型预测越来越接近真实值


    #手写梯度#梯度=对损失函数loss求导，过程遵循链式法则
    dw=(2/n_samples)*(X.T@error)#计算权重梯度
    db=(2/n_samples)*np.sum(error)#计算偏置梯度

    #更新参数#新参数=旧参数-学习率*梯度，梯度指向损失增大的方向，所以要减去梯度，朝损失梯度减小的方向移动
    w=w-learning_rate*dw
    b=b-learning_rate*db

# print("w的形状:",w.shape)
# print("w的元素个数:",w.size)
# print("w的内容:",w)

#输出训练结果
print("训练完成")
print("学习到的w:",w.item())
print("学习到的b:",b)
print("最终loss:",loss_history[-1])

#绘制loss曲线
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss")
plt.grid(True)
plt.show()