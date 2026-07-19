import numpy as np

#固定随机种子，保证每次运行结果一致
np.random.seed(42)

#样本数和特征数
n_samples=100
n_features=3

#X:输入特征矩阵，形状为（样本数，特征数）
X=np.random.randn(n_samples,n_features)

#真实模型参数
true_w=np.array([
    [2.0],
    [1.5],
    [0.5]
])#shape:(3,1)#代表3个样本，每个样本1个特征

true_b=1.0

#生成噪声
noise=0.1*np.random.randn(n_samples,1)#生成n_samples个噪声值，每个噪声值占1行

#根据y=X@w+b生成标签
y=X@true_w+true_b+noise

#用另一组参数计算预测值
w=np.array([
    [1.8],
    [-1.2],
    [0.3]
])#shape:(3,1)

b=0.8

#矩阵乘法：一次计算所有样本
y_hat=X@w+b

#转置
X_T=X.T

#输出形状
print("X的形状:",X.shape)
print("X.T的形状:",X_T.shape)
print("w的形状:",w.shape)
print("y的形状:",y.shape)
print("y_hat的形状,",y_hat.shape)

#输出前5个真实标签和预测值
print("\n前5个真实标签")
print(y[:5])

print("\n前5个预测值")
print(y_hat[:5])