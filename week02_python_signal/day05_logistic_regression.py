# logistic_regression:
# y_hat = sigmoid(X @ w + b)
# 损失函数/二元交叉熵：
# L = -1/n × Σ[y log(y_hat) + (1-y)log(1-y_hat)]
# 求导：
# ∂L/∂z = y_hat - y

# error = y_hat - y

# dw = (X.T @ error) / n_samples
# db = np.mean(error)


import numpy as np
import matplotlib.pyplot as plt

#sigmoid函数
def sigmoid(z):
    #防止exp数值溢出
    z=np.clip(z,-50,50)
    return 1/(1+np.exp(-z))#sigmoid函数公式，sigmoid函数是常用于二分类模型的激活函数，也叫逻辑函数，可将任意实数转换到0~1之间

np.random.seed(42)

#生成随机比特标签：0或1
n_samples=1000
bits=np.random.randint(0,2,size=n_samples)

#BPSK映射：
#bit=0->-1
#bit=1->+1
symbols=2*bits-1

#添加高斯噪声
noise_std=0.8
noise=noise_std*np.random.randn(n_samples)
received=symbols+noise

#转换为机器学习数据形状：“每行一个样本，每列一个特征”的二维向量格式
X=received.reshape(-1,1)#将以为数据转换为“多行一列”的二维数组，-1表示让numpy根据元素总数自动计算行数
y=bits.reshape(-1,1)

#初始化参数
w=np.zeros((1,1))
b=0.0

learning_rate=0.1
epochs=2000
loss_history=[]

for epoch in range(epochs):
    #计算预测概率
    z=X@w+b
    y_hat=sigmoid(z)

    #防止；log(0)
    eps=1e-8
    y_hat_clip=np.clip(y_hat,eps,1-eps)

    #二元交叉熵损失#二元交叉熵BCE公式：L=-[y log(p)+(1-y) log(1-p)]
    loss=-np.mean(
        y*np.log(y_hat_clip)
        +(1-y)*np.log(1-y_hat_clip)
    )
    loss_history.append(loss)

    #计算梯度
    error=y_hat-y
    dw=(X.T@error)/n_samples
    db=np.mean(error)

    #更新参数
    w=w-learning_rate*dw
    b=b-learning_rate*db

#根据概率进行分类
probabilities=sigmoid(X@w+b)
predicted_bits=(probabilities>=0.5).astype(int)

#计算分类准确率
accuracy=np.mean(predicted_bits==y)

print("训练完成")
print("w:",w)
print("b:",b)
print("最终loss:",loss_history[-1])
print("分类准确率:",accuracy)

#显示部分预测结果
print("\n前10个真实比特:")
print(y[:10].ravel())

print("前10个预测概率:")
print(probabilities[:10].ravel())

print("前10个预测比特:")
print(predicted_bits[:10].ravel())

#绘制loss曲线
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("Logistic Regerssion Training Loss")
plt.grid(True)
plt.show()