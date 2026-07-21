import numpy as np
import matplotlib.pyplot as plt

#固定随机种子
np.random.seed(42)

#生成0/1比特
n_samples=1000
bits=np.random.randint(0,2,size=n_samples)

#BPSK映射：0->-1,1->+1
symbols=2*bits-1

#添加高斯噪声
noise_std=0.8
noise=noise_std*np.random.randn(n_samples)
received=symbols+noise

#输入数据和标签
X=received.reshape(-1,1)#SVM标签必须是1或-1
y=symbols.reshape(-1,1)#svm使用-1/+1标签

#初始化参数
w=np.zeros((1,1))#创建一个全部由0组成的Numpy数组，数组形状为1行1列，内容为[[0.]]，等价于w=np.array([[0.0]])
b=0.0

learning_rate=0.01
epochs=1000
loss_history=[]

for epoch in range(epochs):
    #计算分类得分
    score=X@w+b

    #计算y*score
    margin=y*score

    #Hinge Loss:L=max(0,1-y*score)，是SVM常用的损失函数，用于判断分类结果是否正确，并要求预测结果具有足够大的安全间隔（至少1）
    #y:真实标签，必须是-1或+1
    #score=X@w+b：模型的分类分数
    #y*score：分类间隔

    loss_values=np.maximum(0,1-margin)
    loss=np.mean(loss_values)
    loss_history.append(loss)

    #只有margin<1的样本才产生梯度#margin>1，loss恒为0（这样的样本①分类方向正确②距离分类边界足够远③满足了安全间隔要求）
    active=(margin<1).astype(float)

    #手写hinge loss梯度
    dw=-np.mean(active*y*X,axis=0).reshape(1,1)
    db=-np.mean(active*y)

    #更新参数
    w=w-learning_rate*dw
    b=b-learning_rate*db

#训练完成后重新计算得分
score=X@w+b

#根据score的正负进行分类
predicted_symbols=np.where(score>=0,1,-1)#np.where(条件，条件为真时的值，条件为假时的值)

#将-1/+1转回0/1
predicted_bits=((predicted_symbols+1)//2).astype(int)#predicted_symbols为float型，因此需要转为int型
true_bits=bits.reshape(-1,1)

#统计准确率
accuracy=np.mean(predicted_bits==true_bits)

print("训练完成")
print("w:",w)
print("b:",b)
print("最终hinge loss:",loss_history[-1])
print("BPSK检测准确率:",accuracy)

#查看部分结果
print("\n前10个真实比特:")
print(bits[:10])

print("\n前10个预测比特:")
print(predicted_bits[:10].ravel())#ravel()用于将数组展平为一维数组#ravel()和reshape(-1)都可将数组转换为一维形式，但ravel()更直接表达

#绘制损失曲线
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Hinge Loss")
plt.title("SVM Hinge Loss")
plt.grid(True)
plt.show()