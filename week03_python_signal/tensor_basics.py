#Python 是语言，Torch 是 Python 中的深度学习工具库，Tensor 是 Torch 用来保存和计算数据的核心对象。
#torch 是库，提供：Tensor 的创建和运算；CPU、GPU 计算；自动求导；神经网络模块；优化器和损失函数等
#tensor 是库提供的函数，"张量"，本质上是一个可以保存多维数据的容器
# Python
#   └── 导入 torch 库
#         └── 创建 Tensor
#               └── 进行数据计算和深度学习

import torch
import numpy as np

#1.创建tensor
a=torch.tensor([1,2,3,4])
b=torch.tensor([
    [1.0,2.0,3.0],
    [4.0,5.0,6.0]
])

print("a=",a)#pytorch为求简洁，省略0，保留.，表示是浮点数而非整数，若想显示完整的.0，可更改为print("a=",a.tolist())
print("b=\n",b)

#2.查看shape、dtype、device
print("\na.shape=",a.shape)
print("a.dtype=",a.dtype)
print("a.device=",a.device)

print("\nb.shape=",b.shape)
print("b.dtype=",b.dtype)
print("b.device=",b.device)

#3.Tensor索引
print("\nb[0]=",b[0])#第一行
print("b[1,2]=",b[1,2])#第二行第三列
print("b[:,1]=",b[:,1])#第二列#结果为一维tensor，固定打印格式为1行

#4.reshape：改变形状
c=torch.arange(12)
print("\nc=",c)

c=c.reshape(3,4)
print("c.reshape(3,4)=\n",c)

#5.矩阵乘法
m1=torch.tensor([
    [1.0,2.0],
    [3.0,4.0]
])

m2=torch.tensor([
    [5.0,6.0],
    [7.0,8.0]
])

print("\nm1*m2(逐元素乘法=\n)",m1*m2)
print("m1@m2(矩阵乘法)=\n",m1@m2)
print("torch.matmul(m1,m2)=\n",torch.matmul(m1,m2))#结果等效矩阵乘法

#6.numpy转tensor
arr=np.array([10,20,30],dtype=np.float32)
tensor_from_numpy=torch.from_numpy(arr)

print("\nnumpy数组:",arr)
print("转换后的tensor:",tensor_from_numpy)

#7.tensor转numpy
tensor=torch.tensor([7.0,8.0,9.0])
arr_from_tensor=tensor.numpy()

print("tensor:",tensor)
print("转换后的Numpy数组:",arr_from_tensor)

#8.device
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
x=torch.tensor([1.0,2.0,3.0]).to(device)

print("\n当前device:",device)
print("x.device:",x.device)