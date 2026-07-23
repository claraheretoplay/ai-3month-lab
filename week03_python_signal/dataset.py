#导入PyTorch主库，后面可以使用：torch.tensor(),torch.randn(),torch.generator()
import torch
#从PyTorch的数据工具模块中导入两个功能：random.split：随机划分数据集，例如划分训练集和验证集；DataLoader：按batch分批读取数据，并支持打乱数据
from torch.utils.data import random_split,DataLoader
#从torchvision导入：datasets:提供图像数据集，例如MNIST；transforms：对图片进行处理和转换，例如把图片转成Tensor
from torchvision import datasets,transforms

#1.将图片转换为tensor
##将图片转换为PyTorch Tensor,并通常将像素值缩放到[0,1]
##灰度图片像素值为0~255(黑->白)，缩放至[0,1],数值范围小计算稳定；梯度波动小；不同输入特征的尺度更统一有助于模型训练
transform=transforms.ToTensor() 

#2.下载MNIST训练数据
full_dataset=datasets.MNIST(    #datasets.MNIST():调用torchvision提供的MNIST数据集类
    root="./data",  #保存到当前运行目录下的data文件夹
    train=True, #True:加载训练集，共60000张手写数字图片；False:加载测试集，共10000张图片
    download=True,  #若本地没有数据就自动下载；若已经下载完成就直接读取本地数据
    transform=transform #对每张图片进行预处理
)

print("完整训练集大小:",len(full_dataset))

#3.划分训练集和验证集
train_size=55000
val_size=5000

train_dataset,val_dataset=random_split(
    full_dataset,
    [train_size,val_size],
    generator=torch.Generator().manual_seed(42) #创建PyTorch随机数生成器，设置随机种子为42；每次运行程序，训练集和验证集的划分结果都相同，便于实验复现
)

print("训练集大小:",len(train_dataset))
print("验证集大小:",len(val_dataset))

#4.创建DataLoader
train_loader=DataLoader(    #负责把多个样本组成一个batch，并在训练时逐批提供数据
    train_dataset,  #负责保存和读取单个样本，例如一张图片和它的标签
    batch_size=64,  #每次读取64张图片
    shuffle=True,   #每轮训练前打乱样本顺序，有助于模型学习
    num_workers=0   #=0：表示不创建额外进程，由主进程自己读取数据，稳定简单适合windows和学习阶段，等代码运行稳定后再尝试2/4，比较训练速度；=2：表示使用2个额外进程并读取数据，当数据预处理较复杂、数据量较大时，适当增加num_workers可提高训练速度
)
#num_workers在windows下设置大于0时，建议将程序入口写成
# if __name__ == "__main__":
#     main()

val_loader=DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,  #验证时通常不需要打乱，结果更稳定、可复现
    num_workers=0
)

#5.读取一个训练batch
images,labels=next(iter(train_loader))#从train_loader中读取一个训练batch，并把它拆成图片和标签
# iter():将train_loader转换成迭代器,准备按批次读取数据
# next():读取下一个batch,读取完所有数据后,再调用会产生StopIteration
# 训练时通常使用循环读取所有batch
# for images,labels in train_loader:
#     #使用当前batch训练模型
#     pass

print("一个batch的图像形状:",images.shape)#输出torch.Size([64,1,28,28]),[图片数量，通道数，高度，宽度]，灰度图只有1个通道，如果是彩色RGB图片通道数是3
print("一个batch的标签形状:",labels.shape)
print("图像数据类型:",images.dtype)
print("前10个标签:",labels[:10])
