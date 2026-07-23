import torch
import torch.nn as nn   
#torch.nn是PyTorch 中专门用于构建神经网络的模块，提供
# #网络层：nn.Linear、nn.Conv2d
# #激活函数：nn.ReLU
# #损失函数：nn.CrossEntropyLoss
# #模型基类：nn.Module
# #常见容器：nn.Sequential
#nn.Module 是自定义模型的基础
#nn.Linear 等是具体的网络层

class SimpleNet(nn.Module): #M需大写#nn.Module是所有 PyTorch 模型的基础类，用于组织网络层和参数
    def __init__(self): #定义网络结构
        super().__init__()

        #定义网络层#在__init__()中定义神经网络的各个层，数据之后按顺序通过
        #self.xxx表示将这些网络层保存为模型的属性，使pyTorch能自动管理他们的权重和偏置
        #若输入为x.shape = [batch_size, 1, 28, 28]
        ##展平图像：将每张28*28的图像展平成一维向量[batch_size, 1, 28, 28]->[batch_size,784]
        self.flatten=nn.Flatten()
        ##第一层全连接层，表示784个输入数字->128个输出特征，自动创建可学习的权重和偏置（y=x*w.T+b）,[batch_size,784]->[batch_size,128]
        self.linear1=nn.Linear(28*28,128)#784个输入特征，128个隐藏特征
        ##ReLU激活函数:ReLU(x)=max(0,x),负数为0，正数不变，为网络加入非线性表达能力
        self.relu=nn.ReLU()
        ##第二层全连接层，128个特征->10个输出值
        self.linear2=nn.Linear(128,10)  #最10个输出类别
        ##数据路线：28*28图像（Flatten）->784个数字(linear)->128个特征(ReLU)->128个非线性特征(linear)->10个类别分数

    def forward(self,x):    #定义数据如何经过网络
        #x:[batch_size,1,28,28]
        x=self.flatten(x)#[batch_size,784]
        x=self.linear1(x)#[batch_size,128]
        x=self.relu(x)#[batch_size,128]
        x=self.linear2(x)#[batch_size,10]

        return x #logits

if __name__=="__main__":
    model=SimpleNet()

    #模拟4张28*28的灰度图像
    ##torch.randn()生成服从标准正态分布的随机浮点数，因此这些数据只是模拟图片，不是真实图像
    ##若print(images[0,0])会显示一个28*28的随机矩阵
    ##真实灰度图片通常也采用类似格式[图片数量，通道数，高度，宽度][batch_size,channels,height,width]
    images=torch.randn(4,1,28,28)#4：图片数量，也叫batch_size；1：通道数，灰度图只有1个通道；28：图片高度；28：图片宽度


    logits=model(images)

    print(model)
    #打印模型结构
#   SimpleNet(
#   (flatten): Flatten(start_dim=1, end_dim=-1)
#   (linear1): Linear(in_features=784, out_features=128, bias=True)
#   (relu): ReLU()
#   (linear2): Linear(in_features=128, out_features=10, bias=True)
#   )
    print("输入形状:",images.shape)#显示torch.Size([4, 1, 28, 28])，表示4张图片，每张图片1个灰度通道，每个通道大小为28*28
    print("输出形状:",logits.shape)#显示torch.Size([4,10])
    print("输出logits:\n",logits)#logits是模型最后一层输出的原始分数(而非概率)，每行对应一张图片，每列对应一个类别

    #每张图片得到10个类别分数
    predictions=logits.argmax(dim=1)#找出每行分数最高的类别，作为模型预测的结果#dim=1表示沿着第2个维度，也就是每张图片的10个类别进行比较，类别编号从0开始#argmax()只返回最大值的索引，不返回概率
    print("预测类别:",predictions)

#使用 nn.CrossEntropyLoss() 时，通常不需要手动添加 Softmax，训练用法如下
# model=SimpleNet()
# loss_fn=nn.CrossEntropyLoss()

# images=torch.randn(4,1,128,128)
# labels=torch.tensor([0,1,2,3])

# logits=model(images)
# loss=loss_fn(logits,labels)

# print("loss=",loss.item())