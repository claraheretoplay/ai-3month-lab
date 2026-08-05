#CNN常见层：
# 卷积层Conv1d/Conv2d:nn.Conv1d
# 激活层ReLU:nn.ReLU()
# 池化层MaxPooled:nn.MaxPool1d(kernel_size=2)
# 自适应平均池化层AdaptiveAvgPool1d:nn.AdaptiveAvgPool1d(1)
# 展平或维度调整层:x.squeeze(-1)
# 全连接层Linear:nn.Linear(64,4)

#通信信号通常是一维时间序列，Conv1d可以沿信号长度方向提取局部特征，例如相位、幅度变化和临近采样点之间的关系，同时参数量较少，因此适合处理一维通信信号

import torch
import torch.nn as nn

class ModulationCNN(nn.Module):
    def __init__(self,num_classes:int=4):
        super().__init__()

        #定义CNN的第一组特征提取层
        ##用于按照代码顺序依次执行多个网络层：Conv1d->ReLU,前一层的输出自动作为后一层的输入
        self.features=nn.Sequential(
            #输入:(batch,2,128),batch表示一次输入的样本数量，128表示每条I/Q信号长度，即每个通道包含128个采样点或符号数据
            ##表示一维卷积层
            ##卷积步长(stride)表示卷积核每次在信号上移动的距离，kernel_size为卷积核长度，stride=2,每次移动2个位置，未指定时默认为1(可逐点扫描，不易跳过局部特征，能保留更多信息)
            ##输入长度128，卷积输出长度公式：floor[(输入长度+2*padding-kernel_size)/stride]+1
            nn.Conv1d(
                #输入有两个通道，分别是I路和Q路
                in_channels=2,
                #学习32组不同的特征
                out_channels=32,
                #卷积核每次观察连续3个信号点
                kernel_size=3,
                #两端各补1个点，使长度基本保持不变
                padding=1
            ),
            #输出:(batch,32,128)
            #使用ReLU激活函数，将负数输出变为0，保留正数，增加模型的非线性表达能力
            nn.ReLU(),

            #池化层用于对特征进行压缩和下采样，减少数据长度，同时保留重要特征
            ##MaxPool1d()取局部最大值，若kernel_size=2表示每两个相邻位置取较大值;主要作用：缩短信号长度，减少计算量；保留较明显的局部特征；降低模型过拟合风险；对信号的小幅位置变化更稳定
            ##AdaptiveAvgPool1d()求整体平均值；主要作用：对通道的全部位置求平均，得到每个通道的整体特征，方便后续输入全连接层

            #(batch,32,128)->(batch,32,64)
            #此处虽未显示指定步长，但MaxPool1d的stride默认等于kernel_size,即stride=2,因此池化层将会从长度128降为64
            nn.MaxPool1d(kernel_size=2),

            #输入：(batch,32,64),输入通道32
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            
            #输出:(batch,64,1)
            ##将一维信号的长度压缩为1(对每个通道的整个长度方向求平均)，并保留通道数
            ##优点：无论输入长度是多少，都能输出长度为1；将每个通道压缩成一个整体特征；便于连接全连接层
            nn.AdaptiveAvgPool1d(1)
        )

        #定义CNN最后的全连接分类层
        self.classifier=nn.Linear(
            #输入每个样本有64个特征
            in_features=64,
            #输出类别数量
            out_features=num_classes
        )

    #定义模型的数据前向传播过程，即规定输入数据如何经过各层得到输出
    def forward(self,x):
        #输入信号经过卷积层、激活函数、池化层和自适应平均池化层
        ##形状变化:(batch,2,128)->(batch,64,1)
        x=self.features(x)

        #(batch,64,1)->(batch,64)
        ##删除最后一个长度为1的维度，-1表示最后一维，它不会删除batch维度
        x=x.squeeze(-1)

        #输出:(batch,4),通常是分类logits，训练时可直接配合criterion=nn.CrossEntropyLoss()使用，则无需在模型末尾手动添加Softmax
        ##logits:模型最后一层输出的原始分类分数，不是概率,分数越大模型越倾向于该类别，可能为负数，也不要求和为0
        ##Softmax:作用是将logits转换为概率，torch.softmax(logits,dim=1),softmax(z_i)=exp(z_i)/Σexp(z_j),每个概率在0-1之间，概率之和为1
        ##CrossEntropyLoss:内部包含logsoftmax计算和交叉熵计算，训练时用
        ##torch.argmax:选择分数最高的类别，预测时用
        #将64个特征输入输入全连接层：nn.Linear(64,4),输出4个类别的分数，4个输出分别对应BPSK、QPSK、8PSK、16QAM
        x=self.classifier(x)

        return x

if __name__=="__main__":
    model=ModulationCNN()

    #模拟一个batch
    x=torch.randn(64,2,128)

    #调用模型时，PyTorch会自动执行model.forward(input_data)
    output=model(x)

    print("输入形状:",x.shape)
    print("输出形状:",output.shape)