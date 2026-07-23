#导入参数解析、PyTorch、数据及工具和自己定义的模型
import argparse #用于读取命令行参数

import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
from torchvision import datasets,transforms

from model import SimpleNet

#main()主函数
def main():
    parser=argparse.ArgumentParser()    #创建一个“命令行参数解析器”，并保存到变量parser中
    #读取参数
    parser.add_argument("--epochs",type=int,default=3)  #接续3行添加参数
    parser.add_argument("--batch_size,type=int,default=64")
    parser.add_argument("--lr",type=float,default=0.001)

    #需要再创建参数解析器时定义batch_size参数，否则会报错：AttributeError: 'Namespace' object has no attribute 'batch_size'
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="每个batch的样本数量"
    )

    args=parser.parse_args()    #解析参数

    #选择设备
    device=torch.device(
        "cuda"if torch.cuda.is_available() else "cpu"
    )
    print("使用设备:",device)

    #图片转换为Tensor,并缩放到[0,1]
    transform=transforms.ToTensor()

    #数据模块
    #加载MNIST训练数据
    full_dataset=datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    #划分训练集和验证集
    train_dataset,val_dataset=random_split(
        full_dataset,
        [55000,5000],
        generator=torch.Generator().manual_seed(42)
    )

    #创建DataLoader
    train_loader=DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader=DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    #创建模型、损失函数和优化器
    model=SimpleNet().to(device)    #创建模型并将模型移动到指定设备，训练时，输入数据也必须放到同一个设备，否则报“模型和数据不在同一设备”
    loss_fn=nn.CrossEntropyLoss()   #计算预测错误程度
    optimizer=torch.optim.Adam(     #创建Adam优化器，根据梯度更新模型参数
        model.parameters(),         #获取模型中所有可训练参数，例如linear层的权重、偏置，这些参数在训练过程中被更新
        lr=args.lr                  #设置学习率
    )

    #torch.softmax()将logits转换成概率
    #torch.log_softmax()先计算softmax，再取自然对数，结果是“对数概率”
    #torch.lof_softmax(logits,dim=1)等价于torch.log(torch.softmax(logits,dim=1))
    #CrossEntropy内部逻辑近似理解为：
    ##log_probs=torch.log_softmax(logits,dim=1)
    ##loss=torch.nn.functional.nll_loss(log_probs,labels)
    #使用交叉熵时：
    ##logits=model(images)
    ##loss=loss_fn(logits,labels)
    # 总结：
    # #softmax:得到概率，用于查看结果
    # #log_softmax:得到对数概率，用于损失计算
    # #CrossEntropyLoss:内部已处理log_softmax


    #训练多个epoch
    for epoch in range(args.epochs):
        #将模型切换到“训练模式”，不会立即训练模型，也不会更新参数，而是告诉模型接下来要进行训练#训练模式会影响某些特殊层，Dropout：训练时随机丢弃部分神经元，BatchNorm：训练时使用当前batch的统计信息并更新运行统计量
        #验证或测试时使用：model.eval()，并配合：
        #with torch.no_grad():
        #   logits=model(images)
        model.train()               
        total_loss=0.0
        total_samples=0

        for images,labels in train_loader:
            images=images.to(device)
            labels=labels.to(device)

            #1.前向计算
            logits=model(images)

            #2.计算损失
            loss=loss_fn(logits,labels) #衡量预测结果和真实标签之间的差距

            #3.清空旧梯度
            optimizer.zero_grad()

            #4.反向传播
            loss.backward()

            #5.更新模型参数
            optimizer.step()    #根据梯度和学习率，更新模型中linear层的权重和偏置，更新model.parameters()

            #loss.item()将Tensor类型的损失值转换为Python浮点数；CrossEntropy默认reduction="mean",所以此处loss.item()表示当前batch内所有样本的平均损失，乘以batch大小后才得到当前batch总损失
            #images.size(0)获取images第0维的大小，也就是当前batch中的图片数量
            total_loss+=loss.item()*images.size(0)
            total_samples+=images.size(0)

        average_loss=total_loss/total_samples

        #验证模型
        #验证时不计算梯度，只检查模型预测正确率，不更新参数
        model.eval()
        correct=0
        total=0

        with torch.no_grad():
            for images,labels in val_loader:
                images=images.to(device)
                labels=labels.to(device)

                logits=model(images)
                predictions=logits.argmax(dim=1)

                correct+=(predictions==labels).sum().item()#item():将只有一个元素的Tensor转换为普通Python数字，以完成数量累加
                total+=labels.size(0)
        accuracy=correct/total

        print(
            #f"……"将变量放进字符串中；epoch通常从0开始，显示epoch+1更符合阅读习惯
            #{epoch+1}当前训练轮数，{args.epochs}总训练轮数
            #相邻的两个f-string会自动连接，因此最好在字符串末尾加逗号或空格避免输出粘连
            f"Epoch{epoch+1}/{args.epochs},"
            f"loss={average_loss:.4f}"
            f"val_accuracy={accuracy:.4f}"
        )

#程序入口
if __name__ == "__main__":
    main()

#若要修改epochs、batch-size、lr参数，可通过命令行设置
#python train.py --epochs 1 --batch_size 128 --lr 0.001