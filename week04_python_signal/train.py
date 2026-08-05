#CrossEntropyLoss可比较预测分数和真实类别，适用于多分类任务；本例模型输出4个类别的Logits,真实标签是0~3的整数，因而合适使用
#optimizer.step()根据loss.backward()计算出的梯度，使用Adam优化器更新模型参数，使下一轮预测更接近真实标签
#验证集未参与参数更新，验证准确率最高通常表示模型具有更好的泛化能力，可避免保存只在训练集上表现好的过拟合模型
#训练Loss和验证accuracy的作用
##train loss:模型在训练集上的预测误差，越低通常表示拟合越好
##validation accuracy:模型在未参与训练的验证集上的分类准确率，越高通常表示泛化能力越好
##如果训练loss持续下降而验证accuracy开始下降，可能说明模型出现了过拟合

#导入Python内置的os模块，用于与操作系统交互，用于os.makedirs()
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import generate_dataset,ModulationDataset
from model import ModulationCNN

#新增，为evalute_snr.py的调用做准备
import matplotlib.pyplot as plt

#新增，为checkpoints和figures存储在绝对项目路径做准备
from pathlib import Path

BASE_DIR=Path(__file__).resolve().parent
CHECKPOINT_DIR=BASE_DIR/"checkpoints"
FIGURES_DIR=BASE_DIR/"figures"

CHECKPOINT_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
#新增，为checkpoints和figures存储在绝对项目路径做准备

#===============================
#1.超参数
#===============================
BATCH_SIZE=64
LEARNING_RATE=0.001
EPOCHS=10
NUM_SYMBOLS=128
NUM_CLASSES=4

DEVICE=torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

#===============================
#2.验证函数
#===============================
def evaluate(model,data_loader,device):
    #model.eval()是PyTorch模型方法，不是Python的eval()函数
    # 将模型切换到评估模式，通常在验证和测试阶段使用
    # 训练阶段使用model.train()
    # 无Dropout或BatchNorm层时无区别，有这些层不使用会导致结果不稳定或不准确
    # model.eval()时会关闭随机丢弃，使用已经保存的均值和方差而不是继续更新，model.trian()则会开启/更新
    # 它不会关闭梯度计算，通常还要配合with torch.no-grad():
    model.eval()

    correct=0
    total=0

    with torch.no_grad():
        for batch_X,batch_y in data_loader:
            batch_X=batch_X.to(device)
            batch_y=batch_y.to(device)

            logits=model(batch_X)
            predictions=torch.argmax(logits,dim=1)#对比week03的train.py，torch.argmax(logits,dim=1)是Pytorch函数，logits.argmax(dim=1)是Tensor方法，二者等效的前提是，logits是Pytorch Tensor，且指定了相同的维度。若Logits是Numpy数组，应使用logits.argmax(axis=1)

            correct+=(predictions==batch_y).sum().item()#item():将只有一个元素的Tensor转换为普通Python数字，以完成数量累加
            total+=batch_y.size(0)

    accuracy=correct/total
    return accuracy

#================================
#3.主训练流程
#================================
def main():
    print("Using device:",DEVICE)

    #生成训练集和验证集
    X_train,y_train,X_val,y_val=generate_dataset(
        train_per_class=1000,
        val_per_class=200,
        num_symbols=NUM_SYMBOLS,
        seed=42,
    )

    train_dataset=ModulationDataset(
        X_train,
        y_train
    )

    val_dataset=ModulationDataset(
        X_val,
        y_val
    )

    train_loader=DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader=DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    #创建模型
    model=ModulationCNN(
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    #多分类损失函数
    loss_fn=nn.CrossEntropyLoss()

    #Adam优化器
    ##特点是收敛较快、对学习率不太敏感，适合当前的调制信号分类任务
    #结合了：
    ##Momentum：利用过去梯度的方向
    ##RMSProp：根据梯度大小自适应调整学习率
    #对每个参数，Adam会保存两个量：
    ##一阶矩：梯度的移动平均值，m_t=β1 m_(t-1)+(1-β1) g_t
    ##二阶矩：梯度平方的移动平均值，v_t=β2 v_(t-1)+(1-β2) g_t^2
    ##g_t：当前梯度
    ##m_t：梯度方向的平均
    ##v_t：梯度大小的平均
    ##β1、β2：移动平均系数
    #根据两个量更新参数：参数=参数-学习率*m_t/(sqrt(v_t)+ε)
    optimizer=torch.optim.Adam(
        #将模型中的所有可训练参数交给优化器管理，例如：卷积层权重和偏置、全连接层权重和偏置
        #优化器根据损失函数计算出的梯度更新这些参数
        model.parameters(),
        #学习率，太大训练不稳定，太小训练速度较慢
        lr=LEARNING_RATE
    )

    #创建保存模型的文件夹，若没有checkpoints文件夹，就创建；若有，通过exist_ok=True避免报错
    #后续可安全保存模型torch.save(model.state_dict()，"checkpoints/best_model.pth")
    os.makedirs("checkpoints",exist_ok=True)

    best_accuracy=-1.0

    #新增，为evaluate_snr.py的调用做准备
    train_losses=[]
    val_accuracies=[]

    #==============================
    #4.训练多个epoch
    #==============================
    for epoch in range(EPOCHS):
        model.train()

        total_loss=0.0

        for batch_X,batch_y in train_loader:
            batch_X=batch_X.to(DEVICE)
            batch_y=batch_y.to(DEVICE)

            #清空上一轮梯度
            optimizer.zero_grad()

            #前向传播
            logits=model(batch_X)

            #计算损失
            loss=loss_fn(logits,batch_y)

            #反向传播,计算每个参数的梯度
            loss.backward()

            #更新模型参数
            optimizer.step()

            total_loss+=loss.item()

        average_loss=total_loss/len(train_loader)

        #在验证集上计算准确率
        val_accuracy=evaluate(
            model,
            val_loader,
            DEVICE
        )

        #新增，为evaluate_snr.py的调用做准备
        train_losses.append(average_loss)
        val_accuracies.append(val_accuracy)

        #训练循环的range()通常从0开始，输出应为符合习惯的实际训练次数标签
        print(
            f"Epoch[{epoch+1}/{EPOCHS}]"
            f"Train Loss:{average_loss:.4f}"
            f"Validation Accuracy:{val_accuracy:.4f}"
        )

        #保存验证准确率最高的模型
        if val_accuracy>best_accuracy:
            best_accuracy=val_accuracy

            torch.save(
                {
                    "epoch":epoch+1,
                    #保存模型参数：卷积层权重，卷积层偏置，全连接层权重，全连接层偏置；保存的是模型学到的知识
                    # 加载方式为model.load_state_dict(checkpoint["model_state_dict"])
                    "model_state_dict":model.state_dict(),
                    #保存优化器的内部状态，例如Adam的：一阶梯度移动平均、二阶梯度移动平均、参数更新步数、学习率等配置；保存的是优化器“如何继续更新参数”的状态
                    # 加载方式为optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    "optimizer_state_dict":optimizer.state_dict(),
                    "val_accuracy":val_accuracy,
                },
                #"checkpoints/best_model.pth"
                #调整为绝对路径
                CHECKPOINT_DIR/"best_model.pth"
            )

            print("Best model saved.")

    print("Training finished.")
    print("Best validation accuracy:",best_accuracy)

    #新增，为evaluate_snr.py的调用做准备
    torch.save(
        {
            "train_losses":train_losses,
            "val_accuracies":val_accuracies,
        },
        #"checkpoints/training_history.pth"
        #调整为绝对路径
        CHECKPOINT_DIR/"training_history.pth"
    )

    #新增，为evaluate_snr.py的调用做准备
    history=torch.load(
        # "checkpoints/training_history.pth"
        # 调整为绝对路径
        CHECKPOINT_DIR/"training_history.pth"
    )

    os.makedirs("figures",exist_ok=True)

    plt.figure(figsize=(8,5))

    plt.plot(
        history["train_losses"],
        marker="o"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Curve")
    plt.grid(
        True,
        linestyle="--",
        alpha=0.5
    )

    plt.tight_layout()

    plt.savefig(
        #"figures/loss_curve.png",
        #调整为绝对路径
        FIGURES_DIR/"loss_curve.png",
        dpi=150
    )

    plt.show()

if __name__=="__main__":
    main()