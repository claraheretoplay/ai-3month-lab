#从Python标准库pathlib中导入Path类，用来更方便地处理文件和文件夹路径
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

from model import SimpleNet

def main():
    #1.选择运行设备
    device=torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("使用设备:",device)

    #2.加载MNIST测试集
    test_dataset=datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    #创建DataLoader
    test_loader=DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    #3.创建与训练时结构完全相同的模型
    model=SimpleNet().to(device)

    #4.加载模型参数
    # checkpoint_dir=Path("checkpoints")：创建一个表示checkpoints文件夹的路径对象
    # checkpoint_dir.mkdir(exist_ok=True)：创建文件夹，如果已存在则不报错
    # model_path=checkpoint_dir/"mnist_mlp.pth"：拼接路径，得到checkpoints/mnist_mlp.pth
    # print(model_path)
    model_path=Path("checkpoints/mnist_mlp.pth")

    #注意：若保存参数使用的是torch.save(model.state_doct(),model_path),即保存的是“参数字典”而非完整模型，加载时需使用如下机制
    #用于读取保存的模型参数，并加载到模型中
    ##读取保存文件
    state_dict=torch.load(      #torch.load():从文件中读取保存的路径
        model_path,             #模型参数文件路径
        map_location=device     #指定读取后参数存放的设备，如在GPU保存、现在用CPU加载，设置后可以避免设备不匹配错误
    )

    #state_dict是一个字典，保存模型各层参数，(只保存权重和偏置，不保存完整的模型结构)例如：
    # linear1.weight
    # linear1.bias
    # linear2.weight
    # linear2.bias
    model.load_state_dict(state_dict)   #将读取的参数加载到模型

    #若保存的是完整模型：torch.save(model,"model.pth"),加载方式应为：
    #model=torch.load("model.pth",map_location="cpu")

    #5.切换到验证模式
    model.eval()
    ##训练模式和验证模式的区别
    ###训练模式
    # Dropout:随机丢弃神经元
    # BatchNorm:使用当前batch的统计信息并更新统计量
    ###验证/测试模式
    # Dropout:停止随机丢弃
    # BatchNorm:使用已经保存的统计量

    correct=0
    total=0

    #6.关闭梯度计算
    with torch.no_grad():
        for images,labels in test_loader:
            images=images.to(device)
            labels=labels.to(device)

            logits=model(images)
            predictions=logits.argmax(dim=1)

            correct+=(predictions==labels).sum().item()
            total+=labels.size(0)

    accuracy=correct/total

    print("正确数量:",correct)
    print("总样本数:",total)
    print(f"测试集准确率:{accuracy:.4f}")


if __name__ == "__main__":
    main()