import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from signal_generation import(
    generate_bpsk,
    generate_qpsk,
    generate_8psk,
    generate_16qam,
    add_awgn,
)

from dataset import ModulationDataset
from model import ModulationCNN

DEVICE=torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_SYMBOLS=128
SAMPLES_PER_CLASS=200
BATCH_SIZE=64
NUM_CLASSES=4

SNR_LIST=[0,5,10,15,20]

LABEL_MAP={
    "BPSK":0,
    "QPSK":1,
    "8PSK":2,
    "16QAM":3,
}

def create_snr_dataset(snr_db,seed=100):
    #生成指定SNR下的测试数据

    #rng:random number generator,创建一个Numpy随机数生成器，后续可用其生成随机数据
    rng=np.random.default_rng(seed)

    generators={
        "BPSK":generate_bpsk,
        "QPSK":generate_qpsk,
        "8PSK":generate_8psk,
        "16QAM":generate_16qam,
    }

    X=[]
    y=[]

    for modulation_name,generator in generators.items():
        label=LABEL_MAP[modulation_name]

        for _ in range(SAMPLES_PER_CLASS):
            symbols,_=generator(
                num_symbols=NUM_SYMBOLS,
                seed=int(rng.integers(0,1_000_000))#生成一个新的整数随机种子，[0,999999)
            )

            noisy_symbols=add_awgn(
                symbols,
                snr_db=snr_db,
                seed=int(rng.integers(0,1_000_000))
            )

            #复数信号转换为I/Q两个通道
            iq_sample=np.stack([
                noisy_symbols.real,
                noisy_symbols.imag,
            ]).astype(np.float32)

            X.append(iq_sample)
            y.append(label)
    #原始X和y数据类型不统一，形状不明确，不能使用Numpy高级索引，不方便转换为PyTorch Tensor
    #将列表转换为 NumPy 数组，并指定数据类型;float32适合神经网络计算，比float64占内存少，与PyTorch的torch，float32对应
    #转换为Numpy数组是为了让数据格式更规范，便于打乱、切分和输入神经网络；统一数据结构和形状、支持Numpy的批量索引、保证数据类型符合PyTorch要求、提高批量计算效率
    X=np.array(X,dtype=np.float32)
    y=np.array(y,dtype=np.int64)

    return X,y

def evaluate_model(model,data_loader):
    #计算模型准确率

    model.eval()

    correct=0
    total=0

    with torch.no_grad():
        for batch_X,batch_y in data_loader:
            batch_X=batch_X.to(DEVICE)
            batch_y=batch_y.to(DEVICE)

            logits=model(batch_X)
            predictions=torch.argmax(logits,dim=1)

            correct+=(
                predictions==batch_y
            ).sum().item()

            #返回第0维的长度，batch_y.size(0)也可写成len(batch_y)或batch_y.shape[0]
            total+=batch_y.size(0)

    return correct/total

def main():
    os.makedirs("figures",exist_ok=True)

    #创建模型
    model=ModulationCNN(
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    #加载第6天保存的最佳模型
    ##加载路径，指定加载模型参数时使用的设备
    checkpoint=torch.load(
        "checkpoints/best_model.pth",
        map_location=DEVICE
    )
    ##加载模型参数
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print("Loaded model from epoch:",checkpoint["epoch"])
    print("Best validation accuracy:",checkpoint["val_accuracy"])

    accuracy_list=[]

    for snr_db in SNR_LIST:
        X_test,y_test=create_snr_dataset(
            snr_db=snr_db,
            seed=100+snr_db
        )

        test_dataset=ModulationDataset(
            X_test,
            y_test
        )

        test_loader=DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        accuracy=evaluate_model(
            model,
            test_loader
        )

        # #测试代码
        # print("X_test shape:",X_test.shape)
        # print("y_test shape:",y_test.shape)
        # print("test_dataset length:",len(test_dataset))
        # print("test_loader length:",len(test_loader))

        #解决因为accuracy_list=[]长度不明，导致绘图时X与y的shape不一致无法绘图报错问题
        accuracy_list.append(accuracy)

        print(
            f"SNR={snr_db:2d}dB,"#输出十进制整数，最小字段宽度为2个字符,如10/02
            f"Accuracy={accuracy:.4f}"
        )

    #绘制不同SNR下的准确率曲线
    plt.figure(figsize=(8,5))#宽8英寸，高5英寸

    plt.plot(
        SNR_LIST,
        accuracy_list,
        marker="o",
        linewidth=2
    )

    plt.xlabel("SNR(dB)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs SNR")
    plt.ylim(0,1.05)
    plt.grid(True,linestyle="--",alpha=0.5)

    plt.tight_layout()#自动调整图像中各元素的布局，避免：坐标轴标签被裁剪、标题和图像重叠、多个子图之间距离太近、横纵坐标文字显示不完整
    plt.savefig(
        "figures/accuracy_vs_snr.png",
        dpi=150
    )#dpi是像素
    #图像大小为：宽8*150=1200像素，高5*150=750像素

    plt.show()

if __name__=="__main__":
    main()

