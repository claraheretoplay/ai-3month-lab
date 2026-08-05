#生成无噪声数字调制I/Q数据，并绘制星座图
#数据约定：输入为I/Q信号，标签为BPSK、QPSK、8PSK或16QAM
#PSK只改变载波相位，星座点通常位于同一个圆上；QAM同时改变幅度和相位，星座点分布在多个不同半径上

#让函数和变量的类型注解延迟解析，通常以字符串形式保存
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

#设置中文字体
plt.rcParams["font.sans-serif"]=[
    "Microsoft YaHei",  #微软雅黑
    "SimHei",   #黑体
]
#正常显示正负号
plt.rcParams["axes.unicode_minus"]=False

#调制方式与数字标签的对应关系
label_map={
    "BPSK":0,
    "QPSK":1,
    "8PSK":2,
    "16QAM":3,
}

#输入：num_symbols：要生成的BPSK符号数量；seed:随机种子，相同随机种子生成相同结果
#输出：symbols:BPSK复数符号；bits:原始比特
def generate_bpsk(num_symbols:int,seed:int | None=0)->tuple[np.ndarray,np.ndarray]:
    #生成BPSK符号及比特，星座点为+1和-1
    ##创建Numpy随机数生成器，rng:random number generator;default_rng():创建Numpy推荐的新式随机数生成器
    rng=np.random.default_rng(seed)
    ##生成num_symbols个随机比特，integers(0,2)生成范围是[0,2)，只会生成0和1；BPSK每个符号只需要1个比特，因此是一维数组
    bits=rng.integers(0,2,size=num_symbols)
    ##1-2*bits:进行BPSK映射；.astype(complex):把数组中的数据类型转换为复数类型
    symbols=(1-2*bits).astype(complex)
    return symbols,bits

#随机生成QPSK比特，并将每两个比特映射为一个复数星座点
def generate_qpsk(num_symbols:int,seed:int | None=0)->tuple[np.ndarray,np.ndarray]:
    #生成QPSK符号及比特，每两个比特映射为一个复数符号
    rng=np.random.default_rng(seed)
    ##size=(num_symbols,2),num_symbols：行数，表示生成多少个符号；2：列数，表示每个符号使用2个比特
    ##因为QPSK每个符号携带2个比特，所以每一行对应一个QPSK符号的比特组合，因此是二维数组
    ##生成形状为（num_symbols,2）的数组：每行表示一个QPSK符号，每行包含2个比特，如[[0,0],[0,1],[1,1]]
    bits=rng.integers(0,2,size=(num_symbols,2))
    #00，01，10，11沿星座圆周排列，减少相邻符号的比特变化
    ##将每行的两个比特转换为编号：00->0,01->1,10->2,11->3
    symbol_index=2*bits[:,0]+bits[:,1]
    ##根据编号计算相位：0->45°，1->135°，2->225°，3->315°
    phase=np.pi/4+symbol_index*np.pi/2
    ##使用欧拉公式生成复数信号：e^(jθ)=cos(θ)+j sin(θ)：45°->:0.707+0.707j,135°->:-0.707+0.707j,225°->:-0.707-0.707j,315°->:0.707-0.707j,
    symbols=np.exp(1j*phase)
    #sumbols:长度为num_symbols的复数数组，bits.reshape(-1):将二维比特数组展平成一维数组[1 0 0 … 1 0 1]
    #如希望保留每两个比特对应一个符号的结构，可直接返回bits，[[1 0] [0 0] [0 1] … [1 1] [1 1] [0 1]]
    # print(bits.reshape(-1))
    # print(bits)
    return symbols,bits.reshape(-1)

#8psk:每个符号携带3bits
def generate_8psk(num_symbols:int,seed:int | None=0)->tuple[np.ndarray,np.ndarray]:
    rng=np.random.default_rng(seed)
    bits=rng.integers(0,2,size=(num_symbols,3))
    #将3个比特转换为0~7的编号
    symbol_index=(
        4*bits[:,0]+2*bits[:,1]+bits[:,2]
    )
    #8psk将一个完整圆周360°平均分成8个位置：2*pi/8
    phase=symbol_index*2*np.pi/8
    symbols=np.exp(1j*phase)
    return symbols,bits.reshape(-1)

#16QAM：每个符号携带4bits
#tuple->[np.ndarray,np.ndarray],表示函数返回一个包含两个Numpy数组的元组，第一个np.ndarray表示调制后的符号数组，第二个表示原始比特数组
def generate_16qam(num_symbols:int,seed:int | None=0)->tuple[np.ndarray,np.ndarray]:
    rng=np.random.default_rng(seed)
    bits=rng.integers(0,2,size=(num_symbols,4))

    #前两位决定I路，后两位决定Q路
    i_index=2*bits[:,0]+bits[:,1]
    q_index=2*bits[:,2]+bits[:,3]

    #16QAM的I/Q电平：I路或Q路可以取到的幅度值，均有4种（需2比特表示，两轴共4bit,2^4=16，称之16QAM），因此共计4*4=16个星座点
    levels=np.array([-3,-1,3,1])

    i_values=levels[i_index]
    q_values=levels[q_index]

    #除以sqrt(10)，使平均功率约为1；信号功率与幅度的平方成正比，因此按sqrt缩放
    symbols=(i_values+1j*q_values)/np.sqrt(10)
    return symbols,bits.reshape(-1)

#理想QPSK的星座点是4个清晰的点，加入AWGN后，星座点会在原位置附近扩散成4个点云
#根据指定SNR给复数信号加入AWGN噪声
def add_awgn(signal:np.ndarray,snr_db:float,seed:int | None=0)->np.ndarray:
    #给复基带信号加入AWGN，供后续调制识别实验使用
    rng=np.random.default_rng(seed)
    #计算输入信号的平均功率（复数信号功率：I^2+Q^2）
    signal_power=np.mean(np.abs(signal)**2)
    #计算SNR噪声功率，从信噪比公式反推：SNR=10*log10(signal_power/noise_power)
    noise_power=signal_power/(10**(snr_db/10))
    #生成与输入信号形状相同的标准高斯随机数，用于产生I路和Q路噪声
    ##np.sqrt(noise_power/2),将总噪声功率平均分配给I路和Q路，并转换为噪声标准差
    noise=np.sqrt(noise_power/2)*(rng.standard_normal(signal.shape)+1j*rng.standard_normal(signal.shape))
    #返回带噪信号
    return signal+noise

def plot_constellations(data:dict[str,np.ndarray])->None:
    #data是一个字典例如{"BPSK":bpsk_symbols,"QPSK":qpsk_symbols}
    #->None表示函数不返回数据，只负责绘图

    fig,axes=plt.subplots(1,len(data),figsize=(10,4),squeeze=False)
    #绘制I/Q星座图
    #axes[0]表示第一行的所有子图，同时遍历axes[0]中的每个子图和data中的每个(标签，符号数组)
    #第一次循环：ax->BPSK子图，label->"BPSK",symbols->bpsk_symbols
    for ax,(label,symbols) in zip(axes[0],data.items()):
        #s=20：点的大小，alpha=0.7：点的透明度
        ax.scatter(symbols.real,symbols.imag,s=20,alpha=0.7)
        #分别绘制Q=0的水平参考线和I=O的垂直参考线
        ax.axhline(0,color="gray",linewidth=0.8)
        ax.axvline(0,color="gray",linewidth=0.8)
        ax.set_title(label)
        ax.set_xlabel("I路")
        ax.set_ylabel("Q路")

        #绘图新增，保持横纵坐标范围、比例一致
        ax.set_xlim(-1.2,1.2)
        ax.set_ylim(-1.2,1.2)
        ax.set_aspect("equal",adjustable="box")
        #绘图新增，保持横纵坐标范围、比例一致

        #显示网格虚线
        ax.grid(True,linestyle="--",alpha=0.4)

    #自动调整子图间距，防止标题和坐标轴标签重叠
    plt.tight_layout()
    #显示最终图像
    plt.show()

#绘制不同SNR下的星座图函数
# SNR = 0 dB   ：噪声很强，星座点扩散明显
# # 低 SNR 下不同调制方式更难区分：低 SNR 表示噪声较强，星座点会严重扩散甚至相互重叠，使原本不同的幅度、相位和星座结构变得不明显
# SNR = 5 dB   ：仍然有较明显扩散
# SNR = 10 dB  ：四个星座点逐渐清晰
# SNR = 20 dB  ：星座点集中，接近理想 QPSK
# # SNR 越大，噪声功率越小，星座点越集中在理想位置，信号越清晰、越容易识别
# # 训练数据要有多种噪声强度：实际通信环境中的噪声强度会变化；使用多个 SNR 的训练数据，可以让模型学习不同噪声条件下的信号特征，提高泛化能力和识别准确率
def plot_snr_constellations(signal:np.ndarray,snr_list:list[int])->None:
    #创建一个包含2行2列，共4个子图的绘图窗口。fig:整个画布，axes：4个子图的坐标轴对象，2，2：创建2行2列子图，figsize=(10,8):整个画布宽10英寸，高8英寸
    fig,axes=plt.subplots(2,2,figsize=(10,8))
    #axes的形状是二维数组，使用.ravel()将其转变为一维数组，以便使用一层循环遍历所有子图（否则需要使用双重循环for row in axess: for ax in row: ……）
    axes=axes.ravel()

    for ax,snr_db in zip(axes,snr_list):
        #seed=snr_db+100，每次运行结果可以复现：相同的snr和相同的seed会生成相同的噪声；不同SNR使用不同的噪声序列，避免所有SNR使用相同的随机噪声
        #若snr_db为浮点数，应写成seed=int(snr_db)+100
        noisy_signal=add_awgn(signal,snr_db=snr_db,seed=snr_db+100)
        #s表示散点大大小，数值越大，点越大。s=5点较小，s=20中等大小，s=50点较大
        #alpha表示散点的透明度。alpha=1.0完全不透明，alpha=0.6半透明，alpha=0.2更透明
        ax.scatter(noisy_signal.real,noisy_signal.imag,s=20,alpha=0.6)
        ax.axhline(0,color="gray",linewidth=0.8)
        ax.axvline(0,color="gray",linewidth=0.8)
        ax.set_title(f"QPSK,SNR={snr_db}dB")
        ax.set_xlabel("I路")
        ax.set_ylabel("Q路")
        ax.set_xlim(-1.5,1.5)
        ax.set_ylim(-1.5,1.5)
        ax.set_aspect("equal",adjustable="box")
        ax.grid(True,linestyle="--",alpha=0.4)

    plt.tight_layout()
    plt.show()
    

#定义主函数，->None表示不返回结果
def main()->None:
    # #生成1000个BPSK符号，
    # num_symbols=1000
    num_symbols=128
    #_表示忽略返回的原始比特
    bpsk_symbols, _=generate_bpsk(num_symbols,seed=1)
    qpsk_symbols, _=generate_qpsk(num_symbols,seed=2)
    psk8_symbols, _=generate_8psk(num_symbols,seed=3)
    qam16_symbols, _=generate_16qam(num_symbols,seed=4)

    print("BPSK标签:",label_map["BPSK"]) 
    print("QPSK标签:",label_map["QPSK"]) 
    print("8PSK标签:",label_map["8PSK"]) 
    print("16QAM标签:",label_map["16QAM"]) 

    print("BPSK数据长度:",len(bpsk_symbols))
    print("QPSK数据长度:",len(qpsk_symbols))
    print("8PSK数据长度:",len(psk8_symbols))
    print("16QAM数据长度:",len(qam16_symbols))

    #复数z=I+jQ，因此一条通信信号由I、Q两路共同表示
    print("BPSK前5个I/Q样本")
    #[:5]：表示取前5个
    #np.column_stack():将I、Q两列拼接起来，输出形式类似[[1. 0.] [-1. 0.] [1. 0.]]
    print(np.column_stack((bpsk_symbols.real[:5],bpsk_symbols.imag[:5])))
    print("标签:BPSK")
    print("QPSK前5个I/Q样本:")
    print(np.column_stack((qpsk_symbols.real[:5],qpsk_symbols.imag[:5])))
    print("标签:QPSK")

    plot_constellations({
        "BPSK":bpsk_symbols,
        "QPSK":qpsk_symbols,
    #   "QPSK with noise"：noisy_qpsk,
        "8PSK":psk8_symbols,
        "16QAM":qam16_symbols,
    })

    #后续加入噪声时可以使用：
    #noisy_qpsk=add_awgn(qpsk_symbols,snr_db=10,seed=3)

    snr_list=[0,5,10,20]
    plot_snr_constellations(signal=qpsk_symbols,snr_list=snr_list)

    #绘图新增,检查是否生成了正确的星座点
    print("BPSK星座点:",np.unique(bpsk_symbols))
    print("QPSK数据形状:",qpsk_symbols.shape)
    print("QPSK星座点:",np.unique(qpsk_symbols))
    #绘图新增,检查是否生成了正确的星座点


if __name__ == "__main__":
    main()