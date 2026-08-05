import numpy as np
import torch
#Dataset管理单条数据（保存数据，通过__getitem__()提供单条样本和标签），DataLoader批量加载数据（将多个样本组成batch，批量读取、打乱数据）
from torch.utils.data import Dataset,DataLoader

#从week04的signal_generation.py导入
from signal_generation import(
    generate_bpsk,
    generate_qpsk,
    generate_8psk,
    generate_16qam,
    add_awgn,
)

LABEL_MAP={
    "BPSK":0,
    "QPSK":1,
    "8PSK":2,
    "16QAM":3,
}

#调制信号数据集
#class:定义类或对象模板；def:定义函数或类的方法；self:表示当前对象本身
#定义一个名为class ModulationDataset(Dataset)的数据集类，并继承PyTorch的Dataset，用于保存
##信号数据X
##标签数据y
##获取样本的方法
##获取样本数量的方法

#CNN输入需要(batch,channel,length)：(批次大小，通道数，序列长度)->(一次输入的样本数，2为I路和Q路，128为每路信号长度)
class ModulationDataset(Dataset):
    #__init__()是初始化方法，创建对象时会自动执行
    def __init__(self,X:np.ndarray,y:np.ndarray):
        #X:(样本数，2，128)，2表示I路和Q路
        self.X=torch.tensor(X,dtype=torch.float32)
        #y:(样本数，)
        self.y=torch.tensor(y,dtype=torch.long)
    #返回数据集中的样本数量
    def __len__(self):
        return len(self.y)
    #根据索引获取一条样本和对应标签
    def __getitem__(self,index):
        return self.X[index],self.y[index]

#生成训练集和验证集
def generate_dataset(
        train_per_class:int=1000,
        val_per_class:int=200,
        num_symbols:int=128,
        seed:int=0,
):
    #创建一个Numpy随机数生成器，random number generator
    rng=np.random.default_rng(seed)

    generators={
        "BPSK":generate_bpsk,
        "QPSK":generate_qpsk,
        "8PSK":generate_8psk,
        "16QAM":generate_16qam,
    }

    #创建4个空列表，用于暂时保存训练数据集和验证数据集，后续使用append()添加数据，添加完成后再转换为Numpy数据
    X_train=[]
    y_train=[]
    X_val=[]
    y_val=[]

    for modulation_name,generator in generators.items():
        label=LABEL_MAP[modulation_name]

        #将数据划分改成“每个类别单独划分”：
        class_X=[]
        class_y=[]

        total_samples=train_per_class+val_per_class
    
        for _ in range(total_samples):
            #生成无噪声调制信号
            symbols,_=generator(
                num_symbols=num_symbols,
                seed=int(rng.integers(0,1_000_000))
            )

            #随机选择一个SNR，例如0~20dB
            snr_db=rng.uniform(0,20)

            #添加AWGN噪声
            noisy_symbols=add_awgn(
                symbols,
                snr_db=snr_db,
                seed=int(rng.integers(0,1_000_000))
            )

            #复数信号转换为I/Q两路
            #形状：(2,128)
            iq_sample=np.stack([
                noisy_symbols.real,
                noisy_symbols.imag,
            ]).astype(np.float32)

            class_X.append(iq_sample)
            class_y.append(label)

        #当前类别单独划分训练集和验证集
        X_train.extend(class_X[:train_per_class])
        y_train.extend(class_y[:train_per_class])

        X_val.extend(class_X[train_per_class:])
        y_val.extend(class_y[train_per_class:])

            # if len(X_train)<len(generators)*train_per_class:
            #     X_train.append(iq_sample)
            #     y_train.append(label)
            # else:
            #     X_val.append(iq_sample)
            #     y_val.append(label)

    ##为train.py新增打印
    print("训练集标签分布:",np.bincount(y_train,minlength=4))
    print("验证集标签分布:",np.bincount(y_val,minlength=4))

        

    X_train=np.array(X_train,dtype=np.float32)
    y_train=np.array(y_train,dtype=np.int64)

    X_val=np.array(X_val,dtype=np.float32)
    y_val=np.array(y_val,dtype=np.int64)

    #分别打乱训练集和验证集
    train_indices=rng.permutation(len(X_train))
    val_indices=rng.permutation(len(X_val))

    X_train=X_train[train_indices]
    y_train=y_train[train_indices]

    X_val=X_val[val_indices]
    y_val=y_val[val_indices]

    return X_train,y_train,X_val,y_val

def main():
    X_train,y_train,X_val,y_val=generate_dataset()

    train_dataset=ModulationDataset(X_train,y_train)
    val_dataset=ModulationDataset(X_val,y_val)

    train_loader=DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
    )

    val_loader=DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
    )

    print("训练集 X shape:",X_train.shape)
    print("训练集 y shape:",y_train.shape)
    print("验证集 X shape:",X_val.shape)
    print("验证集 y shape:",y_val.shape)

    #查看一个batch
    ##iter()用于将train_loader转换为迭代器，按顺序逐批读取数据
    ##next()用于获取迭代器中的下一项数据
    batch_X,batch_y=next(iter(train_loader))

    print("Batch X shape:",batch_X.shape)
    print("Batch y shape:",batch_y.shape)
    print("Batch labels:",batch_y[:10])

if __name__ =="__main__":
    main()


