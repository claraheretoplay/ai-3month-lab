import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from week04_python_signal.signal_generation import(
    generate_bpsk,generate_qpsk,generate_8psk,
    generate_16qam,add_awgn,
)

GENERATORS = (
    generate_bpsk,generate_qpsk,generate_8psk,generate_16qam
)
SNR_LEVELS = (0,5,10,15,20)

class ModulationDataset(Dataset):
    def __init__(self,signals,labels):
        self.signals = torch.from_numpy(signals) #float32:(N,2,128)
        self.labels = torch.from_numpy(labels) #int64:(N,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,index):
        return self.signals[index],self.labels[index]

def make_split(samples_per_class,seed):
    rng = np.random.default_rng(seed)
    signals,labels = [],[]

    for label,generator in enumerate(GENERATORS):
        for _ in range(samples_per_class):
            symbols, _ = generator(
                num_symbols = 128,
                seed = int(rng.integers(0,2**32 - 1)),
            )
            noisy = add_awgn(
                symbols,
                snr_db = float(rng.choice(SNR_LEVELS)),
                seed = int(rng.integers(0,2**32 - 1)),
            )

            signals.append(
                np.stack((noisy.real,noisy.imag)).astype(np.float32)
            )
            labels.append(label)

    X = np.stack(signals)   #(N,2,128)
    y = np.asarray(labels,dtype = np.int64) # (N,)
    order = rng.permutation(len(y))
    return ModulationDataset(X[order],y[order])

def make_loaders():
    #用不同随机种子独立生成，训练样本不会直接进入验证集
    train_set = make_split(samples_per_class = 1000,seed = 42)
    val_set = make_split(samples_per_class = 200,seed = 43)

    return(
        DataLoader(train_set,batch_size = 64,shuffle = True),
        DataLoader(val_set,batch_size = 64,shuffle = False),
    )
