import numpy as np

def make_data(n_per_class=100,seed=42):
    """生成三个类别的二维数据,返回x.shape=(N,2),y.shape=(N,)"""
    #创建一个随机数生成器
    rng=np.random.default_rng(seed)
    #3个类别，样本是二维；结合输入数据，样本量为100
    centers=np.array([
        [-2.0,-1.5],
        [2.0,-1.0],
        [0.0,2.0]
    ])

    X_list=[]
    y_list=[]

    for class_id,center in enumerate(centers):      #返回类别编号（从0开始）和类别中心点
        points=center+rng.normal(                   #生成随机样本，以center为中心，叠加服从正态分布的随机数
            loc=0.0,                                #该正态分布随机数平均值为0
            scale=0.65,                             #标准差为0.65，数值越大，点分布越分散
            size=(n_per_class,2)                    #生成100*2的数组
        )
        X_list.append(points)                       #生成的二维点保存到X_list，包含3类各100个二维数组，共300个二维数组
        y_list.append(np.full(n_per_class,class_id))#生成的100个相同的类别编号，循环3次，最终保存300个样本每个样本对应的类别标签

    X=np.vstack(X_list)                             #按行拼接数组，X.shape==(300,2),样本总数为300，每个样本有2个特征
    y=np.concatenate(y_list)                        #将多个一维数组连接起来，y.shape==(300,)

    #随机打乱数据
    indices=rng.permutation(len(X))
    return X[indices],y[indices]

def train_val_split(X,y,val_ratio=0.2,seed=42):
    rng=np.random.default_rng(seed)
    indices=rng.permutation(len(X))

    val_size=int(len(X)*val_ratio)                  #验证集有60个样本
    val_indices=indices[:val_size]                  #使用切片取出索引表的前60个元素，用于验证集
    train_indices=indices[val_size:]                #取出从第60个到末尾的索引，用于训练集

    X_train = X[train_indices]                      #X_train.shape:(240,2)
    y_train = y[train_indices]                      #y_train.shape:(240,)
    X_val = X[val_indices]                          #X_val.shape:(60,2)
    y_val = y[val_indices]                          #y_val.shape:(60,)

    return X_train,y_train,X_val,y_val