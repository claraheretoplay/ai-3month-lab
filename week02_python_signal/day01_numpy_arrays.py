import numpy as np

#4个接收信号，每个信号包含5个采样点
signals=[
    [0.1,0.2,0.3,0.4,0.5],
    [1.1,1.2,1.3,1.4,1.5],
    [2.1,2.2,2.3,2.4,2.5],
    [3.1,3.2,3.3,3.4,3.5],
]

#将Python列表转换为NumPy二维数组
X=np.array(signals,dtype=np.float32)

print("数组内容：")
print(X)

print("shape:",X.shape)
print("ndim:",X.ndim)
print("dtype:",X.dtype)

#索引，第1行、第2列
print("第1行、第2列:",X[0,1])

#切片：取前两行、前后三列
print("前两行前三列:")
print(X[:2,:3])#冒号右侧的数字/结束位置不包含

#reshape:将数组变为一维数组
X_1d=X.reshape(-1)#让Numpy自动计算这一维的长度，转变为一维数组
print("reshape后:",X_1d)
print("reshape后的shape:",X_1d.shape)

#对比Python列表和Numpy数组
python_list=[1,2,3]
numpy_array=np.array([1,2,3])

print("Python列表乘2:",python_list*2)#表示把整个列表重复两次
print("Numpy数组乘2",numpy_array*2)#表示让每个元素乘以2