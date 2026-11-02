import numpy as np
import matplotlib.pyplot as plt

from data import make_data,train_val_split

class MulticlassPerceptron:
    def __init__(self,n_features,n_classes,learning_rate=0.1):
        # W.shape = (特征数，类别数)
        # b.shape = (类别数，)
        self.W = np.zeros((n_features,n_classes))           #权重矩阵，n_features行，n_classes列的Numpy浮点型数组
        self.b = np.zeros(n_classes)                        #偏置向量，包含3个浮点数的一维数组，每个类别对应一个偏置值
        self.learning_rate = learning_rate

    #计算各类别得分
    def scores(self,X):
        # X.shape = (N,2)
        # 返回shape = (N,3)
        return X @ self.W +self.b                           #X @ self.W形状未（N,3）,b形状为（3，），基于Numpy广播机制自动加到每一行，最终形状仍为（N,3）

    #根据得分进行预测
    def predict(self,X):
        return np.argmax(self.scores(X),axis=1)             #调用计算得分后，在每一行中寻找最大值，对应的类别即预测类别

    #训练过程，逐个样本、预测错误就立即更新
    ## X为训练数据，形状为(N,2);y为真实标签，形状为(N,)
    def fit(self,X,y,epochs = 30):
        for epoch in range(epochs):
            mistakes = 0

            ## zip(X,y)将特征和标签一一配对
            ## xi一个样本的特征，yi样本的真实类别
            for xi,yi in zip(X,y):
                score = xi @ self.W + self.b
                predicted = np.argmax(score)

                #如果预测错误，更新参数
                if predicted != yi:
                    #错误类别分数降低
                    self.W[:,predicted] -= self.learning_rate * xi      #错误类别对应的整列权重
                    self.b[predicted] -= self.learning_rate

                    #正确类别分数提高
                    self.W[:,yi] += self.learning_rate * xi             #真实类别对应的整列权重
                    self.b[yi] += self.learning_rate

                    mistakes += 1                                       #每发现一个预测错误样本，错误次数加1


            train_acc = np.mean(self.predict(X) == y)                    #计算本轮训练准确率；ture/false对等1/0，求和取平均
            print(
                f"Epoch {epoch + 1:02d},"   #编号从0开始转换为从1开始，d十进制整数，总宽度为2，不足两位左侧补0
                f"mistakes = {mistakes},"
                f"train_acc = {train_acc : .3f}" #保留3位小数
            )

def accuracy(y_true,y_pred):
    return np.mean(y_true == y_pred)

def plot_data(X,y,title):
    #类别0->红色，类别1->绿色，类别2->蓝色
    colors = ["red","green","blue"]
    names = ["类别 0","类别 1","类别 2"]

    for class_id in range(3):       #range(3)依次产生0，1，2
        mask = y == class_id        #创建布尔筛选条件，在y中找到命中class_id的位置（可能有多个），赋给mask，以便作为筛选标签在X中取值
        plt.scatter(
            X[mask,0],              #在X的第0列/特征1中取出命中当前class_id的若干数据
            X[mask,1],              #在X的第1列/特征2中取出命中当前class_id的若干数据
            c=colors[class_id],     #根据当前class_id选择颜色
            label=names[class_id],  #根据当前class_id设置图例文字
            alpha = 0.7             #设置透明度，0.7为半透明，点重叠时，半透明效果更容易观察数据密度；0全透明，1不透明
        )

    plt.xlabel("特征 1")            #设置横坐标名称
    plt.ylabel("特征 2")            #设置纵坐标名称
    plt.title(title)                #设置图像标题名称
    plt.legend()                    #显示图例
    plt.grid(alpha=0.3)             #显示网格，透明度为0.3，方便观察点位置

if __name__ == "__main__":
    #1. 生成三类二维数据
    X,y = make_data(n_per_class = 100)

    print("X.shape =",X.shape)
    print("y.shape =",y.shape)

    #2. 划分训练集和验证集
    X_train,y_train,X_val,y_val = train_val_split(X,y)

    #3. 创建并训练感知机
    model = MulticlassPerceptron(
        n_features = 2,
        n_classes = 3,
        learning_rate = 0.1
    )
    model.fit(X_train,y_train,epochs=30)

    #4. 验证集评估
    val_pred = model.predict(X_val)
    val_acc =accuracy(y_val,val_pred)
    print(f"验证集准确率 : {val_acc:.3f}")

    #5. 绘制二维散点图
    #设置中文字体和坐标轴中的负号(普通ASCII减号)，以便绘图时正确显示
    plt.rcParams["font.sans-serif"] = ["SimHei"]  #微软雅黑"Microsoft YaHei"
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize = (7,5))         #创建一个新的图像窗口，宽7英寸高5英寸
    plot_data(X,y,"三类二维数据")
    plt.show()                          #显示第一张图，查看原始数据分布

    #6. 绘制预测结果
    plt.figure(figsize = (7,5))         #再次创建一个新的图像窗口，宽7英寸高5英寸，不覆盖第一张图
    plot_data(X_val,val_pred,"感知机预测结果")
    plt.show()                          #显示第二张图，查看模型预测结果