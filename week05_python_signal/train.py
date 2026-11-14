import numpy as np
import matplotlib.pyplot as plt

from data import make_data,train_val_split
from model import MLP
from losses import(
    cross_entropy_loss,
    softmax_cross_entropy_backward
)

def accuracy(y_true,y_pred):
    return np.mean(y_true == y_pred)

def train(model,X_train,y_train,X_val,y_val,epochs=100,learning_rate=0.05):

    train_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        # 1. 前向传播
        train_logits = model.forward(X_train)

        # 2. 计算训练损失
        loss = cross_entropy_loss(
            train_logits,
            y_train
        )

        # 3. 计算损失对Logits的梯度
        grad_logits = softmax_cross_entropy_backward(
            train_logits,
            y_train
        )

        # 4. 反向传播
        model.backward(grad_logits)

        # 5. 使用SGD更新参数
        model.update(learning_rate)

        # 6. 计算训练集准确率
        train_pred = np.argmax(
            model.forward(X_train),
            axis = 1
        )
        train_acc = accuracy(
            y_train,
            train_pred
        )

        # 7. 计算验证集准确率
        val_pred = np.argmax(
            model.forward(X_val),
            axis = 1
        )
        val_acc = accuracy(
            y_val,
            val_pred
        )

        train_losses.append(loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if(epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:03d},"
                f"loss={loss:.4f},"
                f"train_acc={train_acc:.3f},"
                f"val_acc={val_acc:.3f}"
            )

    return train_losses,train_accs,val_accs

def plot_training_curves(
        train_losses,
        train_accs,
        val_accs
):
    epochs = range(1,len(train_losses) + 1)

    plt.figure(1,figsize = (12,4))

    # Loss 曲线
    plt.subplot(1,2,1)
    plt.plot(epochs,train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(alpha=0.3)

    # Accuracy 曲线
    plt.subplot(1,2,2)
    plt.plot(epochs,train_accs,label="Train Accuracy")
    plt.plot(epochs,val_accs,label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    #plt.show()

def plot_decision_boundary(model,X,y):
    x_min = X[:,0].min() - 1
    x_max = X[:,0].max() + 1
    y_min = X[:,1].min() - 1
    y_max = X[:,1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min,x_max,300),
        np.linspace(y_min,y_max,300)
    )

    # ravel()将二维数组展开成一维数组，np.c_将两个一维数组按列拼接，相当于每行就是一个[横坐标，纵坐标]
    grid = np.c_[
        xx.ravel(),
        yy.ravel()
    ]

    grid_pred = np.argmax(
        model.forward(grid),
        axis = 1
    )

    grid_pred = grid_pred.reshape(xx.shape)

    plt.figure(2,figsize = (7,5))

    # 给模型预测的不同类别区域填充颜色，从而显示分类区域和决策边界
    plt.contourf(
        xx,
        yy,
        grid_pred,
        alpha = 0.25,
        cmap = "viridis"
    )

    colors = ["red","green","blue"]

    for class_id in range(3):
        mask = y ==class_id

        plt.scatter(
            X[mask,0],
            X[mask,1],
            color= colors[class_id],
            label = f"Class {class_id}",
            alpha = 0.7
        )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("MLP Decision Boundary")
    plt.legend()
    plt.grid(alpha = 0.3)
    #plt.show()

def main():
    np.random.seed(42)

    # 1. 生成三类二维数据
    X,y = make_data(
        n_per_class = 100,
        seed = 42
    )

    # 2. 划分训练集和验证集
    X_train,y_train,X_val,y_val = train_val_split(
        X,
        y,
        val_ratio = 0.2,
        seed = 42
    )

    # 3. 创建两层MLP
    model = MLP(
        in_features = 2,
        hidden_features = 8,
        num_classes = 3
    )

    # 4. 训练模型
    train_losses,train_accs,val_accs = train(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs = 200,
        learning_rate = 0.05
    )

    # 5. 输出最终准确率
    print(
        f"最终训练准确率：{train_accs[-1]:.3f}"
    )

    print(
        f"最终验证准确率：{val_accs[-1]:.3f}"
    )

    # 6.绘制训练曲线
    plot_training_curves(
        train_losses,
        train_accs,
        val_accs
    )

    # 7. 绘制训练后的决策边界
    plot_decision_boundary(
        model,
        X,
        y
    )

    # 前述两个子函数中原各有一个plt.show(),
    # 默认会阻塞程序需关闭一个才能绘制另一个，
    # 故改为main函数中调用完绘图函数统一调用一次，
    # 这样可以同时生成两张图片
    plt.show()

if __name__ == "__main__":
    main()