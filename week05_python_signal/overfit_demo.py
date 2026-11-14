import copy
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from model import MLP
from optimizers import Adam
from losses import(
    cross_entropy_loss,
    softmax_cross_entropy_backward
)

def make_experiment_data(seed = 42):
    rng = np.random.default_rng(seed)

    train_X,train_y = [] , []
    val_X,val_y = [] , []

    for class_id, radius in enumerate([0.6, 1.5, 2.4]):
        #每类：12个训练样本， 150个验证样本
        angle = rng.uniform(0,2*np.pi,162)
        r = radius + rng.normal(0,0.10,162)

        points = np.column_stack((
            r * np.cos(angle),
            r * np.sin(angle)
        ))

        train_X.append(points[:12])
        train_y.append(np.full(12,class_id))

        val_X.append(points[12:])
        val_y.append(np.full(150,class_id))

    X_train = np.vstack(train_X)
    y_train = np.concatenate(train_y)
    X_val = np.vstack(val_X)
    y_val = np.concatenate(val_y)

    # 为演示过拟合，只将6个训练标签改成错误类别
    noisy = rng.choice(len(y_train), size = 6, replace = False)

    y_train[noisy] = (
        y_train[noisy] + rng.integers(1,3,size = 6)
    ) % 3

    return X_train, y_train, X_val, y_val

def parameters(model):
    # 字典中的数组直接引用模型参数
    return {
        "W1": model.linear1.W,
        "b1": model.linear1.b,
        "W2": model.linear2.W,
        "b2": model.linear2.b
    }

def gradients(model):
    # 必须在backward()之后调用
    return{
        "W1": model.linear1.W,
        "b1": model.linear1.b,
        "W2": model.linear2.W,
        "b2": model.linear2.b
    }

def evaluate(model, X, y):
    logits = model.forward(X)

    loss = cross_entropy_loss(logits,y)
    predictions = np.argmax(logits, axis = 1)
    accuracy = np.mean(predictions == y)

    return float(loss),float(accuracy)


def train_model(model, data, l2, save_path, epochs = 2000):
    X_train, y_train, X_val, y_val = data

    optimizer = Adam(learning_rate = 0.01)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    best_score = (-np.inf, -np.inf)
    best_weights = None
    best_epoch = 0

    for epoch in range(1,epochs + 1):
        # 1. 前向传播
        logits = model.forward(X_train)

        # 2. 交叉熵的反向传播
        grad_logits = softmax_cross_entropy_backward(
            logits, y_train
        )

        model.backward(grad_logits)

        params = parameters(model)
        grads = parameters(model)

        # 3. 加入 L2正则化梯度
        # 仅正则化权重，不正则化偏置
        for name in("W1","W2"):
            grads[name] = grads[name] + l2 * params[name]

        # 4. 更新参数
        optimizer.step(params,grads)

        # 5. 使用更新后的参数评估
        # 为便于比较，记录不含L2惩罚项的交叉熵
        train_loss, train_acc = evaluate(
            model, X_train, y_train
        )
        val_loss, val_acc = evaluate(
            model, X_val, y_val
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # 6. 优先选择验证准确率最高的模型
        # 准确率相同时，选择验证损失更低的模型
        score = (val_acc, -val_loss)

        if score > best_score :
            best_score = score
            best_epoch = epoch

            # 必须copy，否则后续训练会继续修改这些数组
            best_weights = {
                name : value.copy()
                for name,value in params.items()
            }

    # 7. 保存最佳权重及重建模型所需的信息
    np.savez(
        save_path,
        **best_weights,
        in_features = 2,
        hidden_features = model.linear1.W.shape[1],
        num_classes = 3,
        best_epoch = best_epoch,
        best_val_acc = best_score[0],
        best_val_loss = -best_score[1],
        l2 = l2
    )

    print(
        f"L2 = {l2:g}|"
        f"final train_acc = {train_acc:.3f}|"
        f"final val_acc = {val_acc:.3f}|"
        f"final val_loss = {val_loss:.3f}|"
        f"best val_acc = {best_score[0]:.3f}"
        f"at epoch {best_epoch}"
    )

    return history

def load_model(path):
    with np.load(path,allow_pickle = False) as saved:
        model = MLP(
            int(saved["in_features"]),
            int(saved["hidden_features"]),
            int(saved["num_classes"])
        )

        for name,param in parameters(model).items():
            param[...] = saved[name]

    return model

def main():
    np.random.seed(42)

    data = make_experiment_data()

    # 两组实验从相同初始参数开始
    initial_model = MLP(
        in_features = 2,
        hidden_features = 128,
        num_classes = 3
    )

    output_dir = (
        Path(__file__).resolve().parent / "results_day7"
    )
    output_dir.mkdir(exist_ok = True)

    histories = {}

    # 数据、模型结构、初始化、优化器及训练轮数相同
    # 只改变L2系数
    for name, l2 in(("No L2",0.0),("With L2",0.01)):
        filename = (
            "best_no_l2.npz" if l2 == 0
            else "best_l2.npz"
        )
        save_path = output_dir / filename

        histories[name] = train_model(
            copy.deepcopy(initial_model),
            data,
            l2,
            save_path
        )

        # 加载保存的最佳模型，重新评估
        restored = load_model(save_path)
        _, restored_acc = evaluate(
            restored, data[2], data[3]
        )

        print(
            f"Loaded best model val_acc:"
            f"{restored_acc:.3f}"
        )

    # 绘制两组实验的训练、验证曲线
    fig, axes = plt.subplots(1,2,figsize = (12,4))

    for name,history in histories.items():
        epochs = np.arange(
            1,len(history["train_loss"]) + 1
        )

        axes[0].plot(
            epochs,history["train_loss"],
            label = f"{name}:train"
        )
        axes[0].plot(
            epochs,history["val_loss"],
            "--",label = f"{name}:val"
        )

        axes[1].plot(
            epochs,history["train_acc"],
            label = f"{name}:train"
        )
        axes[1].plot(
            epochs,history["val_acc"],
            "--",label = f"{name}:val"
        )

    axes[0].set_title("Cross-entropy")
    axes[0].set_ylabel("Loss")

    axes[1].set_title("Accuracy")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0,1.05)

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha = 0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "comparison.png",dpi = 150)
    plt.show()

if __name__=="__main__":
    main()