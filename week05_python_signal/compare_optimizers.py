import copy
import numpy as np
import matplotlib.pyplot as plt

from data import make_data,train_val_split
from model import MLP
from losses import(
    cross_entropy_loss,
    softmax_cross_entropy_backward
)
from optimizers import SGD,Adam

def get_params(model):
    """获取模型参数,字典中的数组直接引用模型参数"""
    return{
        "W1": model.linear1.W,
        "b1": model.linear1.b,
        "W2": model.linear2.W,
        "b2": model.linear2.b
    }

def get_grads(model):
    """必须在model.backward()之后调用"""
    return{
        "W1": model.linear1.dW,
        "b1": model.linear1.db,
        "W2": model.linear2.dW,
        "b2": model.linear2.db
    }

def evaluate(model,X,y):
    logits = model.forward(X)
    loss = cross_entropy_loss(logits,y)
    predictions = np.argmax(logits,axis = 1)
    accuracy = np.mean(predictions == y)

    return loss,accuracy

def main():
    np.random.seed(42)

    # 1. 生成并划分数据
    X,y = make_data(n_per_class = 100, seed = 42)

    X_train,y_train,X_val,y_val = train_val_split(
        X,y,val_ratio = 0.2, seed = 42
    )

    # 2. 创建一个初始模型，再复制为两个独立模型
    initial_model = MLP(
        in_features = 2,
        hidden_features = 8,
        num_classes = 3
    )

    models = {
        "SGD": copy.deepcopy(initial_model),
        "Adam": copy.deepcopy(initial_model)
    }

    # 初始数值完全相同，但训练时不会共享参数数组
    for key in get_params(initial_model):
        np.testing.assert_array_equal(
            get_params(models["SGD"])[key],
            get_params(models["Adam"])[key]
        )
    
    optimizers = {
        "SGD": SGD(learning_rate = 0.05),
        "Adam": Adam(learning_rate = 0.001)
    }

    histories = {
        name: {"loss": [], "val_acc": []}
        for name in models
    }

    epochs = 200
    batch_size = 32
    rng = np.random.default_rng(123)

    # 3. 完整训练循环
    for epoch in range(epochs):
        # 两个模型共用这一轮的样本顺序
        indices = rng.permutation(len(X_train))

        for start in range(0, len(X_train),batch_size):
            batch_indices = indices[start:start+batch_size]

            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            # 两个模型使用完全相同的当前批次
            for name,model in models.items():
                # 前向传播
                logits = model.forward(X_batch)

                # 计算损失对logits的梯度
                grad_logits = softmax_cross_entropy_backward(
                    logits,y_batch
                )

                # 反向传播，得到所有参数梯度
                model.backward(grad_logits)

                # 优化器更新参数
                optimizers[name].step(
                    get_params(model),
                    get_grads(model)
                )

        # 4. 每轮结束后，用当前参数统一评估
        for name,model in models.items():
            train_loss, _ = evaluate(
                model,X_train,y_train
            )
            _, val_acc = evaluate(
                model,X_val,y_val
            )

            histories[name]["loss"].append(train_loss)
            histories[name]["val_acc"].append(val_acc)

            if(epoch + 1) % 20 == 0:
                print(
                    f"Epoch {epoch + 1 :03d} | "
                    f"{name:4s}"
                    f"loss={train_loss:.4f} | "
                    f"val_acc = {val_acc: .3f}"
                )
    # 5. 绘图
    epoch_numbers = np.arange(1, epochs + 1)
    fig, axes = plt.subplots(1,2,figsize=(12,4))

    for name, history in histories.items():
        axes[0].plot(
            epoch_numbers,
            history["loss"],
            label = name
        )
        axes[1].plot(
            epoch_numbers,
            history["val_acc"],
            label = name
        )

    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0,1.05)

    for ax in axes:
        ax.legend()
        ax.grid(alpha = 0.3)

    fig.tight_layout()

    # 6. 输出最终结果
    for name, history in histories.items():
        print(
            f"{name}:"
            f"final_loss = {history['loss'][-1]:.4f},"
            f"final_val_acc = {history['val_acc'][-1]:.3f}"
        )

    plt.show()

if __name__ == "__main__":
    main()

