import numpy as np

from losses import cross_entropy_loss
from model import MLP
from losses import softmax_cross_entropy_backward

def numerical_gradient(model,X,y,epsilon=1e-5):
    """
    使用中心差分计算linear1.W的数值梯度
    """
    numerical_grad = np.zeros_like(model.linear1.W)

    for i in range(model.linear1.W.shape[0]):
        for j in range(model.linear1.W.shape[1]):
            old_value = model.linear1.W[i,j]

            #W[i,j] + epsilon
            model.linear1.W[i,j] = old_value + epsilon
            loss_plus = cross_entropy_loss(
                model.forward(X),
                y
            )

            #W[i,j] - epsilon
            model.linear1.W[i,j] = old_value - epsilon
            loss_minus = cross_entropy_loss(
                model.forward(X),
                y
            )

            # 中心差分
            numerical_grad[i,j] = (
                loss_plus - loss_minus
            ) / (2 * epsilon)

            # 恢复原参数
            model.linear1.W[i,j] = old_value

    return numerical_grad

def main():
    np.random.seed(42)

    # 少量样本，便于梯度检查
    X = np.random.randn(4,2)
    y = np.array([0,1,2,1])

    model = MLP(
        in_features = 2,
        hidden_features = 3,
        num_classes = 3
    )

    # 1. 前向传播
    logits = model.forward(X)

    # 2. 计算损失对Logits的梯度
    grad_logits = softmax_cross_entropy_backward(
        logits,
        y
    )

    # 3. 手动反向传播
    model.backward(grad_logits)

    # 4. 计算W1的数值梯度
    numerical_grad = numerical_gradient(
        model,
        X,
        y
    )

    # 5. 读取反向传播得到的W1梯度
    backward_grad = model.linear1.dW

    # 6. 比较误差
    difference = np.mean(
        np.abs(
            numerical_grad - backward_grad
        )
    )

    print("numerical_grad:")
    print(numerical_grad)

    print("\nbackward_grad:")
    print(backward_grad)

    print("\n平均绝对误差:",difference)

    if difference < 1e-6:
        print("梯度检查通过")
    else:
        print("梯度检查未通过,请检查backward实现")

if __name__ == "__main__":
    main()
