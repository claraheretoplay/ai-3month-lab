import numpy as np

from losses import cross_entropy_loss
from model import MLP
from losses import softmax_cross_entropy_backward

def numerical_gradient(model,X,y,epsilon=1e-5):
    numerical_grad = np.zeros_like(model.linear1.W)

    for i in range(model.linear1.W.shape[0]):
        for j in range(model.linear1.W.shape[1]):
            old_value = model.linear1.W[i,j]

            model.linear1.W[i,j] = old_value + epsilon
            loss_plus = cross_entropy_loss(
                model.forward(X),
                y
            )

            model.linear1.W[i,j] = old_value - epsilon
            loss_minus = cross_entropy_loss(
                model.forward(X),
                y
            )

            numerical_grad[i,j] = (
                loss_plus - loss_minus
            ) / (2 * epsilon)

            model.linear1.W[i,j] = old_value

    return numerical_grad

def main():
    np.random.seed(42)

    X = np.random.randn(4,2)
    y = np.array([0,1,2,1])

    model = MLP(
        in_features = 2,
        hidden_features = 3,
        num_classes = 3
    )

    logits = model.forward(X)

    grad_logits = softmax_cross_entropy_backward(
        logits,
        y
    )

    model.backward(grad_logits)

    numerical_grad = numerical_gradient(
        model,
        X,
        y
    )

    backward_grad = model.linear1.dW
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
