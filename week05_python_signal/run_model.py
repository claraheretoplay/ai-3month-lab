import numpy as np

from model import MLP
from losses import cross_entropy_loss
from losses import softmax_cross_entropy_backward

np.random.seed(42)

#5个样本，每个样本有2个特征
X = np.random.randn(5,2)

#3个分类标签
y = np.array([0,1,2,1,0])

model = MLP(
    in_features = 2,
    hidden_features = 4,
    num_classes = 3
)

for epoch in range(100):
    #1. 前向传播
    logits = model.forward(X)

    #2. 计算损失
    loss = cross_entropy_loss(logits,y)

    #3. 计算损失对Logits的梯度
    grad_logits = softmax_cross_entropy_backward(
        logits,
        y
    )

    #4. 反向传播
    model.backward(grad_logits)

    #5. 更新参数
    model.update(learning_rate = 0.1)

    if(epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1:03d},"
            f"loss = {loss:.4f}"
        )