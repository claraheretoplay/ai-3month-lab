import numpy as np

def softmax(logits):
    """
    logits.shape:(batch_size,numclasses)
    返回每个样本术语各类别的概率
    """
    #减去每行最大值，避免exp溢出
    shifted_logits = logits - np.max(
        logits,
        axis = 1,
        keepdims = True
    )

    exp_logits = np.exp(shifted_logits)
    probabilities = exp_logits / np.sum(
        exp_logits,
        axis = 1,
        keepdims = True
    )

    return probabilities

def cross_entropy_loss(logits,y_true):
    """
    logits.shape:(batch_size,num_classes)
    y_true.shape:(batch_size,)
    y_true中保存类别编号,例如0、1、2
    """
    probabilities = softmax(logits)

    batch_size = logits.shape[0]

    #取出每个样本真实类别对应的概率
    correct_probabilities = probabilities[
        np.arange(batch_size),
        y_true
    ]

    #交叉熵
    loss = -np.mean(np.log(correct_probabilities + 1e-12))

    return loss

def softmax_cross_entropy_backward(logits,y_true):
    """
    计算Softmax + 交叉熵对logits的梯度
    返回值形状与logits相同
    """
    probabilities = softmax(logits).copy()
    batch_size = logits.shape[0]

    #构造one-hot标签
    probabilities[
        np.arange(batch_size),
        y_true
    ] -= 1

    #对batch求平均
    grad_logits = probabilities / batch_size

    return grad_logits