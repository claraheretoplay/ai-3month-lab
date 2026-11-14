import numpy as np

class SGD:
    def __init__(self,learning_rate = 0.05):
        self.learning_rate = learning_rate

    def step(self,params,grads):
        """params、grads 都是字典,使用相同的参数名称"""

        for name in params:
            params[name] -= self.learning_rate * grads[name]

class Adam:
    def __init__(
            self,
            learning_rate = 0.01,
            beta1 = 0.9,
            beta2 = 0.999,
            epsilon = 1e-8
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # 保存每个参数的一阶、二阶矩估计
        self.m = {}
        self.v = {}

        # 参数更新次数，不是epoch数
        self.t = 0

    def step(self,params,grads):
        # 一次step更新所欲参数，t只加一次
        self.t += 1

        for name in params:
            grad = grads[name]

            # 第一次更新时，初始化历史状态
            if name not in self.m:
                self.m[name] = np.zeros_like(params[name])
                self.v[name] = np.zeros_like(params[name])

            # 1. 一阶矩：梯度的指数移动平均
            self.m[name] = (
                self.beta1 * self.m[name]
                +(1 - self.beta1) * grad
            )

            # 2. 二阶矩：梯度平方的指数移动平均
            self.v[name] = (
                self.beta2 * self.v[name]
                + (1 - self.beta2) * grad ** 2
            )

            # 3. 偏差修正
            m_hat = self.m[name] / (
                1 - self.beta1 ** self.t
            )
            v_hat = self.v[name] / (
                1 - self.beta2 ** self.t
            )

            # 4. 更新参数
            params[name] -= (
                self.learning_rate
                * m_hat
                / (np.sqrt(v_hat) + self.epsilon)
            )