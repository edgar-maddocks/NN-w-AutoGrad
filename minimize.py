"""
    Use autograd to minimize x**2
"""

from autograd.tensor import Tensor, Tensorable
import numpy as np


x_data = Tensor(np.random.randn(100, 3))
coef = Tensor(np.array([-1, +3, -2], dtype=np.float64))
y_data = x_data @ coef + 5


class LinearRegression:
    def __init__(self, lr: float = 0.001):
        self.lr = lr

        self.w = None
        self.b = None

    def _SoS(self, true: Tensor, preds: Tensor) -> Tensor:
        error = true - preds
        return (error * error).sum()

    def fit(self, x_train: Tensorable, y_train: Tensorable, epochs: int = 100):
        self.w = Tensor(np.random.randn(x_train.shape[1]), requires_grad=True)
        self.b = Tensor(np.random.randn(), requires_grad=True)

        for epoch in range(epochs):
            self.w.zero_grad()
            self.b.zero_grad()
            preds = x_data @ self.w + self.b

            loss = self._SoS(y_train, preds)

            print(loss)

            loss.backward()

            self.w -= self.lr * self.w.grad
            self.b -= self.lr * self.b.grad


model = LinearRegression()
model.fit(x_data, y_data)
