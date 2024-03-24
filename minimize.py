"""
    Use autograd to minimize x**2
"""

from autograd import *
from nn2 import *
import numpy as np

x_data = Tensor(np.random.randn(100, 3))
coef = Tensor(np.array([-1, +3, -2], dtype=np.float64))
y_data = x_data @ coef + 5


class LinRegModule(Module):
    def __init__(self, x_shape):
        self.w = Parameter(x_shape)
        self.b = Parameter()

    def predict(self, x):
        return x @ self.w + self.b


class LinearRegression:
    def __init__(self):
        self.mod = None
        self.optim = SGD(lr=0.001)

    def _SoS(self, y_true, y_hat):
        error = y_true - y_hat
        return (error * error).sum()

    def fit(self, x_train, y_train, epochs: int = 100):
        self.mod = LinRegModule(x_train.shape[1])
        for epoch in range(epochs):
            self.mod.zero_grad()

            preds = self.mod.predict(x_train)

            loss = self._SoS(y_train, preds)

            print(loss)

            loss.backward()

            self.optim.step(self.mod)


model = LinearRegression()
model.fit(x_data, y_data)
