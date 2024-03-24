import numpy as np
from autograd.tensor import Tensor


class Loss:
    def loss(self, y_true: Tensor, y_hat: Tensor) -> float:
        raise NotImplementedError

    def grad(self, y_true: Tensor, y_hat: Tensor) -> Tensor:
        raise NotImplementedError


class MSE(Loss):
    def loss(self, y_true: Tensor, y_hat: Tensor) -> float:
        return np.mean(np.power(y_true.data - y_hat.data, 2))

    def grad(self, y_true: Tensor, y_hat: Tensor) -> Tensor:
        return 2 * (y_hat.data - y_true.data) / np.size(y_true.data)
