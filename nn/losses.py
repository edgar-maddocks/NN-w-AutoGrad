from autograd import Tensor
import numpy as np


class Loss:
    def loss(self, y_true: Tensor, y_hat: Tensor) -> float:
        raise NotImplementedError

    def __call__(self, y_true, y_hat):
        return self.loss(y_true, y_hat)


class ASE(Loss):
    def loss(self, y_true: Tensor, y_hat: Tensor) -> Tensor:
        error = y_true - y_hat
        return (error * error).sum()


class MSE(Loss):
    def loss(self, y_true: Tensor, y_hat: Tensor) -> Tensor:
        error = y_true - y_hat
        return (error * error).mean()
