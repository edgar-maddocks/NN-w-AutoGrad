from autograd import Tensor
import numpy as np


class Loss:
    def loss(self, y_true: Tensor, y_hat: Tensor) -> float:
        raise NotImplementedError

    def grad(self, y_true: Tensor, y_hat: Tensor) -> Tensor:
        raise NotImplementedError  #


class ASE(Loss):
    def loss(self, y_true: Tensor, y_hat: Tensor) -> Tensor:
        error = y_true - y_hat
        return (error * error).sum()
