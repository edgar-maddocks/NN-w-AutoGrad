from autograd import Module, Parameter
from typing import List


class Optimizer:
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr
        self.params = []

    def step(self) -> None:
        raise NotImplementedError

    def zero_grad(self) -> None:
        for param in self.params:
            param.zero_grad()


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01, l2_reg_coeff: float = 0.0) -> None:
        super().__init__(lr)
        self.l2_reg_coeff = l2_reg_coeff

    def step(self) -> None:
        for param in self.params:
            param -= self.lr * (param.grad + 2 * (self.l2_reg_coeff * param.data))

    def add_parameter(self, parameter: Parameter) -> None:
        self.params.append(parameter)
