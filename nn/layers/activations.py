from autograd import Module, Tensor, Dependency
import numpy as np


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        data = np.maximum(0, x.data)

        requires_grad = x.requires_grad

        dependencies = []

        if requires_grad:

            def d_ReLU(grad: np.ndarray) -> np.ndarray:
                grad = np.where(grad <= 0, 0, 1)
                return grad

            dependencies.append(Dependency(x, d_ReLU))

        return Tensor(data, requires_grad, dependencies)


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        data = np.tanh(x.data)

        requires_grad = x.requires_grad

        dependencies = []

        if requires_grad:

            def d_tanh(grad: np.ndarray) -> np.ndarray:
                grad = grad * (1 - (data * data))
                return grad

            dependencies.append(Dependency(x, d_tanh))

        return Tensor(data, requires_grad, dependencies)
