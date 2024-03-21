import numpy as np
from scipy.signal import correlate2d, convolve
from tensor import Tensor
from typing import Dict, Callable


class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}
        self.input = None
        self.output = None

    def forward(self, input: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, output_grad: Tensor) -> Tensor:
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, n_features: int, n_nodes: int) -> None:
        super().__init__()
        # input is shape (samples, features)
        # output is shape (samples, outputs)
        self.params["w"] = Tensor(np.random.randn(n_features, n_nodes))
        self.params["b"] = Tensor(np.random.randn(n_nodes))

        self.layer_type = "Dense"

    def forward(self, input: Tensor) -> Tensor:
        self.input = input
        return Tensor(np.dot(input.data, self.params["w"].data) + self.params["b"].data)

    def backward(self, output_grad: Tensor) -> Tensor:
        self.grads["b"] = Tensor(np.sum(output_grad.data, axis=0))
        self.grads["w"] = Tensor(np.dot(self.input.data.T, output_grad.data))
        return Tensor(np.dot(output_grad.data, self.params["w"].data.T))


class Convolutional(Layer):
    def __init__(
        self, input_shape: tuple[int], kernel_size: int, n_kernels: int
    ) -> None:
        input_depth, input_width, input_height = input_shape
        self.input_shape = input_shape
        self.input_depth = input_depth

        self.depth = n_kernels
        self.output_shape = (
            self.depth,
            (input_height - kernel_size + 1),
            (input_height - kernel_size + 1),
        )
        self.kernels_shape = (self.depth, self.input_depth, kernel_size, kernel_size)

        self.params["w"] = Tensor(np.random.randn(*self.kernels_shape))
        self.params["b"] = Tensor(np.random.randn(*self.output_shape))

    def forward(self, input: Tensor) -> Tensor:
        self.output = np.copy(self.params["b"].data)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += correlate2d(
                    self.input.data[j], self.params["w"].data[i, j], "valid"
                )
        return Tensor(self.output)


F = Callable[[Tensor], Tensor]


class Activation(Layer):
    def __init__(self, func: F, grad_func: F):
        super().__init__()
        self.func = func
        self.grad_func = grad_func

    def forward(self, input: Tensor) -> Tensor:
        self.input = input
        return self.func(input)

    def backward(self, output_grad: Tensor) -> Tensor:
        return Tensor(self.grad_func(self.input).data * output_grad.data)


class Tanh(Activation):
    def __init__(self):

        def tanh(t: Tensor) -> Tensor:
            return Tensor(np.tanh(t.data))

        def d_tanh(t: Tensor) -> Tensor:
            return Tensor((1 - (np.tanh(t.data) ** 2)))

        super().__init__(tanh, d_tanh)
