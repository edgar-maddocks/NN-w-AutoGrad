from layers import Layer
from tensor import Tensor
from typing import List, Iterator, Tuple, Union
import numpy as np

import os
import time

Arrayable = Union[Tensor, np.ndarray, list]


class Sequential:
    def __init__(self, layers: List[Layer]):
        self.layers = layers

    def forward(self, input: Tensor) -> Tensor:
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, grad: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def get_params_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad

    def evaluate(self, val_inputs, val_targets):
        raise NotImplementedError

    def __call__(self, input: Arrayable) -> Tensor:
        input_tensor = self.to_tensor(input)
        return self.forward(input_tensor)

    def to_tensor(self, input: Arrayable) -> Tensor:
        if isinstance(input, Tensor):
            return input
        elif isinstance(input, Arrayable):
            if isinstance(input, list):
                return Tensor(np.array(input))
            elif isinstance(input, np.ndarray):
                return Tensor(input)


from optimizers import Optimizer, SGD
from losses import Loss, MSE
from dataiterators import DataIterator, BatchIterator


def fit(
    network: Sequential,
    train_inputs: Arrayable,
    train_targets: Arrayable,
    loss: Loss = MSE(),
    optimizer: Optimizer = SGD(),
    epochs: int = 1000,
    iterator: DataIterator = BatchIterator(),
) -> None:
    start_time = time.time()
    train_inputs = network.to_tensor(train_inputs)
    train_targets = network.to_tensor(train_targets)
    for epoch in range(epochs):
        ep_loss = 0.0
        for batch in iterator(train_inputs, train_targets):
            y_hat = network.forward(batch.inputs)
            ep_loss += loss.loss(batch.targets, y_hat)
            grad = loss.grad(batch.targets, y_hat)
            network.backward(grad)
            optimizer.step(network)
        print(
            f"Epoch: {epoch}     Loss: {round(ep_loss, 9)}     Elapsed: {round(time.time() - start_time, 3)}s",
            end="\r",
        )
    os.system("cls" if os.name == "nt" else "clear")
    print("#################### TRAINING COMPLETE ####################")
    print(f"Total time elapsed: {round(time.time() - start_time, 3)}s")
    print("###########################################################")
