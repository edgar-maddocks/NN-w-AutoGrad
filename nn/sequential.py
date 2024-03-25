from autograd import *
from nn.losses import Loss, MSE
from nn.optimizers import Optimizer, SGD
from typing import List
import time


class Sequential:
    def __init__(
        self, layers: List[Module], loss: Loss = MSE(), optimizer: Optimizer = SGD()
    ) -> None:
        self.layers = layers
        self.loss = loss
        self.optim = optimizer

        for layer in self.layers:
            for param in layer.parameters():
                self.optim.add_parameter(param)

    def __call__(self, x) -> Tensor:
        return self.predict(x)

    def predict(self, x) -> Tensor:
        x = Tensor(x)
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def fit(self, x_train, y_train, epochs: int = 100) -> None:
        start = time.time()
        for epoch in range(epochs):
            preds = self.predict(x_train)

            loss = self.loss(y_train, preds)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            print("|--------------------------------|")
            print(f"| Epoch: {epoch} | Loss : {round(float(loss.data),9)} |")
            print("|--------------------------------|")

        print(f"Total Elapsed: {time.time() - start}")
