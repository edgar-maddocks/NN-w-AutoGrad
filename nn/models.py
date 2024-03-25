from autograd import *
from typing import List


class Sequential:
    def __init__(self, layers: List[Module]):
        self.layers = layers

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
