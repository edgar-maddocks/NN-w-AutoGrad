from autograd import *


class Dense(Module):
    def __init__(self, n_features, n_nodes):
        self.w = Parameter((n_features, n_nodes))
        self.b = Parameter((n_nodes,))

    def forward(self, x):
        return x @ self.w + self.b
