from network import Sequential, fit
from layers import Dense, Tanh
from losses import MSE
from optimizers import SGD
from tensor import Tensor

# from train import fit

import numpy as np

x = Tensor(np.array([[0, 0], [1, 0], [0, 1], [1, 1]]))
y = Tensor(np.array([[0], [1], [1], [0]]))

model = Sequential([Dense(2, 3), Tanh(), Dense(3, 1), Tanh()])

fit(model, x, y, epochs=10000, optimizer=SGD(lr=0.1))
