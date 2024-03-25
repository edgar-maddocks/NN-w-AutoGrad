from network import Sequential, fit
from layers import Dense, Tanh
from losses import MSE
from optimizers import SGD
from autograd.tensor import Tensor
import numpy as np

x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y = np.array([[0], [1], [1], [0]])

model = Sequential([Dense(2, 3), Tanh(), Dense(3, 1), Tanh()])

fit(model, x, y, epochs=1000, optimizer=SGD(lr=0.1))

print(model.forward(x))