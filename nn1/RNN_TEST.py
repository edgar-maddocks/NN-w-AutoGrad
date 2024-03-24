import numpy as np

np.random.seed(0)

weights_i = np.random.randn(1, 5)

weights_h = np.random.randn(5, 5)
bias_h = np.random.randn(1, 5)

weights_o = np.random.randn(5, 1)
bias_o = np.random.randn(1, 1)

temps = np.array([66.0, 70.0, 72.0])
x0 = temps[0].reshape(1, 1)
x1 = temps[1].reshape(1, 1)
x2 = temps[2].reshape(1, 1)

outputs = np.zeros(3)

hiddens = np.zeros((3, 5))

prev_hidden = False
