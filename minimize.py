"""
    Use autograd to minimize x**2
"""

from tensor import Tensor
import random
import numpy as np

x = Tensor([random.randint(0, 10) for x in range(100)], requires_grad=True)


for i in range(100):
    sos = (x * x).sum()
    sos.backward()

    delta_x = Tensor(0.1) * x.grad
    x -= delta_x

    print(i, sos)
