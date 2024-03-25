import numpy as np
from autograd.tensor import *


class Parameter(Tensor):
    def __init__(self, shape: tuple) -> None:
        data = np.random.randn(*shape)
        super().__init__(data, requires_grad=True)
