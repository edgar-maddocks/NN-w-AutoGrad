import numpy as np


class Tensor:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.shape = data.shape

    def __repr__(self) -> str:
        return f"{self.data}"
