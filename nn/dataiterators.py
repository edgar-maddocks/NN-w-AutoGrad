import numpy as np
from tensor import Tensor
from typing import Iterator, NamedTuple

Batch = NamedTuple("Batch", [("inputs", Tensor), ("targets", Tensor)])


class DataIterator:
    def __call__(self, input: Tensor, targets: Tensor) -> Iterator[Batch]:
        raise NotImplementedError


class BatchIterator(DataIterator):
    def __init__(self, batch_size: int = 32, shuffle: bool = True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, input: Tensor, targets: Tensor) -> Iterator[Batch]:
        indicies = np.arange(0, len(input.data), self.batch_size)
        if self.shuffle:
            np.random.shuffle(indicies)

        for start_idx in indicies:
            end_idx = start_idx + self.batch_size
            input_batch = input.data[start_idx:end_idx]
            targets_batch = targets.data[start_idx:end_idx]
            yield Batch(Tensor(input_batch), Tensor(targets_batch))
