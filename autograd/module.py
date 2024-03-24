from typing import Iterator
import inspect

from autograd.tensor import Tensor
from autograd.parameter import Parameter


class Module:
    def parameters(self) -> Iterator[Parameter]:
        for name, val in inspect.getmembers(self):
            if isinstance(val, Parameter):
                yield val
            elif isinstance(val, Module):
                yield from val.parameters()

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()
