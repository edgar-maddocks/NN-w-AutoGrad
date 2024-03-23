import numpy as np
from typing import List, NamedTuple, Callable, Union

Arrayable = Union[float, list, np.ndarray]


def to_array(x: Arrayable) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)


class Dependency(NamedTuple):
    tensor: "Tensor"
    grad_func: Callable[[np.ndarray], np.ndarray]

    def __repr__(self) -> str:
        return f"Dependency((tensor: {self.tensor}), grad_func: {self.grad_func})"


class Tensor:
    def __init__(
        self,
        data: np.ndarray,
        requires_grad: bool = False,
        dependencies: List[Dependency] = None,
    ) -> None:
        self.data = to_array(data)
        self.shape = self.data.shape

        self.requires_grad = requires_grad
        self.dependencies = dependencies or []

        self.grad = None
        self.grad_func = None

        if self.requires_grad:
            self.zero_grad()

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data))

    def backward(self, grad: "Tensor" = None) -> None:
        assert self.requires_grad, "backward() called on non-requires-grad Tensor"

        if grad is None:
            if self.shape == ():
                grad = Tensor(1)
            else:
                raise RuntimeError("Grad Tensor must be provided for non-zero Tensor")

        self.grad.data += grad.data

        for dependecy in self.dependencies:
            backward_grad = dependecy.grad_func(grad.data)
            dependecy.tensor.backward(Tensor(backward_grad))

    def sum(self):
        data = self.data.sum()
        requires_grad = self.requires_grad

        if requires_grad:

            def grad_func(grad: np.ndarray):
                return grad * np.ones_like(self.data)

            dependencies = [Dependency(self, grad_func)]
        else:
            dependencies = []

        return Tensor(data, requires_grad, dependencies)

    def __add__(self, other: "Tensor") -> "Tensor":
        data = self.data + other.data
        requires_grad = self.requires_grad or other.requires_grad

        dependencies = []

        if self.requires_grad:

            def grad_func1(grad: np.ndarray) -> np.ndarray:
                ## Need to manage broadcasting.
                ## i.e. (5,2) + (2,) still works as it transforms (2,) into (1,2) then (5,2)

                ## [[1, 2, 3], [4, 5, 6]] + [7, 8, 9]
                ## == [[1, 2, 3], [4, 5, 6]] + [[7, 8, 9], [7, 8, 9]]

                # if we do backward w respect t1 for Tensor [1, 1, 1] = [[1,1,1], [1,1,1]]
                # but if we do w respect t2, we should get [2, 2, 2]
                # because the originial [7, 8, 9] goes into the gradient for both [1, 2, 3] and [4, 5, 6]

                ndims_added = (
                    data.ndim
                    - self.data.ndim  # calculate the number of dimensions added
                )

                for dim in range(ndims_added):
                    grad = grad.sum(axis=0)

                # Also need to sum out broadcasted, but non-added dims. e.g (2, 3) + (1, 3)
                # (2, 3) + (1, 3) = (2, 3) => but we need to get back to (1,3)

                for i, dim in enumerate(
                    self.shape
                ):  # enumerate((2, 3) would produce 0, 2 then 1, 3
                    if dim == 1:  # Potential broadcast
                        grad = grad.sum(axis=i, keepdims=True)

                return grad

            dependencies.append(Dependency(self, grad_func1))

        if other.requires_grad:

            def grad_func2(grad: np.ndarray) -> np.ndarray:
                ndims_added = (
                    data.ndim
                    - other.data.ndim  # calculate the number of dimensions adde
                )

                for dim in range(ndims_added):
                    grad = grad.sum(axis=0)

                for i, dim in enumerate(other.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)

                return grad

            dependencies.append(Dependency(other, grad_func2))

        return Tensor(data, requires_grad, dependencies)

    def __neg__(t: "Tensor") -> "Tensor":
        data = -t.data
        requires_grad = t.requires_grad

        dependencies = []

        if requires_grad:
            dependencies.append(Dependency(t, lambda x: -x))

        return Tensor(data, requires_grad, dependencies)

    def __sub__(self, other: "Tensor") -> "Tensor":
        return self + -other
