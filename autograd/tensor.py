import numpy as np
from typing import List, NamedTuple, Callable, Union


Arrayable = Union[float, list, np.ndarray]
Tensorable = Union["Tensor", float, np.ndarray]


def to_tensor(x: Tensorable) -> "Tensor":
    if isinstance(x, Tensor):
        return x
    else:
        return Tensor(x)


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
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float64))

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

    def __sum_added_dims(
        self, new_tensor_data: np.ndarray, grad: np.ndarray, t: "Tensor"
    ) -> np.ndarray:
        ## Need to manage broadcasting.
        ## i.e. (5,2) + (2,) still works as it transforms (2,) into (1,2) then (5,2)

        ## [[1, 2, 3], [4, 5, 6]] + [7, 8, 9]
        ## == [[1, 2, 3], [4, 5, 6]] + [[7, 8, 9], [7, 8, 9]]

        # if we do backward w respect t1 for Tensor [1, 1, 1] = [[1,1,1], [1,1,1]]
        # but if we do w respect t2, we should get [2, 2, 2]
        # because the originial [7, 8, 9] goes into the gradient for both [1, 2, 3] and [4, 5, 6]

        ndims_added = (
            new_tensor_data.ndim
            - t.data.ndim  # calculate the number of dimensions added
        )

        for dim in range(ndims_added):
            grad = grad.sum(axis=0)

        return grad

    def __sum_broadcasted_dims(self, grad: np.ndarray, t: "Tensor"):
        # Also need to sum out broadcasted, but non-added dims. e.g (2, 3) + (1, 3)
        # (2, 3) + (1, 3) = (2, 3) => but we need to get back to (1,3)

        for i, dim in enumerate(
            t.shape
        ):  # enumerate((2, 3) would produce 0, 2 then 1, 3
            if dim == 1:  # Potential broadcast
                grad = grad.sum(axis=i, keepdims=True)

        return grad

    def __add__(self, other) -> "Tensor":
        return self._add(self, to_tensor(other))

    def __iadd__(self, other) -> "Tensor":
        self.data = self.data + to_tensor(other).data
        return self

    def __radd__(self, other) -> "Tensor":
        return self._add(to_tensor(other), self)

    def __neg__(self) -> "Tensor":
        return self._neg(self)

    def __sub__(self, other) -> "Tensor":
        return self._add(self, -(to_tensor(other)))

    def __isub__(self, other) -> "Tensor":
        self.data = self.data - to_tensor(other).data
        return self

    def __rsub__(self, other) -> "Tensor":
        return self._sub(to_tensor(other), self)

    def __mul__(self, other) -> "Tensor":
        return self._mul(self, to_tensor(other))

    def __imul__(self, other) -> "Tensor":
        self.data = self.data * to_tensor(other).data
        return self

    def __rmul__(self, other) -> "Tensor":
        return self._mul(to_tensor(other), self)

    def __matmul__(self, other) -> "Tensor":
        return self._matmul(self, to_tensor(other))

    def mean(self) -> "Tensor":
        return self._mean(to_tensor(self))

    def _add(self, t: "Tensor", other: "Tensor") -> "Tensor":

        data = t.data + other.data

        requires_grad = t.requires_grad or other.requires_grad

        dependencies = []

        if t.requires_grad:

            def grad_func1(grad: np.ndarray) -> np.ndarray:

                grad = t.__sum_added_dims(data, grad, t)

                grad = t.__sum_broadcasted_dims(grad, t)

                return grad

            dependencies.append(Dependency(t, grad_func1))

        if other.requires_grad:

            def grad_func2(grad: np.ndarray) -> np.ndarray:

                grad = t.__sum_added_dims(data, grad, other)

                grad = t.__sum_broadcasted_dims(grad, other)

                return grad

            dependencies.append(Dependency(other, grad_func2))

        return Tensor(data, requires_grad, dependencies)

    def _neg(self, t: "Tensor") -> "Tensor":

        data = -t.data

        requires_grad = t.requires_grad

        dependencies = []

        if requires_grad:

            dependencies.append(Dependency(t, lambda x: -x))

        return Tensor(data, requires_grad, dependencies)

    def _sub(self, t: "Tensor", other: "Tensor") -> "Tensor":

        return t + -other

    def _mul(self, t: "Tensor", other: "Tensor") -> "Tensor":

        data = t.data * other.data

        requires_grad = t.requires_grad or other.requires_grad

        dependencies = []

        if t.requires_grad:

            def grad_func1(grad: np.ndarray) -> np.ndarray:

                # y = a * b

                # dy/da = b

                grad = grad * other.data

                grad = t.__sum_added_dims(data, grad, t)

                grad = t.__sum_broadcasted_dims(grad, t)

                return grad

            dependencies.append(Dependency(t, grad_func1))

        if other.requires_grad:

            def grad_func2(grad: np.ndarray) -> np.ndarray:

                # dy/db = a

                grad = grad * t.data

                grad = t.__sum_added_dims(data, grad, other)

                grad = t.__sum_broadcasted_dims(grad, other)

                return grad

            dependencies.append(Dependency(other, grad_func2))

        return Tensor(data, requires_grad, dependencies)

    def _matmul(self, t: "Tensor", other: "Tensor") -> "Tensor":

        data = t.data @ other.data

        requires_grad = t.requires_grad or other.requires_grad

        dependencies = []

        if t.requires_grad:

            def grad_func1(grad: np.ndarray) -> np.ndarray:

                # y = A * B

                # dy/dA = grad @ B.T

                # No broadcasting for matmul so no need to sum out dims

                return grad @ other.data.T

            dependencies.append(Dependency(t, grad_func1))

        if other.requires_grad:

            def grad_func2(grad: np.ndarray) -> np.ndarray:

                # dy/dB = A.T @ grad

                return t.data.T @ grad

            dependencies.append(Dependency(other, grad_func2))

        return Tensor(data, requires_grad, dependencies)

    def _mean(self, t: "Tensor") -> "Tensor":
        data = np.mean(t.data)

        requires_grad = t.requires_grad

        dependencies = []

        if requires_grad:

            def grad_func(grad: np.ndarray) -> np.ndarray:
                return np.ones_like(grad) / np.prod(grad.shape)

            dependencies.append(Dependency(t, grad_func))

        return Tensor(data, requires_grad, dependencies)
