from autograd import Module


class Optimizer:
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, module: Module) -> None:
        raise NotImplementedError

    def zero_grad(self, module: Module) -> None:
        for param in module.parameters():
            param.zero_grad()


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        super().__init__(lr)

    def step(self, module: Module) -> None:
        for param in module.parameters():
            param -= self.lr * param.grad
