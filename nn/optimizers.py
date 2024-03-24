from autograd import Module


class SGD:
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, module: Module) -> None:
        for param in module.parameters():
            param -= self.lr * param.grad
