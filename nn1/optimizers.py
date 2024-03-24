from network import Sequential


class Optimizer:
    def step(self, net: Sequential) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, net: Sequential) -> None:
        for param, grad in net.get_params_grads():
            param.data -= self.lr * grad.data
