from network import Sequential
from optimizers import Optimizer, SGD
from layers import Layer
from tensor import Tensor
from losses import Loss, MSE
from dataiterators import DataIterator, BatchIterator


def fit(
    network: Sequential,
    train_inputs: Tensor,
    train_targets: Tensor,
    loss: Loss = MSE(),
    optimizer: Optimizer = SGD(),
    epochs: int = 100,
    iterator: DataIterator = BatchIterator(),
) -> None:
    for epoch in range(epochs):
        ep_loss = 0.0
        for batch in iterator(train_inputs, train_targets):
            y_hat = network.forward(batch.inputs)
            ep_loss += loss.loss(batch.targets, y_hat)
            grad = loss.grad(batch.targets, y_hat)
            network.backward(grad)
            optimizer.step(network)
        print(f"Epoch: {epoch}     Loss: {ep_loss}", end="\r")
    print("#################### TRAINING COMPLETE ####################")
