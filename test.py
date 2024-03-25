from nn.sequential import Sequential
from nn.optimizers import SGD
from nn.layers import Dense, Tanh
from nn.losses import MSE

x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [0]]

model = Sequential(
    [Dense(2, 3), Tanh(), Dense(3, 1), Tanh()],
    loss=MSE(),
    optimizer=SGD(lr=0.1),
)

model.fit(x, y, epochs=200)
print(model(x))
