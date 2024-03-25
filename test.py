from nn.sequential import Sequential
from nn.layers import Dense

x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [0]]

model = Sequential([Dense(2, 3), Dense(3, 1)])

model.fit(x, y, epochs=10)
print(model(x))
