from numpy import array, exp, random, dot

X = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
y = array([[0, 1, 1, 0]]).T

random.seed(1)

weights = 2 * random.random((3, 1)) - 1

for it in range(10000):
    output = 1/(1+exp(-dot(X, weights)))
    error = y - output
    delta = error * output * (1-output)
    weights += dot(X.T, delta)

print(weights)
print(1/(1+exp(-dot([[1, 0, 0]], weights))))
