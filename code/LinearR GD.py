import numpy as np
import matplotlib.pyplot as plt

m = 30
mean = [1, 2]
cov = [[1, -0.3], [-0.3, 2]]
y1 = np.random.multivariate_normal(mean, cov, m)
x1 = y1[:, 0]
y2 = y1[:, 1]

x2 = np.mat(x1).T
x3 = np.ones((m, 1))
X = np.hstack((x3, x2))
y = np.mat(y2).T
# alpha
alpha = 0.01
print(X)
print(y)


def gradient_descent(X, y, alpha):
    theta = np.array([1, 1]).reshape(2, 1)
    gradient = ((1. / m) * np.dot(np.mat(X).T, np.dot(X, theta) - y))
    while not np.all(np.absolute(gradient) <= 1e-5):
        theta = theta - alpha * gradient
        gradient = ((1. / m) * np.dot(np.mat(X).T, np.dot(X, theta) - y))
    return theta


[theta0, theta1] = gradient_descent(X, y, alpha)

Y = []
for i in range(0, 30):
    y3 = theta0 + theta1 * x1[i]
    y4 = y3.tolist()
    Y.append(y4[0])
plt.scatter(x1, y2)
plt.plot(x1, Y, color='r')
plt.show()
