import matplotlib.pyplot as plt
import numpy as np

alpha = 0.01
dot_num = 50
mean = [0, 0]
cov = [[1, 0], [0, 1]]
y0 = np.random.multivariate_normal(mean, cov, dot_num)
x1 = y0[:, 0]
y2 = y0[:, 1]

dot_num1 = 50
mean1 = [1, 2]
cov1 = [[1, -0.3], [-0.3, 2]]
y1 = np.random.multivariate_normal(mean1, cov1, dot_num1)
x11 = y1[:, 0]
y12 = y1[:, 1]
x = np.append(x1, x11)
y = np.append(y2, y12)
x2 = np.ones((100, 1))
X = np.hstack((x2, np.mat(x).T, np.mat(y).T))
theta = np.array([1, 1, 1]).reshape(3, 1)
for j in range(100):
    Y1 = []
    for i in range(0, 100):
        if np.dot(np.mat(theta).T, np.mat(X[i]).T) > 0:
            Y1.append(1)
        else:
            Y1.append(0)
    er = 0
    Y = np.mat(Y1).T
    # print(Y)
    gradient = (1. / 50) * np.dot((1 / (1 + np.exp(-np.dot(np.mat(theta).T, np.mat(X).T))) - np.mat(Y).T), X)
    for i in range(0, 50):
        if Y[i] != 0:
            er = er + 1
    for i in range(50, 100):
        if Y[i] != 1:
            er = er + 1
    if er > 1:
        theta = theta - alpha * np.mat(gradient).T
        gradient = (1. / 50) * np.dot((1 / (1 + np.exp(-np.dot(np.mat(theta).T, np.mat(X).T))) - np.mat(Y).T), X)
print(theta)
theta1 = theta.tolist()
print(theta1)
Y1 = []
for i in range(0, 100):
    y3 = theta1[0][0] + theta1[1][0] * x[i] / (-1 * theta1[2][0])
    Y1.append(y3)
plt.scatter(y0[:, 0], y0[:, 1], c='r')
plt.scatter(y1[:, 0], y1[:, 1], c='b')
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(x, Y1, color='r')
plt.show()
