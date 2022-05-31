import matplotlib.pyplot as plt
from numpy import *
import numpy as np

Er = 0
dot_num0 = 50
y = []
mean0 = [2, 1]
cov0 = [[0.5, 0.3], [0.3, 1]]
a = random.multivariate_normal(mean0, cov0, dot_num0)
x0 = a[:, 0]
y0 = a[:, 1]
X1 = []
for i in range(50):
    X1.append([x0[i], y0[i]])
X = np.array(X1)
miu = np.sum(X, axis=0)
miu1 = miu[0] / dot_num0
miu2 = miu[1] / dot_num0
for i in range(50):
    X[i][0] = X[i][0] - miu1
    X[i][1] = X[i][1] - miu2
delta = 1 / dot_num0 * np.dot(np.mat(X).T, X)
w, v = np.linalg.eig(delta)
n = 0
m = 0
for i in range(len(w)):
    if w[i] > w[0]:
        n = i
    else:
        m = i
er = 0
k = v[:, n][0, 0] / v[:, n][1, 0]
for i in range(dot_num0):
    y.append(x0[i] * k)
for i in range(dot_num0):
    er = er + (X[i][0] ** 2 + X[i][1] ** 2) - (
            np.dot(np.dot(np.mat(v[:, n]).T, X[i])[0, 0], v[:, n])[0, 0] ** 2 +
            np.dot(np.dot(np.mat(v[:, n]).T, X[i])[0, 0], v[:, n])[1, 0] ** 2)
Er = Er + er
print(Er / dot_num0)
plt.scatter(x0, y0, c='r')
plt.plot(x0, y, c='b')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
