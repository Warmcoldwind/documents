import matplotlib.pyplot as plt
from numpy import *
import numpy as np

dot_num = 100
mean = [0, 0]
cov = [[1, 0], [0, 1]]
a = random.multivariate_normal(mean, cov, dot_num)
x0 = a[:30, 0]
y0 = a[:30, 1]
x1 = a[30:100, 0]
y1 = a[30:100, 1]

mean1 = [1, 2]
cov1 = [[1, 0], [0, 2]]
b = random.multivariate_normal(mean1, cov1, dot_num)
x2 = b[:30, 0]
y2 = b[:30, 1]
x3 = b[30:100, 0]
y3 = b[30:100, 1]

xa = np.append(x1, x3)
xa_ = np.append(x0, x2)
y = np.append(y1, y3)
y_ = np.append(y0, y2)
x4 = np.ones((140, 1))
x4_ = np.ones((60, 1))
X = np.hstack((x4, np.mat(xa).T, np.mat(y).T))
X_ = np.hstack((x4_, np.mat(xa_).T, np.mat(y_).T))

rate = 0.1
Theta = []
B= []
for m in range(0, 20):
    C = []
    theta = np.array([0, 0, 0]).reshape(3, 1)
    for n in range(0, 140):
        c = np.random.randint(0, 140)
        while c in C:
            c = np.random.randint(0, 140)
        C.append(c)
    B.append(C)
    for i in range(0, 1000):
        for j in range(0, 140):
            if C[j] < 70:
                if np.dot(np.mat(theta).T, np.mat(X[C[j]]).T) < 0:
                    theta = theta + rate * np.mat(X[C[j]]).T
                else:
                    theta = theta
            else:
                if np.dot(np.mat(theta).T, np.mat(X[C[j]]).T) >= 0:
                    theta = theta - rate * np.mat(X[C[j]]).T
                else:
                    theta = theta
    Theta.append(theta)
Er = []
for i in range(0, 20):
    er = 0
    for j in range(0, 60):
        if j < 30:
            if np.dot(np.mat(Theta[i]).T, np.mat(X_[j]).T) < 0:
                er = er + 1
        else:
            if np.dot(np.mat(Theta[i]).T, np.mat(X_[j]).T) >= 0:
                er = er + 1
    Er.append([Theta[i], er])
Er1 = sorted(Er, key=(lambda x: x[1]))
print(Er1[0][1]/60)
Y1 = []
for i in range(0, 60):
    y3 = (Er1[0][0][0][0] + Er1[0][0][1][0] * xa_[i]) / (-1 * Er1[0][0][2][0])
    Y1.append(y3.tolist()[0][0])

plt.scatter(x0, y0)
plt.scatter(x2, y2)
plt.plot(xa_, Y1, 'y')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()