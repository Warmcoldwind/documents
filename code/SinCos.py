import math
import numpy as np
from matplotlib import pyplot as plt


def sigmoid(x):
    # Sigmoid 激活函数: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    # sigmoid 函数的导数: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)


data = np.linspace(-5, 5, 1000)
data_size = data.size
all_y_trues = np.zeros((data_size, 1))
for i in range(data_size):
    all_y_trues[i] = 1 / math.cos(data[i]) + 1 / math.sin(data[i])
y_pred = np.zeros((data_size, 1))  # 模型的输出结果


class OurNeuralNetwork:
    def __init__(self):
        # 权重，Weights
        self.w1 = np.random.random((80, 1))
        self.b1 = np.random.normal()
        # 截距项，Biases
        self.w2 = np.random.random((1, 80))
        self.b2 = np.random.normal()

    def train(self, data, all_y_trues):
        learn_rate = 0.005
        epochs = 20000  # number of times to loop through the entire dataset
        for epoch in range(epochs):
            er = 0
            for m in range(data_size):
                if abs(all_y_trues[m] - y_pred[m]) > 0.3:
                    er = 1
                    break
            if er == 1:
                for n in range(data_size):
                    # 正向传输
                    h1 = np.dot(self.w1, data[n]) + self.b1
                    o1 = np.dot(self.w2, sigmoid(h1)) + self.b2
                    y_pred[n] = o1
                    # 计算偏导数
                    # Naming: d_L_d_w1 represents "partial L / partial w1"
                    # Neuron o1
                    d_L_d_ypred = -2 * (all_y_trues[n] - y_pred[n])
                    d_ypred_d_w2 = sigmoid(h1)
                    d_ypred_d_b2 = 1
                    d_ypred_d_h1 = self.w2

                    # Neuron h1
                    d_h1_d_w1 = np.dot(data[n], deriv_sigmoid(h1))
                    d_h1_d_b1 = deriv_sigmoid(h1)

                    # 更新权值和偏值
                    # Neuron h1
                    self.w1 -= learn_rate * np.dot(d_ypred_d_h1, d_h1_d_w1) * d_L_d_ypred

                    self.b1 -= learn_rate * np.dot(d_ypred_d_h1, d_h1_d_b1) * d_L_d_ypred

                    # Neuron o1
                    self.w2 -= learn_rate * d_L_d_ypred * np.mat(d_ypred_d_w2).T
                    self.b2 -= learn_rate * d_ypred_d_b2 * d_L_d_ypred
            else:
                break


# Train our neural network!
network = OurNeuralNetwork()
network.train(data, all_y_trues)
# Make some predictions
plt.plot(data, all_y_trues)
plt.plot(data, y_pred, color='red', linestyle='--')
err = []
for i in range(0, 1000):
    a = all_y_trues[i] - y_pred[i]
    err.append(a)
print(err)
plt.show()
print(all_y_trues)
print(y_pred)
