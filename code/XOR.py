import numpy as np


def sigmoid(x):
    # Sigmoid 激活函数: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    # sigmoid 函数的导数: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(y_true, y_pred):
    # y_true 为真实值 y_pred 为预测值，二者长度相同.
    return ((y_true - y_pred) ** 2).mean()


class OurNeuralNetwork:
    def __init__(self):
        # 权重，Weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # 截距项，Biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        # x is a numpy array with 2 elements.
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data, all_y_trues):
        learn_rate = 0.1
        epochs = 100000  # number of times to loop through the entire dataset
        for epoch in range(epochs):
            y_preds = np.apply_along_axis(self.feedforward, 1, data)
            er = 0
            for i in range(0, 4):
                if abs(all_y_trues[i] - y_preds[i]) > 0.02:
                    er = 1
            if er == 1:
                for x, y_true in zip(data, all_y_trues):  # 两个对象中的元素一一对应
                    # 正向
                    sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                    h1 = sigmoid(sum_h1)

                    sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                    h2 = sigmoid(sum_h2)

                    sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                    o1 = sigmoid(sum_o1)
                    y_pred = o1

                    # 计算偏导数
                    # Naming: d_L_d_w1 represents "partial L / partial w1"
                    d_L_d_ypred = -2 * (y_true - y_pred)

                    # Neuron o1
                    d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                    d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                    d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                    d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                    d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                    # Neuron h1
                    d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                    d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                    d_h1_d_b1 = deriv_sigmoid(sum_h1)

                    # Neuron h2
                    d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                    d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                    d_h2_d_b2 = deriv_sigmoid(sum_h2)

                    # 更新权重和偏值
                    # Neuron h1
                    self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                    self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                    self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                    # Neuron h2
                    self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                    self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                    self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                    # Neuron o1
                    self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                    self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                    self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3
            else:
                break



# Define dataset
data = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])
all_y_trues = np.array([[0, 1, 1, 0]]).T

# Train our neural network!
network = OurNeuralNetwork()
network.train(data, all_y_trues)
print(network.feedforward(np.array([0, 0])))
print(network.feedforward(np.array([1, 0])))
print(network.feedforward(np.array([0, 1])))
print(network.feedforward(np.array([1, 1])))
