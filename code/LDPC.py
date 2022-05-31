import math
from numpy import random
import matplotlib.pyplot as plt
import numpy
import datetime
import time


def number_of_certain_probability(sequence, pro):  # 给定概率p产生随机数的函数
    item = 0
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(sequence, pro):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item


# 为了构造Tanner graph设的两个二维数组
graph = []  # 用来存储v_nodes和c_nodes对应关系的二维数组，共3000行，每一行4个元素，分别位v_nodes的行、列和c_nodes的行、列
re = []
num = numpy.zeros(1000)
c1_nodes = numpy.zeros([500, 6])
v1_nodes = numpy.zeros([1000, 3])
start1 = time.time()
for i in range(500):  # 随机构造Tanner graph函数
    j = 0
    while j < 6:
        a = random.randint(0, 1000)
        b = random.randint(0, 3)
        if v1_nodes[a][b] != -1 and a not in re and num[a] < 6:  # 排除掉已经和v_nodes有对应关系的c_nodes和v_nodes被占用的小节点
            graph.append([i, j, a, b])  # 将对应关系加入graph
            v1_nodes[a][b] = -1  # 定义Tanner graph
            num[a] += 1  # variable nodes 被占用6次即被排除
            re.append(a)
            j += 1
    re = []
end1 = time.time()
print(end1-start1)
print(graph)
print(len(graph))
graph1 = numpy.array(graph)
data = graph1[graph1[:, 2].argsort()]  # 按照第3列对行排序
graph2 = data.tolist()
graph3 = []
print(graph2)
print(len(graph2))
for i in range(1000):
    for j in range(i * 3, (i + 1) * 3):
        if graph2[j][3] == 0:
            graph3.append(graph2[j])
    for j in range(i * 3, (i + 1) * 3):
        if graph2[j][3] == 1:
            graph3.append(graph2[j])
    for j in range(i * 3, (i + 1) * 3):
        if graph2[j][3] == 2:
            graph3.append(graph2[j])

print(graph3)
P = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
P_false = []  # 存放15个解码出错率
for e in range(15):
    start = datetime.datetime.now()
    false = 0  # 解码n次出错的次数
    p = P[e]
    value_list = [0, 1]
    probability = [1 - p, p]  # 给上面给定概率产生随机数的函数赋值p
    code = []  # 存放n个经过bsc的长度为1000的码
    for t in range(1000):
        re_code = []  # 经过bsc接收到的码
        for i in range(1000):  # 利用随机数函数生成通过bsc后的码
            result = number_of_certain_probability(value_list, probability)
            if result == 1:
                re_code.append(1)
            elif result == 0:
                re_code.append(0)
        code.append(re_code)
        # print(re_code)
    # print(code)
    L_U0 = []  # 存放每个variable node的L(u0)
    for time in range(1000):  # 解码n次
        result1 = []
        v_nodes = numpy.zeros([1000, 3])
        c_nodes = numpy.zeros([500, 6])
        for i in range(1000):  # 计算每个variable node的L(u0)
            if code[time][i] == 0:
                L_U0.append(math.log(p / (1 - p)))
            else:
                L_U0.append(math.log((1 - p) / p))
        for i in range(1000):
            for j in range(3):
                v_nodes[i][j] = L_U0[i]
        # print(L_U0)
        # print(v_nodes)
        for q in range(49):  # 迭代50次
            for d in range(3000):  # variable nodes 向 check nodes 传递
                c_nodes[graph[d][0]][graph[d][1]] = v_nodes[graph[d][2]][graph[d][3]]
            for i in range(500):
                for j in range(6):
                    mul = 1.
                    for m in range(6):
                        if m != j:
                            mul = mul * math.tanh(v_nodes[graph[i * 6 + m][2]][graph[i * 6 + m][3]] / 2)
                    c_nodes[i][j] = 2 * math.atanh(max(mul, -0.99999999999))
            for n in range(3000):  # check nodes 向 variable nodes 传递
                v_nodes[graph[n][2]][graph[n][3]] = c_nodes[graph[n][0]][graph[n][1]]
            for i in range(1000):
                for j in range(3):
                    jia = 0
                    for m in range(3):
                        if m != j:
                            jia = jia + c_nodes[graph3[i * 3 + m][0]][graph3[i * 3 + m][1]]
                    v_nodes[i][j] = jia + L_U0[i]
            # print(v_nodes)
            # print(c_nodes)

        for m in range(1000):
            final = 0
            for n in range(3):
                final = final + v_nodes[m][n]
            result1.append(final + L_U0[m])
        for i in range(1000):
            if result1[i] > 0:
                print("False")
                break
        v_node = numpy.zeros(1000)
        for i in range(1000):  # 根据解出的L(v)判断是1还是0
            if result1[i] > 0:
                v_node[i] = 1
        # print(v_node)
        for i in range(1000):  # 判断解码是否正确
            if v_node[i] != 0:
                false += 1  # 若解码不正确则false+1
                break
    P_false.append(false / 1000)
    end = datetime.datetime.now()
    print(end - start)
plt.plot(P, P_false)
plt.xlabel("BSC error probability")
plt.ylabel("false probability")
plt.show()
