import matplotlib.pyplot as plt

f = open(r"C:\Users\Roar\Desktop\iris.txt", 'r')
data1 = []
for line in f:
    line = line.strip('\n')
    data1.append(line.split(','))
data = data1[:150]
print(data)

for i in range(0, 150):
    if data[i][4] == 'Iris-setosa':
        data[i][4] = '0'
    if data[i][4] == 'Iris-versicolor':
        data[i][4] = '1'
    if data[i][4] == 'Iris-virginica':
        data[i][4] = '2'
print(data)

data21 = []
for i in range(0, 45):
    data21.append(i)
data31 = []
for i in range(0, 9):
    data31.append(data21[5 * i:5 * (i + 1)])
data22 = []
for i in range(50, 95):
    data22.append(i)
data32 = []
for i in range(0, 9):
    data32.append(data22[5 * i:5 * (i + 1)])
data23 = []
for i in range(100, 145):
    data23.append(i)
data33 = []
for i in range(0, 9):
    data33.append(data23[5 * i:5 * (i + 1)])

final = [] #放错误率
for k in range(1, 120):
    result = 0
    success = 0
    for w in range(0, 9):
        va = []
        tr = []
        for z in range(0, 5):
            va.append(data31[w][z])
            va.append(data32[w][z])
            va.append(data33[w][z])
        for g in range(0, 8):
            for s in range(0, 5):
                y = w+g+1
                if y > 8:
                    y = y-9
                tr.append(data31[y][s])
                tr.append(data32[y][s])
                tr.append(data33[y][s])
        for n in range(0, 15):
            res = []
            re0 = 0
            re1 = 0
            re2 = 0
            v = va[n]
            cha = []
            cha1 = []
            for i in range(0, 120):  # 找k个距离最近的点
                delta = 0
                t = tr[i]
                for j in range(0, 4):
                    delta = delta + (float(data[v][j]) - float(data[t][j])) ** 2
                cha.append([delta, t])
            cha1 = sorted(cha, key=(lambda x: x[0]))
            res = cha1[:k]
            # print(res)
            for e in range(0, k):
                if data[res[e][1]][4] == '0':
                    re0 = re0 + 1
                elif data[res[e][1]][4] == '1':
                    re1 = re1 + 1
                else:
                    re2 = re2 + 1
            re = max(re0, re1, re2)
            if re == re0:
                re = '0'
            elif re == re1:
                re = '1'
            else:
                re = '2'
            if data[v][4] != re:
                result = result + 1
            else:
                success = success + 1
    rate = result / (result+success)
    print(result)
    final.append([rate, k])
final1 = sorted(final, key=(lambda x: x[0]))
final2 = sorted(final, key=(lambda x: x[1]))
print(final1)
final3 = []
k1 = []
for i in range(1, 119):
    k1.append(i)
for i in range(1, 119):
    final3.append(final2[i][0])
plt.plot(k1, final3)
plt.show()
