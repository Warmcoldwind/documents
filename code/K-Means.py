import matplotlib.pyplot as plt
from numpy import *

dot_num0 = 30
mean0 = [0, 0]
cov0 = [[1, 0], [0, 1]]
a = random.multivariate_normal(mean0, cov0, dot_num0)
x0 = a[:, 0]
y0 = a[:, 1]

dot_num = 20
mean1 = [1, 2]
cov1 = [[2, 0], [0, 1]]
b = random.multivariate_normal(mean1, cov1, dot_num)
x1 = b[:, 0]
y1 = b[:, 1]

mean2 = [2, 0]
cov2 = [[1, 0.3], [0.3, 1]]
c = random.multivariate_normal(mean2, cov2, dot_num)
x2 = c[:, 0]
y2 = c[:, 1]
d = []
print(x0)


def random_pick(some_list, probabilities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item


for i in range(0, 30):
    d.append(a[i].tolist())
for i in range(0, 20):
    d.append(b[i].tolist())
for i in range(0, 20):
    d.append(c[i].tolist())

n = random.randint(0, 69)  # 第一个中心
dis = []
P = []
ad = 0
for i in range(0, 70):
    dis_ = (d[n][0] - d[i][0]) ** 2 + (d[n][1] - d[i][1]) ** 2
    ad = ad + dis_
    dis.append(dis_)
for i in range(0, 70):
    P.append(dis[i] / ad)
num = []
for i in range(0, 70):
    num.append(i)
n1 = random_pick(num, P)  # 第二个中心
P_ = []
dis1 = []
P1 = []
dis2 = []
P2 = []
ad1 = ad2 = 0
for i in range(0, 70):
    dis1_ = (d[n][0] - d[i][0]) ** 2 + (d[n][1] - d[i][1]) ** 2
    ad1 = ad1 + dis1_
    dis1.append(dis1_)
    dis2_ = (d[n1][0] - d[i][0]) ** 2 + (d[n1][1] - d[i][1]) ** 2
    ad2 = ad2 + dis2_
    dis2.append(dis2_)
for i in range(0, 70):
    P1.append(dis1[i] / ad1)
for i in range(0, 70):
    P2.append(dis2[i] / ad2)
su = 0
P3 = []
for i in range(0, 70):
    su = su + P1[i] * P2[i]
    P_.append(P1[i] * P2[i])
for i in range(0, 70):
    P3.append(P_[i] / su)
n2 = random_pick(num, P3)  # 第二个中心
print(n, n1, n2)
od = [[n, n1, n2]]
for k in range(0, 10):
    s1 = []
    s2 = []
    s3 = []
    Di0 = []
    Di1 = []
    Di2 = []
    x00 = []
    y00 = []
    x10 = []
    y10 = []
    x20 = []
    y20 = []
    for i in range(0, 70):
        di0 = (d[n][0] - d[i][0]) ** 2 + (d[n][1] - d[i][1]) ** 2
        Di0.append([di0, i])
        di1 = (d[n1][0] - d[i][0]) ** 2 + (d[n1][1] - d[i][1]) ** 2
        Di1.append([di1, i])
        di2 = (d[n2][0] - d[i][0]) ** 2 + (d[n2][1] - d[i][1]) ** 2
        Di2.append([di2, i])
    for i in range(0, 70):
        if min(Di0[i][0], Di1[i][0], Di2[i][0]) == Di0[i][0]:
            s1.append(d[Di0[i][1]])

        elif min(Di0[i][0], Di1[i][0], Di2[i][0]) == Di1[i][0]:
            s2.append(d[Di1[i][1]])
        else:
            s3.append(d[Di2[i][1]])

    for i in range(0, len(s1)):
        x00.append(s1[i][0])
        y00.append(s1[i][1])
    for i in range(0, len(s2)):
        x10.append(s2[i][0])
        y10.append(s2[i][1])
    for i in range(0, len(s3)):
        x20.append(s3[i][0])
        y20.append(s3[i][1])
    cx0 = cy0 = cx1 = cy1 = cx2 = cy2 = 0
    for i in range(0, len(x00)):
        cx0 = cx0 + x00[i]
    for i in range(0, len(y00)):
        cy0 = cy0 + y00[i]
    for i in range(0, len(x10)):
        cx1 = cx1 + x10[i]
    for i in range(0, len(y10)):
        cy1 = cy1 + y10[i]
    for i in range(0, len(x20)):
        cx2 = cx2 + x20[i]
    for i in range(0, len(y20)):
        cy2 = cy2 + y20[i]
    if len(x00) != 0:
        c0 = [cx0 / len(x00), cy0 / len(y00)]
    else:
        c0 = d[n]
    if len(x10) != 0:
        c1 = [cx1 / len(x10), cy1 / len(y10)]
    else:
        c1 = d[n1]
    if len(x20) != 0:
        c2 = [cx2 / len(x20), cy2 / len(y20)]
    else:
        c2 = d[n2]
    ne0 = ne1 = ne2 = 0
    nea0 = []
    nea1 = []
    nea2 = []
    for i in range(0, 70):
        ne0 = (d[i][0] - c0[0]) ** 2 + (d[i][1] - c0[1]) ** 2
        nea0.append([ne0, i])
    for i in range(0, 70):
        ne1 = (d[i][0] - c1[0]) ** 2 + (d[i][1] - c1[1]) ** 2
        nea1.append([ne1, i])
    for i in range(0, 70):
        ne2 = (d[i][0] - c2[0]) ** 2 + (d[i][1] - c2[1]) ** 2
        nea2.append([ne2, i])
    nea0_ = sorted(nea0, key=(lambda x: x[0]))
    nea1_ = sorted(nea1, key=(lambda x: x[0]))
    nea2_ = sorted(nea2, key=(lambda x: x[0]))
    n = nea0_[0][1]
    n1 = nea1_[0][1]
    n2 = nea2_[0][1]
    od.append([n, n1, n2])
    if od[k + 1] == od[k]:
        print(k)
        break
l = len(od)
print(od)
for i in range(0, l):
    if len(od) > 3:
        if 0 <= od[len(od) - 1][0] < 30:
            if 0 <= od[len(od) - 1][1] < 30 or 0 <= od[len(od) - 1][2] < 30:
                del (od[-1])
            elif 30 <= od[len(od) - 1][1] < 50 and 30 <= od[len(od) - 1][2] < 50:
                del (od[-1])
            elif 50 <= od[len(od) - 1][1] < 70 and 50 <= od[len(od) - 1][2] < 70:
                del (od[-1])
            else:
                break
        if 30 <= od[len(od) - 1][0] < 50:
            if 30 <= od[len(od) - 1][1] < 50 or 30 <= od[len(od) - 1][2] < 50:
                del (od[-1])
            elif 0 <= od[len(od) - 1][1] < 30 and 0 <= od[len(od) - 1][2] < 30:
                del (od[-1])
            elif 50 <= od[len(od) - 1][1] < 70 and 50 <= od[len(od) - 1][2] < 70:
                del (od[-1])
            else:
                break
        if 50 <= od[len(od) - 1][0] < 70:
            if 50 <= od[len(od) - 1][1] < 70 or 50 <= od[len(od) - 1][2] < 70:
                del (od[-1])
            elif 0 <= od[len(od) - 1][1] < 30 and 0 <= od[len(od) - 1][2] < 30:
                del (od[-1])
            elif 30 <= od[len(od) - 1][1] < 50 and 30 <= od[len(od) - 1][2] < 50:
                del (od[-1])
            else:
                break
    else:
        break
print(od)
n = od[len(od) - 1][0]
n1 = od[len(od) - 1][1]
n2 = od[len(od) - 1][2]


def cl(x):
    if 0 <= x < 30:
        return 'r'
    elif 30 <= x < 50:
        return 'b'
    else:
        return 'y'


plt.scatter(d[n][0], d[n][1], c=cl(n), marker='+')
plt.scatter(d[n1][0], d[n1][1], c=cl(n1), marker='+')
plt.scatter(d[n2][0], d[n2][1], c=cl(n2), marker='+')
plt.scatter(x00, y00, c=cl(n), alpha=0.5)
plt.scatter(x10, y10, c=cl(n1), alpha=0.5)
plt.scatter(x20, y20, c=cl(n2), alpha=0.5)
plt.show()
plt.scatter(x0, y0, c='r')
plt.scatter(x1, y1, c='b')
plt.scatter(x2, y2, c='y')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
