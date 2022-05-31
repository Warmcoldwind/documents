import re

with open(r"C:\Users\Roar\Desktop\新建文件夹\HAMILTON.txt", 'r') as fd:
    HAMILTON_list = []  # 存放所有单词，全部小写，并去除,.!等后缀，并去除空格字符串
    HAMILTON_dict = {}  # 保留{word: count}键值对
    for line in fd.readlines():
        for word in line.strip().split(" "):  # 移除空格
            HAMILTON_list.append(re.sub(r"[.|!|,]", "", word.lower()))  # 大写字母转小写
    HAMILTON_sets = list(set(HAMILTON_list))  # 确保唯一
    HAMILTON_dict = {word: HAMILTON_list.count(word) for word in HAMILTON_sets if word}
total = sum(HAMILTON_dict.values(), 0.0)
HAM = {k: v / total for k, v in HAMILTON_dict.items()}
# HAMILTON = sorted(HAM.items(), key=lambda d: d[1], reverse=True)[:5]
# print(HAMILTON)

with open(r"C:\Users\Roar\Desktop\新建文件夹\MADISON.txt", 'r') as fd:
    MADISON_list = []  # 存放所有单词，全部小写，并去除,.!等后缀，并去除空格字符串
    MADISON_dict = {}  # 保留{word: count}键值对
    for line in fd.readlines():
        for word in line.strip().split(" "):  # 移除空格
            MADISON_list.append(re.sub(r"[.|!|,]", "", word.lower()))  # 大写字母转小写
    MADISON_sets = list(set(MADISON_list))  # 确保唯一
    MADISON_dict = {word: MADISON_list.count(word) for word in MADISON_sets if word}
total = sum(MADISON_dict.values(), 0.0)
MAD = {k: v / total for k, v in MADISON_dict.items()}
# MADISON = sorted(MAD.items(), key=lambda d: d[1], reverse=True)[:5]
# print(MADISON)

final = []
for i in range(49, 60):
    with open(r"C:\Users\Roar\Desktop\新建文件夹\%s.txt" % i, 'r') as fd:
        word_list = []  # 存放所有单词，全部小写，并去除,.!等后缀，并去除空格字符串
        word_dict = {}  # 保留{word: count}键值对
        res = {}
        result1 = []
        result = {}
        for line in fd.readlines():
            for word in line.strip().split(" "):  # 移除空格
                word_list.append(re.sub(r"[.|!|,]", "", word.lower()))  # 大写字母转小写
        word_sets = list(set(word_list))  # 确保唯一
        word_dict = {word: word_list.count(word) for word in word_sets if word}
    total = sum(word_dict.values(), 0.0)
    res = {k: v / total for k, v in word_dict.items()}
    # result1 = sorted(res.items(), key=lambda d: d[1], reverse=True)[:5]
    # result = dict(result1)
    # print(result)
    num = 0
    num1 = 0
    for key in res:
        if key in HAM.keys() and key in MAD.keys():
            if res[key] > 0.001:
                num = num + 1
                if abs(HAM[key] - res[key]) < abs(MAD[key] - res[key]):
                    num1 = num1 + 1
    if num1 > num / 2:
        final.append('HAMILTON')
    else:
        final.append('MADISON')
print(final)