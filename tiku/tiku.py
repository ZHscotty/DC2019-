import os
import pandas as pd
import random

path = r'C:\Users\ZH\Desktop\计算机网络题库'
tx = []
tg = []
ndz = []
zj = []
a = []
b = []
c = []
d = []
answer = []

# 整合题库
file_list = os.listdir(path)
for f in file_list:
    file_path = os.path.join(path, f)
    file = pd.read_excel(file_path)
    tx.extend(list(file['题型']))
    tg.extend(list(file['题干']))
    ndz.extend(list(file['难度值']))
    zj.extend(list(file['所属章节']))
    a.extend(list(file['选项A']))
    b.extend(list(file['选项B']))
    c.extend(list(file['选项C']))
    d.extend(list(file['选项D']))
    answer.extend(list(file['答案']))

# 乱序
seed = random.randint(0, 100)
random.seed(seed)
random.shuffle(tx)
random.seed(seed)
random.shuffle(tg)
random.seed(seed)
random.shuffle(ndz)
random.seed(seed)
random.shuffle(zj)
random.seed(seed)
random.shuffle(a)
random.seed(seed)
random.shuffle(b)
random.seed(seed)
random.shuffle(c)
random.seed(seed)
random.shuffle(d)
random.seed(seed)
random.shuffle(answer)

# 生成题库
start = 0
batch = 40
for index in range(len(answer)//batch):
    tiku = {}
    end = start + batch
    if end > len(answer):
        end = len(answer)
    tiku['题型'] = tx[start:end]
    tiku['题干'] = tg[start:end]
    tiku['难度值'] = ndz[start:end]
    tiku['所属章节'] = zj[start:end]
    tiku['选项A'] = a[start:end]
    tiku['选项B'] = b[start:end]
    tiku['选项C'] = c[start:end]
    tiku['选项D'] = d[start:end]
    tiku['答案'] = answer[start:end]
    output = pd.DataFrame(tiku)
    output.to_excel(r'C:\Users\ZH\Desktop\计算机网络题库\题库{}(40个选择题).xlsx'.format(index+1), index=False, encoding='utf-8')
    print('生成题库{}成功!'.format(index+1))
    start = end
