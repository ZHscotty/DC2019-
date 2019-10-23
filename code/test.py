import os
path = 'E:\比赛数据\新网银行唇语识别竞赛数据'

train_path = os.path.join(path, '1.训练集')
test_path = os.path.join(path, '2.测试集')
input_train = os.path.join(train_path, 'lip_train')
label_train = os.path.join(train_path, 'lip_train.txt')
maxnum = 24

# 读取训练集
# 读取label
id = os.listdir(input_train)[:10]
input_examples = []
id2label = {}
id2lid = {}
label2lid = {}
label_list = []
with open(label_train, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        content = line.strip().split('\t')
        id2label[content[0]] = content[1]
        label_list.append(content[1])
    label_list = list(set(label_list))
    for index, label in enumerate(label_list):
        label2lid[label] = index
    for x in id2label:
        id2lid[x] = label2lid[id2label[x]]
    print(id2label)
    print(id2lid)
    index = 0
    for x in id:
        example_label = id2lid[x]
        # example_label = to_categorical(example_label, num_classes=313)
        example_path = os.path.join(input_train, x)
        # 处理每一张图片
        input_example = []
        img_list = os.listdir(example_path)
        index += 1
        print('id:', x, 'label_id:', example_label, 'label:', id2label[x], 'label_id:', label2lid[id2label[x]])
