import os
from PIL import Image
import numpy as np
from collections import Counter
from keras.utils import to_categorical
import pickle
import gc
import config
import matplotlib.pyplot as plt


class InputExample:
    def __init__(self, id, input, seqlen, label=None):
        self.id = id
        self.input = input
        self.label = label
        self.seqlen = seqlen


class Data:
    def __init__(self, path):
        self.path = path
        self.maxnum = 24
        self.id2lid, self.lid2label, self.id2label, self.label_list = self.get_map()
        # self.inputExamplesn, self.id2label, self.id2lid, self.label2lid, self.label_list = self.load_example()

    def load_example(self):
        train_path = os.path.join(self.path, '训练集')
        input_train = os.path.join(train_path, 'lip_train')
        maxnum = self.maxnum

        id = os.listdir(input_train)
        input_examples = []
        index = 0
        for x in id:
            example_label = self.id2lid[x]
            example_label = to_categorical(example_label, num_classes=313)
            example_path = os.path.join(input_train, x)
            # 处理每一张图片
            input_example = []
            img_list = os.listdir(example_path)
            for y in img_list:
                img_path = os.path.join(example_path, y)
                # print(img_path)
                im = Image.open(img_path)
                im = im.convert("L")
                im = im.resize((config.WIDTH, config.HEIGHT))
                matrix = np.asarray(im)
                matrix = matrix/255
                input_example.append(matrix)
            # 填充0使每个词的图片数量一致
            seqlen = len(input_example)
            while len(input_example) < maxnum:
                input_example.append(np.zeros(shape=(config.WIDTH, config.HEIGHT)))
            input_example = np.array(input_example)
            print('processing {}....input_example shape:{}'.format(index, input_example.shape))
            index += 1
            inputExample = InputExample(id=x, input=input_example, seqlen=seqlen, label=example_label)
            input_examples.append(inputExample)
        return input_examples

    def load_test(self):
        test_path = os.path.join(self.path, '测试集')
        input_test = os.path.join(test_path, 'lip_test')
        maxnum = self.maxnum
        id = os.listdir(input_test)
        test_examples = []
        index = 0
        for x in id:
            testexample_path = os.path.join(input_test, x)
            # 处理每一张图片
            test_example = []
            img_list = os.listdir(testexample_path)
            for y in img_list:
                img_path = os.path.join(testexample_path, y)
                # print(img_path)
                im = Image.open(img_path)
                im = im.convert("L")
                im = im.resize((config.WIDTH, config.HEIGHT))
                matrix = np.asarray(im)
                matrix = matrix / 255
                test_example.append(matrix)
                if len(test_example) == maxnum:
                    break
            # 填充0使每个词的图片数量一致
            seqlen = len(test_example)
            while len(test_example) < maxnum:
                test_example.append(np.zeros(shape=(config.WIDTH, config.HEIGHT)))
            test_example = np.array(test_example)
            print('processing {}....test_example shape:{}'.format(index, test_example.shape))
            index += 1
            testExample = InputExample(id=x, input=test_example, seqlen=seqlen, label=None)
            test_examples.append(testExample)
        return test_examples

    def get_map(self):
        train_path = os.path.join(self.path, '训练集')
        label_train = os.path.join(train_path, 'lip_train.txt')
        id2label = {}
        id2lid = {}
        label2lid = {}
        label_list = []
        lid2label = {}
        with open(label_train, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                content = line.strip().split('\t')
                id2label[content[0]] = content[1]
                label_list.append(content[1])
        label_list = list(set(label_list))
        label_list.sort()
        for index, label in enumerate(label_list):
            label2lid[label] = index

        for x in id2label:
            # x 是id id2label[x]是label
            id2lid[x] = label2lid[id2label[x]]
        for x in label2lid:
            # x=label label2lid=label_id
            lid2label[label2lid[x]] = x
        return id2lid, lid2label, id2label, label_list


    def get_maxnum(self):
        maxnum = 0
        num_list = []
        train_path = os.path.join(self.path, '训练集')
        input_train = os.path.join(train_path, 'lip_train')
        id = os.listdir(input_train)
        for x in id:
            example_path = os.path.join(input_train, x)
            img_list = os.listdir(example_path)
            img_num = len(img_list)
            num_list.append(img_num)
            if img_num > maxnum:
                maxnum = img_num
        return maxnum, num_list


if __name__ == '__main__':
    d = Data('E:\ZH project\data\新网银行唇语识别竞赛数据')
    d.load_example()
#     id2lid = d.id2lid
#     lid2label = d.lid2label
#     id2label = d.id2label
#     inputExamples = d.load_example()
#     for x in inputExamples:
#         print(x.id)
#         print(np.argmax(x.label))
#         print(lid2label[np.argmax(x.label)])