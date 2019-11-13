import tensorflow as tf
import config
import matplotlib.pyplot as plt
import numpy as np
import math
from keras.utils import to_categorical
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import pickle
np.set_printoptions(threshold=np.inf)
# PIC_NUM = 24
# WIDTH = 64
# HEIGHT = 64
# HIDDEN_SIZE1 = 512
# HIDDEN_SIZE2 = 512
# DROP = 0.5
# TYPE_NUM = 313
# LR = 0.0001
# EPOCH = 200
# BATCH_SIZE = 32
# EARLY_STEP = 3


class Model:
    def __init__(self):
        # batch, maxnum, 4096
        self.input = tf.placeholder(shape=[None, config.PIC_NUM, config.WIDTH, config.HEIGHT], dtype=tf.float32)
        # batch, type_num
        self.label = tf.placeholder(shape=[None, None], dtype=tf.float32)
        self.is_training = tf.placeholder(dtype=tf.bool)
        self.MODEL_PATH = '../result/model/LipNet/model'
        self.MODEL_DIC = '../result/model/LipNet'
        self.ACC_DIC = '../result/pic/LipNet'
        self.LOSS_DIC = '../result/pic/LipNet'
        self.PIC_DIC = '../result/pic/LipNet'
        self.id2label = self.load_table('../data/id2label.pkl')
        self.label2lid = self.load_table('../data/label2lid.pkl')
        self.score, self.acc, self.loss, self.train_step, self.t = self.create_graph()

    def create_graph(self):
        # CNN
        # shape (batch, depth, width, heigh, channel=1)
        cnn_input = self.input
        print('cnn_input shape:', cnn_input.shape)
        cnn_input_ex = tf.expand_dims(cnn_input, axis=-1)
        print('cnn_input_ex shape:', cnn_input_ex.shape)

        conv = tf.layers.conv3d(cnn_input_ex, config.FILTER_NUM, kernel_size=(5, 7, 7), strides=(1, 2, 2),
                                padding='SAME', activation=None, use_bias=False)

        conv_bn = tf.layers.batch_normalization(conv, training=self.is_training)
        print('conv_bn shape:', conv_bn.shape)

        conv_a = tf.nn.relu(conv_bn)
        maxpool = tf.layers.max_pooling3d(conv_a, pool_size=(1, 3, 3), strides=(1, 2, 2), padding='SAME')
        print('maxpool shape:', maxpool.shape)

        # Flatten
        lstm_input = tf.reshape(maxpool, shape=[-1, maxpool.shape[1], maxpool.shape[2]*maxpool.shape[3]*maxpool.shape[4]])
        print('lstm_input shape:', lstm_input.shape)

        for index in range(config.LSTMS):
            with tf.variable_scope('lstm{}'.format(index)):
                cell_fw = tf.contrib.rnn.BasicLSTMCell(config.LSTM_HIDDEN_SIZE)
                cell_bw = tf.contrib.rnn.BasicLSTMCell(config.LSTM_HIDDEN_SIZE)
                lstm_out, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=lstm_input,
                                                              dtype=tf.float32)
                lstm_output = tf.concat(lstm_out, 2)
                if self.is_training is not None:
                    lstm_output = tf.nn.dropout(lstm_output, config.LSTMDROP)
            lstm_input = lstm_output

        dense_input = lstm_output[:, -1, :]
        print('dense_input shape:', dense_input.shape)

        # dense
        score = tf.layers.dense(dense_input, units=config.TYPE_NUM,
                                kernel_initializer=tf.glorot_uniform_initializer())
        print('score:', score.shape)

        t = tf.nn.softmax(score)
        # optimizer
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(score, 1), tf.argmax(self.label, 1)), dtype=tf.float32))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=score))
        train_step = tf.train.AdamOptimizer(config.LR).minimize(loss)

        return score, acc, loss, train_step, t

    def train(self, train_path):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            acc_train = []
            acc_dev = []
            loss_train = []
            loss_dev = []
            should_stop = False
            step = 0
            es_step = 0
            loss_stop = 99999
            n = 0
            while step < config.EPOCH and should_stop is False:
                print('Epoch:{}'.format(step))
                begin = 0
                acc_total = 0
                loss_total = 0
                acc_dev_total = 0
                loss_dev_total = 0

                file_list = os.listdir(train_path)
                train_file, test_file = train_test_split(file_list, test_size=0.2)
                if len(train_file) % config.BATCH_SIZE == 0:
                    num = len(train_file)//config.BATCH_SIZE
                else:
                    num = len(train_file)//config.BATCH_SIZE+1
                begin = 0
                for i in range(num):
                    end = begin + config.BATCH_SIZE
                    if end > len(train_file):
                        end = len(train_file)
                    data_list = train_file[begin:end]
                    begin = end
                    label = []
                    input = []
                    for d in data_list:
                        plabel = self.id2label[d]
                        plabel = self.label2lid[plabel]
                        plabel = to_categorical(plabel, 313)
                        pinput = []
                        file_path = os.path.join(train_path, d)
                        for ff in os.listdir(file_path):
                            pic_path = os.path.join(file_path, ff)
                            im = Image.open(pic_path)
                            im = im.convert("L")
                            im = im.resize((config.WIDTH, config.HEIGHT))
                            matrix = np.asarray(im)
                            matrix = matrix / 255
                            pinput.append(matrix)
                        while len(pinput) < config.PIC_NUM:
                            pinput.append(np.zeros(shape=(config.WIDTH, config.HEIGHT)))
                        pinput = np.array(pinput)
                        label.append(plabel)
                        input.append(pinput)
                    label = np.array(label)
                    input = np.array(input)
                    _, acc_t, loss_t, t = sess.run([self.train_step, self.acc, self.loss, self.t],
                                                   {self.input: input, self.label: label,
                                                    self.is_training: True})
                    acc_total += acc_t
                    loss_total += loss_t
                    print('step{} [{}/{}]  --acc:{}, --loss:{}'.format(i, end, len(train_file),
                                                                       round(acc_t, 2), round(loss_t, 4)))

                acc_t = acc_total/num
                loss_t = loss_total/num
                acc_train.append(acc_t)
                loss_train.append(loss_t)

                # dev
                if len(test_file) % config.BATCH_SIZE == 0:
                    num = len(test_file)//config.BATCH_SIZE
                else:
                    num = len(test_file)//config.BATCH_SIZE+1
                begin = 0
                for i in range(num):
                    end = begin + config.BATCH_SIZE
                    if end > len(test_file):
                        end = len(test_file)
                    data_list = test_file[begin:end]
                    label = []
                    input = []
                    for d in data_list:
                        plabel = self.id2label[d]
                        plabel = self.label2lid[plabel]
                        plabel = to_categorical(plabel, 313)
                        pinput = []
                        file_path = os.path.join(train_path, d)
                        for ff in file_path:
                            pic_path = os.path.join(file_path, ff)
                            im = Image.open(pic_path)
                            im = im.convert("L")
                            im = im.resize((config.WIDTH, config.HEIGHT))
                            matrix = np.asarray(im)
                            matrix = matrix / 255
                            pinput.append(matrix)
                        while len(pinput) < len(config.PIC_NUM):
                            pinput.append(np.zeros(shape=(config.WIDTH, config.HEIGHT)))
                        pinput = np.array(pinput)
                        label.append(plabel)
                        input.append(pinput)
                    label = np.array(label)
                    input = np.array(input)
                    acc_d, loss_d = sess.run([self.acc, self.loss], {self.input: input, self.label: label,
                                                                     self.is_training: False})
                    acc_dev_total += acc_d
                    loss_dev_total += loss_d
                acc_dd = acc_dev_total/num
                loss_dd = loss_dev_total/num
                acc_dev.append(acc_dd)
                loss_dev.append(loss_dd)
                print('Epoch{}----acc:{},loss:{},val_acc:{},val_loss:{}'.format(step, acc_t, loss_t, round(acc_dd, 2),
                                                                                round(loss_dd, 4)))
                if loss_dd > loss_stop:
                    if n >= config.EARLY_STEP:
                        should_stop = True
                    else:
                        n += 1
                else:
                    if not os.path.exists(self.MODEL_DIC):
                        os.makedirs(self.MODEL_DIC)
                    saver.save(sess, self.MODEL_PATH)
                    es_loss = loss_dd
                    es_acc = acc_dd
                    es_step = step
                    n = 0
                    loss_stop = loss_dd
                step += 1

            if should_stop:
                print('Early Stop at Epoch{} acc:{} loss:{}'.format(es_step, es_acc, es_loss))

        if not os.path.exists(self.PIC_DIC):
            os.makedirs(self.PIC_DIC)
        plt.plot(acc_train)
        plt.plot(acc_dev)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(os.path.join(self.PIC_DIC, 'acc.png'))
        plt.close()

        plt.plot(loss_train)
        plt.plot(loss_dev)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(self.PIC_DIC, 'loss.png')
        plt.close()

    def predict(self, test_path):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.latest_checkpoint(self.MODEL_DIC)  # 找到存储变量值的位置
            saver.restore(sess, ckpt)
            # test_input, test_label = self.prepare_data(test_path, self.id2label, self.label2lid)
            # test_list = os.listdir(test_path)
            file_list = os.listdir(test_path)
            if len(file_list) % config.BATCH_SIZE == 0:
                num = len(file_list) // config.BATCH_SIZE
            else:
                num = len(file_list) // config.BATCH_SIZE + 1
            begin = 0
            pre_list = []
            for i in range(num):
                end = begin + config.BATCH_SIZE
                if end > len(file_list):
                    end = len(file_list)
                data_list = file_list[begin:end]
                begin = end
                input = []
                for d in data_list:
                    pinput = []
                    file_path = os.path.join(test_path, d)
                    for ff in os.listdir(file_path):
                        pic_path = os.path.join(file_path, ff)
                        im = Image.open(pic_path)
                        im = im.convert("L")
                        im = im.resize((config.WIDTH, config.HEIGHT))
                        matrix = np.asarray(im)
                        matrix = matrix / 255
                        pinput.append(matrix)
                    while len(pinput) < config.PIC_NUM:
                        pinput.append(np.zeros(shape=(config.WIDTH, config.HEIGHT)))
                    pinput = np.array(pinput)
                    input.append(pinput)
                input = np.array(input)

                predict = sess.run(self.score, {self.input: input, self.is_training:False})
                pre_list.append(predict)
            result = np.array(pre_list)
            return result


    def prepare_data(self, path, id2label, label2lid):
        file_list = os.listdir(path)
        if len(file_list) % config.BATCH_SIZE == 0:
            num = len(file_list) // config.BATCH_SIZE
        else:
            num = len(file_list) // config.BATCH_SIZE + 1
        begin = 0
        for i in range(num):
            end = begin + config.BATCH_SIZE
            if end > len(file_list):
                end = len(file_list)
            data_list = file_list[begin:end]
            begin = end
            label = []
            input = []
            for d in data_list:
                plabel = id2label[d]
                plabel = label2lid[plabel]
                plabel = to_categorical(plabel, 313)
                pinput = []
                file_path = os.path.join(path, d)
                for ff in os.listdir(file_path):
                    pic_path = os.path.join(file_path, ff)
                    im = Image.open(pic_path)
                    im = im.convert("L")
                    im = im.resize((config.WIDTH, config.HEIGHT))
                    matrix = np.asarray(im)
                    matrix = matrix / 255
                    pinput.append(matrix)
                while len(pinput) < config.PIC_NUM:
                    pinput.append(np.zeros(shape=(config.WIDTH, config.HEIGHT)))
                pinput = np.array(pinput)
                label.append(plabel)
                input.append(pinput)
            label = np.array(label)
            input = np.array(input)
        return input, label


    def load_table(self, table_path):
        with open(table_path, 'rb') as f:
            result = pickle.load(f)
        return result
