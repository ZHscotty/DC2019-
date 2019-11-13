import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
import config
from keras.utils import to_categorical
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import pickle


class Model:

    def __init__(self):

        # batch, maxnum, 4096
        self.input = tf.placeholder(shape=[None, 24, 112, 112], dtype=tf.float32)
        # batch, type_num
        self.label = tf.placeholder(shape=[None, 313], dtype=tf.float32)
        self.is_training = tf.placeholder(dtype=tf.bool)
        self.MODEL_PATH = '../result/model/LipNet/d3d/model_cf'
        self.MODEL_DIC = '../result/model/LipNet/d3d'
        self.PIC_DIC = '../result/pic/LipNet/d3d'
        self.id2label = self.load_table('../data/id2label.pkl')
        self.label2lid = self.load_table('../data/label2lid.pkl')
        self.score, self.acc, self.loss, self.train_step, self.t = self.create_graph()

    def conv3d_layer(self, inputs, filters, kernel, strides, pool_kernel, pool_stride):
        conv = tf.layers.conv3d(inputs, filters, kernel, strides, padding='SAME', activation=tf.nn.relu)
        conv = tf.layers.batch_normalization(conv, training=self.is_training)
        pool = tf.layers.max_pooling3d(conv, pool_kernel, pool_stride, padding='SAME')
        return pool

    def dense_layer(self, inputs, filter1, filter2):
        inputs_bn = tf.layers.batch_normalization(inputs, training=self.is_training)
        inputs_a = tf.nn.relu(inputs_bn)
        conv1 = tf.layers.conv3d(inputs_a, filter1, (1, 1, 1), (1, 1, 1), padding='SAME', activation=tf.nn.relu)
        conv1_bn = tf.layers.batch_normalization(conv1, training=self.is_training)
        conv2 = tf.layers.conv3d(conv1_bn, filter2, (3, 3, 3), (1, 1, 1), padding='SAME')
        dense_output = tf.concat([conv1, conv2], axis=-1)
        return dense_output

    def transblock(self, inputs, filters):
        inputs_bn = tf.layers.batch_normalization(inputs, training=self.is_training)
        inputs_a = tf.nn.relu(inputs)
        conv1 = tf.layers.conv3d(inputs_a, filters, (1, 1, 1), (1, 1, 1), padding='SAME')
        output = tf.layers.average_pooling3d(conv1, (1, 2, 2), (1, 2, 2), padding='SAME',)
        return output

    def attention_layer(self, inputs):
        q = tf.layers.dense(inputs=inputs, units=config.ATT_SIZE)
        k = tf.layers.dense(inputs=inputs, units=config.ATT_SIZE)
        v = tf.layers.dense(inputs=inputs, units=config.ATT_SIZE)
        k = tf.transpose(k, perm=[0, 2, 1])
        s = tf.einsum('aij,ajk->aik', q, k)
        s = s/tf.sqrt(float(config.ATT_SIZE))
        s = tf.nn.softmax(s, axis=2)
        att_output = tf.einsum('aij,ajk->aik', s, v)
        return att_output

    def create_graph(self):
        cnn_input = self.input
        print('cnn_input shape:', cnn_input.shape)
        cnn_input_ex = tf.expand_dims(cnn_input, axis=-1)
        print('cnn_input_ex shape:', cnn_input_ex.shape)
        conv1 = self.conv3d_layer(cnn_input_ex, 64, (5, 7, 7), (1, 2, 2), (1, 3, 3), (1, 2, 2))
        print('conv1 shape', conv1.shape)
        inputs = conv1
        filter1 = [32, 64, 96]
        filter2 = [64, 96, 128]
        for x in range(3):
            dense = self.dense_layer(inputs, filter1=filter1[x], filter2=filter2[x])
            transblock = self.transblock(dense, filters=filter1[x])
            inputs = transblock
        dense_out = self.dense_layer(inputs, filter1=96, filter2=128)
        print('dense_out shape', dense_out.shape)

        # Flatten
        lstm_input = tf.reshape(dense_out, shape=[-1, dense_out.shape[1], dense_out.shape[2] * dense_out.shape[3] * dense_out.shape[4]])
        print('before att:', lstm_input.shape)
        lstm_input = self.attention_layer(lstm_input)
        print('lstm_input shape:', lstm_input.shape)

        for index in range(2):
            with tf.variable_scope('lstm{}'.format(index)):
                cell_fw = tf.contrib.rnn.BasicLSTMCell(256)
                cell_bw = tf.contrib.rnn.BasicLSTMCell(256)
                lstm_out, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=lstm_input,
                                                              dtype=tf.float32)
                lstm_output = tf.concat(lstm_out, 2)
                if self.is_training is not None:
                    lstm_output = tf.nn.dropout(lstm_output, 0.5)
            lstm_input = lstm_output
        dense_input = self.attention_layer(lstm_output)
        dense_input = tf.reduce_mean(dense_input, axis=1)
        print('dense_input shape:', dense_input.shape)
        # dense
        score = tf.layers.dense(dense_input, units=313)
        print('score:', score.shape)
        t = tf.nn.softmax(score)
        # optimizer
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(score, 1), tf.argmax(self.label, 1)), dtype=tf.float32))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=score))
        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_op):
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
                acc_total = 0
                loss_total = 0
                acc_dev_total = 0
                loss_dev_total = 0
                train_step = 0
                train_num = 0
                train_list, dev_list = train_test_split(os.listdir(train_path), test_size=0.2)
                for input, label in self.prepare_data(train_path, train_list):
                    _, acc_t, loss_t, t = sess.run([self.train_step, self.acc, self.loss, self.t],
                                                   {self.input: input, self.label: label,
                                                    self.is_training: True})
                    train_num += len(input)
                    acc_total += acc_t
                    loss_total += loss_t
                    print('step{} [{}/{}]  --acc:{}, --loss:{}'.format(train_step, train_num, len(train_list),
                                                                       round(acc_t, 2), round(loss_t, 4)))
                    train_step += 1
                acc_t = acc_total / train_step
                loss_t = loss_total / train_step
                acc_train.append(acc_t)
                loss_train.append(loss_t)

                # dev

                dev_step = 0
                for input, label in self.prepare_data(train_path, dev_list):
                    acc_d, loss_d = sess.run([self.acc, self.loss], {self.input: input, self.label: label,
                                                                     self.is_training: False})
                    dev_step += 1
                    acc_dev_total += acc_d
                    loss_dev_total += loss_d
                acc_dd = acc_dev_total / dev_step
                loss_dd = loss_dev_total / dev_step
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
        plt.savefig(os.path.join(self.PIC_DIC, 'acc_cf.png'))
        plt.close()

        plt.plot(loss_train)
        plt.plot(loss_dev)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(os.path.join(self.PIC_DIC, 'loss_cf.png'))
        plt.close()

    def predict(self, test_path):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.latest_checkpoint(self.MODEL_DIC)  # 找到存储变量值的位置
            saver.restore(sess, ckpt)
            test_list = os.listdir(test_path)
            pre_list = []
            for input in self.prepare_test_data(test_path, test_list):
                predict = sess.run(self.score, {self.input: input, self.is_training: False})
                pre_list.extend(predict)
            result = np.array(pre_list)
            return result

    def prepare_data(self, path, file_list):

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
                plabel = self.id2label[d]
                plabel = self.label2lid[plabel]
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
            yield input, label

    def prepare_test_data(self, path, file_list):
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
            input = []
            for d in data_list:
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
                input.append(pinput)
            input = np.array(input)
            yield input

    def load_table(self, table_path):

        with open(table_path, 'rb') as f:

            result = pickle.load(f)

        return result