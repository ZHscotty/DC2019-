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




def to_sparse(tensor, lengths, maxLength):
    mask = tf.sequence_mask(lengths, maxLength)
    indices = tf.to_int64(tf.where(tf.equal(mask, True)))
    values = tf.to_int32(tf.boolean_mask(tensor, mask))
    shape = tf.to_int64(tf.shape(tensor))
    return tf.SparseTensor(indices, values, shape)


class Model:
    def __init__(self):
        self.input = tf.placeholder(shape=[None, config.PIC_NUM, config.WIDTH, config.HEIGHT, 3], dtype=tf.float32)
        # batch, type_num
        self.label = tf.placeholder(shape=[None, config.MAXLABEL], dtype=tf.float32)
        self.is_training = tf.placeholder(dtype=bool)
        self.input_len = tf.placeholder(shape=[None], dtype=tf.int32)
        self.label_len = tf.placeholder(shape=[None], dtype=tf.int32)
        self.MODEL_PATH = '../result/model/LipNet/model'
        self.MODEL_DIC = '../result/model/LipNet'
        self.ACC_DIC = '../result/pic/LipNet'
        self.LOSS_DIC = '../result/pic/LipNet'
        self.PIC_DIC = '../result/pic/LipNet'
        self.id2label = self.load_table('../data/id2label.pkl')
        self.label2lid = self.load_table('../data/label2lid.pkl')
        self.char2index = self.load_table('../data/char2index.pkl')
        self.logits, self.loss, self.acc, self.train_step = self.create_graph()

    def create_graph(self):
        # CNN
        # shape (batch, depth, width, heigh, channel=2)
        cnn_input = self.input
        print('cnn_input shape:', cnn_input.shape)

        conv = tf.layers.conv3d(cnn_input, config.FILTER_NUM, kernel_size=(5, 7, 7), strides=(1, 2, 2),
                                padding='SAME', activation=None, use_bias=False)
        conv_bn = tf.layers.batch_normalization(conv, training=self.is_training)
        print('conv_bn shape:', conv_bn.shape)

        conv_a = tf.nn.relu(conv_bn)
        maxpool = tf.layers.max_pooling3d(conv_a, pool_size=(1, 3, 3), strides=(1, 2, 2), padding='SAME')
        print('maxpool shape:', maxpool.shape)

        # Flatten
        lstm_input = tf.reshape(maxpool,
                                shape=[-1, maxpool.shape[1], maxpool.shape[2] * maxpool.shape[3] * maxpool.shape[4]])
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

        dense_input = lstm_output
        print('dense_input shape:', dense_input.shape)

        # dense (batch, maxlen, type_num)
        logits = tf.layers.dense(dense_input, units=len(self.char2index)+1)
        logits = tf.transpose(logits, [1, 0, 2])
        print('logits:', logits.shape)
        sparse_targets = to_sparse(self.label, self.label_len, config.MAXLABEL)
        print('sparse_targets shape', sparse_targets.shape)

        # optimizer
        r, _ = tf.nn.ctc_greedy_decoder(inputs=logits, sequence_length=self.input_len, merge_repeated=True)
        acc = tf.reduce_mean(tf.edit_distance(tf.cast(r[0], tf.int32), sparse_targets))
        loss = tf.reduce_mean(tf.nn.ctc_loss(labels=sparse_targets, inputs=logits,
                                             sequence_length=self.input_len, ignore_longer_outputs_than_inputs=True))
        train_step = tf.train.AdamOptimizer(config.LR).minimize(loss)

        return logits, loss, acc, train_step

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
                for input, label, input_len, label_len in self.prepare_data(train_path, train_list):
                    _, loss_t, acc_t = sess.run([self.train_step, self.loss, self.acc],
                                                {self.input: input, self.label: label, self.is_training: True,
                                                 self.input_len: input_len, self.label_len: label_len})
                    train_num += len(input)
                    acc_total += acc_t
                    loss_total += loss_t
                    print('step{} [{}/{}]   --acc:{} --loss:{}'.format(train_step, train_num, len(train_list),
                                                                       acc_t, loss_t))
                    train_step += 1
                acc_t = acc_total / train_step
                loss_t = loss_total / train_step
                acc_train.append(acc_t)
                loss_train.append(loss_t)

                # dev
                dev_step = 0
                for input, label, input_len, label_len in self.prepare_data(train_path, dev_list):
                    loss_d, acc_d = sess.run([self.loss, self.acc], {self.input: input, self.label: label,
                                                                     self.is_training: False,
                                                                     self.input_len: input_len,
                                                                     self.label_len: label_len})
                    dev_step += 1
                    acc_dev_total += acc_d
                    loss_dev_total += loss_d
                acc_dd = acc_dev_total / dev_step
                loss_dd = loss_dev_total / dev_step
                acc_dev.append(acc_dd)
                loss_dev.append(loss_dd)
                print('Epoch{}----loss:{},val_loss:{}'.format(step, loss_t, round(loss_dd, 4)))
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
                print('Early Stop at Epoch{}  loss:{}'.format(es_step, es_loss))

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
        plt.savefig(os.path.join(self.PIC_DIC, 'loss.png'))
        plt.close()

    def predict(self, test_path):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.latest_checkpoint(self.MODEL_DIC)  # 找到存储变量值的位置
            saver.restore(sess, ckpt)
            test_list = os.listdir(test_path)
            pre_list = []
            for input, input_len in self.prepare_test_data(test_path, test_list):
                logits = sess.run(self.logits, {self.input: input, self.input_len: input_len, self.is_training: False})
                decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, input_len, merge_repeated=False)
                t = tf.sparse_tensor_to_dense(decoded[0])
                pre_list.extend(sess.run(t))
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
            label_len = []
            input_len = []
            # each
            for d in data_list:
                plabel = self.id2label[d]
                temp = [self.char2index[x] for x in plabel]
                while len(temp) < config.MAXLABEL:
                    temp.append(0)
                label_len.append(len(plabel))
                pinput = []
                file_path = os.path.join(path, d)
                for ff in os.listdir(file_path):
                    pic_path = os.path.join(file_path, ff)
                    im = Image.open(pic_path)
                    im = im.resize((config.WIDTH, config.HEIGHT))
                    matrix = np.asarray(im)
                    pinput.append(matrix)
                input_len.append(len(pinput))
                while len(pinput) < config.PIC_NUM:
                    pinput.append(np.zeros(shape=(config.WIDTH, config.HEIGHT, 3)))
                pinput = np.array(pinput)
                label.append(temp)
                input.append(pinput)
            label = np.array(label)
            input = np.array(input)
            input_len = np.array(input_len)
            label_len = np.array(label_len)
            yield input, label, input_len, label_len

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
            input_len = []
            for d in data_list:
                pinput = []
                file_path = os.path.join(path, d)
                for ff in os.listdir(file_path):
                    pic_path = os.path.join(file_path, ff)
                    im = Image.open(pic_path)
                    im = im.resize((config.WIDTH, config.HEIGHT))
                    matrix = np.asarray(im)
                    pinput.append(matrix)
                input_len.append(len(pinput))
                while len(pinput) < config.PIC_NUM:
                    pinput.append(np.zeros(shape=(config.WIDTH, config.HEIGHT, 3)))
                pinput = np.array(pinput)
                input.append(pinput)
            input = np.array(input)
            input_len = np.array(input_len)
            yield input, input_len

    def load_table(self, table_path):
        with open(table_path, 'rb') as f:
            result = pickle.load(f)
        return result








