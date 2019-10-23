import tensorflow as tf
import config
import matplotlib.pyplot as plt
import numpy as np
import math
import os
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
        self.mode = 'train'
        self.MODEL_PATH = '../result/model/Dense/dense'
        self.MODEL_DIC = '../result/model/Dense'
        self.ACC_DIC = '../result/pic/LipNet'
        self.LOSS_DIC = '../result/pic/LipNet'
        self.PIC_DIC = '../result/pic/LipNet'
        self.score, self.acc, self.loss, self.train_step, self.t = self.create_graph()

    def create_graph(self):
        # CNN
        # shape (batch, depth, width, heigh, channel=1)
        cnn_input = tf.reshape(self.input, shape=[-1, config.PIC_NUM, config.WIDTH, config.HEIGHT])
        print('cnn_input shape:', cnn_input.shape)
        cnn_input_ex = tf.expand_dims(cnn_input, axis=-1)
        print('cnn_input_ex shape:', cnn_input_ex.shape)

        conv = tf.layers.conv3d(cnn_input_ex, config.FILTER_NUM1, kernel_size=(5, 7, 7), strides=(1, 2, 2),
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

    def train(self, train_examples, dev_examples):
        self.mode = 'train'
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
                num = len(train_examples) // config.BATCH_SIZE + 1
                for i in range(num):
                    end = begin + config.BATCH_SIZE
                    if end > len(train_examples):
                        end = len(train_examples)
                    train_batch = train_examples[begin:end]
                    input_batch = [x.input for x in train_batch]
                    label_batch = [x.label for x in train_batch]
                    seqlen_batch = [x.seqlen for x in train_batch]
                    begin = end
                    _, acc_t, loss_t, t = sess.run([self.train_step, self.acc, self.loss, self.t],
                                                               {self.input: input_batch,
                                                                self.label: label_batch
                                                                })
                    # print(t)
                    # print(t.shape)
                    acc_total += acc_t
                    loss_total += loss_t
                    print('step:{}  [{}/{}]  --acc:{}, --loss:{}'.format(i, end, len(train_examples), round(acc_t, 2),
                                                                         round(loss_t, 4)))

                acc_t = acc_total/num
                loss_t = loss_total/num
                acc_train.append(acc_t)
                loss_train.append(loss_t)

                begin = 0
                if len(dev_examples) % 500 == 0:
                    num = len(dev_examples)//500
                else:
                    num = len(dev_examples)//500+1
                for i in range(num):
                    end = begin + 500
                    if end > len(dev_examples):
                        end = len(dev_examples)
                    dev_batch = dev_examples[begin:end]
                    input_dev = [x.input for x in dev_batch]
                    label_dev = [x.label for x in dev_batch]
                    begin = end
                    acc_d, loss_d = sess.run([self.acc, self.loss], {self.input: input_dev, self.label: label_dev})
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
        plt.savefig(self.RESULT_ACC_PATH)
        plt.close()

        plt.plot(loss_train)
        plt.plot(loss_dev)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(self.RESULT_LOSS_PATH)
        plt.close()

    def predict(self, test_examples):
        self.mode = 'test'
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.latest_checkpoint(self.MODEL_DIC)  # 找到存储变量值的位置
            saver.restore(sess, ckpt)
            test_batch = [x.input for x in test_examples]
            seqlen_batch = [x.seqlen for x in test_examples]
            predict = sess.run(self.score, {self.input: test_batch})
            return predict


if __name__ == '__main__':
     m = Model()