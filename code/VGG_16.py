import tensorflow as tf
import config
import matplotlib.pyplot as plt


class Model:
    def __init__(self):
        # batch, maxnum, 256*256
        self.input = tf.placeholder(shape=[None, None, None])
        # batch, type_num
        self.label = tf.placeholder(shape=[None, None])
        self.score, self.acc, self.loss, self.train_step = self.create_graph()
        self.mode = 'train'

    def create_graph(self):
        cnn_input = self.input
        cnn_input = tf.expand_dims(cnn_input, axis=3)
        with tf.variable_scope('cnn1_1'):
            channel = cnn_input.shape[-1]
            filter_size = [config.FILTER_SIZE, config.FILTER_SIZE, channel, config.FILTER_NUM1]
            w = tf.get_variable(name='weight', shape=filter_size, initializer=tf.truncated_normal())
            b = tf.get_variable(name='bias', shape=[config.FILTER_NUM1], initializer=tf.zeros_initializer())
            # batch, pic_num, width, height, filter_num
            conv1_1 = tf.nn.conv2d(cnn_input, w, strides=[1, 1, 1, 1], padding='SAME')
            conv1_1 = tf.nn.bias_add(conv1_1, b)
            # activation
            conv1_1 = tf.nn.relu(conv1_1)
            cnn_input = conv1_1

        with tf.variable_scope('cnn1_2'):
            channel = cnn_input.shape[-1]
            filter_size = [config.FILTER_SIZE, config.FILTER_SIZE, channel, config.FILTER_NUM1]
            w = tf.get_variable(name='weight', shape=filter_size, initializer=tf.truncated_normal())
            b = tf.get_variable(name='bias', shape=[config.FILTER_NUM1], initializer=tf.zeros_initializer())
            # batch, pic_num, width, height, filter_num
            conv1_2 = tf.nn.conv2d(cnn_input, w, strides=[1, 1, 1, 1], padding='SAME')
            conv1_2 = tf.nn.bias_add(conv1_2, b)
            # activation
            conv1_2 = tf.nn.relu(conv1_2)
            cnn_input = conv1_2

        maxpool1 = tf.nn.max_pool(cnn_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool1')
        cnn_input = maxpool1

        with tf.variable_scope('cnn2_1'):
            channel = cnn_input.shape[-1]
            filter_size = [config.FILTER_SIZE, config.FILTER_SIZE, channel, config.FILTER_NUM2]
            w = tf.get_variable(name='weight', shape=filter_size, initializer=tf.truncated_normal())
            b = tf.get_variable(name='bias', shape=[config.FILTER_NUM2], initializer=tf.zeros_initializer())
            # batch, pic_num, width, height, filter_num
            conv2_1 = tf.nn.conv2d(cnn_input, w, strides=[1, 1, 1, 1], padding='SAME')
            conv2_1 = tf.nn.bias_add(conv2_1, b)
            # activation
            conv2_1 = tf.nn.relu(conv2_1)
            cnn_input = conv2_1

        with tf.variable_scope('cnn2_2'):
            channel = cnn_input.shape[-1]
            filter_size = [config.FILTER_SIZE, config.FILTER_SIZE, channel, config.FILTER_NUM2]
            w = tf.get_variable(name='weight', shape=filter_size, initializer=tf.truncated_normal())
            b = tf.get_variable(name='bias', shape=[config.FILTER_NUM1], initializer=tf.zeros_initializer())
            # batch, pic_num, width, height, filter_num
            conv2_2 = tf.nn.conv2d(cnn_input, w, strides=[1, 1, 1, 1], padding='SAME')
            conv2_2 = tf.nn.bias_add(conv2_2, b)
            # activation
            conv2_2 = tf.nn.relu(conv2_2)
            cnn_input = conv2_2

        maxpool2 = tf.nn.max_pool(cnn_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool2')
        # cnn_output (batch, maxnum, 256*256, filter_num)
        cnn_output = maxpool2
        # Flatten
        flatten_size = config.WIDTH*config.HEIGHT*config.FILTER_NUM2
        cnn_output_flatten = tf.reshape(cnn_output, shape=[-1, config.PIC_NUM, flatten_size])

        with tf.variable_scope('dense1'):
            w = tf.get_variable(name='weight', shape=[flatten_size, config.HIDDEN_SIZE1],
                                initializer=tf.truncated_normal())
            b = tf.get_variable(name='bias', shape=[config.HIDDEN_SIZE1])
            temp = tf.reshape(cnn_output_flatten, shape=[-1, flatten_size])
            dense1_output = tf.matmul(temp, w)
            dense1_output = tf.nn.bias_add(dense1_output, b)
            dense1_output = tf.nn.relu(dense1_output)

        if self.mode == 'train':
            dense1_output = tf.nn.dropout(dense1_output, config.DROP)

        with tf.variable_scope('dense2'):
            w = tf.get_variable(name='weight', shape=[config.HIDDEN_SIZE1, config.HIDDEN_SIZE2],
                                initializer=tf.truncated_normal())
            b = tf.get_variable(name='bias', shape=[config.HIDDEN_SIZE2])
            dense2_output = tf.matmul(dense1_output, w)
            dense2_output = tf.nn.bias_add(dense2_output, b)
            dense2_output = tf.nn.relu(dense2_output)

        if self.mode == 'train':
            dense2_output = tf.nn.dropout(dense2_output, config.DROP)

        lstm_input = tf.reshape(dense2_output, shape=[-1, config.PIC_NUM, config.HIDDEN_SIZE2])

        with tf.variable_scope('lstm'):
            cell = tf.contrib.rnn.BasicLSTMCell(config.LSTM_NUM)
            if self.mode == 'train':
                cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=config.LSTMDROP)
            lstm_out, _ = tf.nn.dynamic_rnn(cell, lstm_input, dtype=tf.float32)
            # batch, lstm_size
            lstm_output = lstm_out[:, -1, :]
            lstm_output = tf.nn.relu(lstm_output)

        with tf.variable_scope('dense3'):
            w = tf.get_variable(name='weight', shape=[config.LSTM_NUM, config.HIDDEN_SIZE3],
                                initializer=tf.truncated_normal())
            b = tf.get_variable(name='bias', shape=[config.HIDDEN_SIZE3])
            dense3_output = tf.matmul(lstm_output, w)
            dense3_output = tf.nn.bias_add(dense3_output, b)
            dense3_output = tf.nn.relu(dense3_output)

        if self.mode == 'train':
            dense3_output = tf.nn.dropout(dense3_output, config.DROP)

        with tf.variable_scope('dense4'):
            w = tf.get_variable(name='weight', shape=[config.HIDDEN_SIZE3, config.TYPE_NUM],
                                initializer=tf.truncated_normal())
            b = tf.get_variable(name='bias', shape=[config.TYPE_NUM])
            dense4_output = tf.matmul(dense3_output, w)
            dense4_output = tf.nn.bias_add(dense4_output, b)

        score = dense4_output

        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(score, 1), tf.argmax(self.label, 1)), dtype=tf.float32))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=score))
        train_step = tf.train.AdamOptimizer(config.LR).minimize(loss)

        return score, acc, loss, train_step

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
                num = len(train_examples) // config.BATCH_SIZE + 1
                for i in range(num):
                    end = begin + config.BATCH_SIZE
                    if end > len(train_examples):
                        end = len(train_examples)
                    train_batch = train_examples[begin:end]
                    input_batch = [x.input for x in train_batch]
                    label_batch = [x.label for x in train_batch]
                    begin = end
                    _, acc_t, loss_t = sess.run([self.train_step, self.acc, self.loss], {self.input: input_batch,
                                                                                         self.label: label_batch})
                    acc_total += acc_t
                    loss_total += loss_t
                    print('step:{}  [{}/{}]'.format(i, end, len(train_examples)))
                    print('acc:{}, loss:{}'.format(acc_t, loss_t))

                acc_t = acc_total/num
                loss_t = loss_total/num
                acc_train.append(acc_t)
                loss_train.append(loss_t)

                input_dev = [x.input for x in dev_examples]
                label_dev = [x.label for x in dev_examples]
                acc_d, loss_d = sess.run([self.acc, self.loss], {self.input: input_dev, self.label: label_dev})
                acc_dev.append(acc_d)
                loss_dev.append(loss_d)

                print('Epoch{}----acc:{},loss:{},val_acc:{},val_loss:{}'.format(step, acc_t, loss_t, acc_d, loss_d))
                if loss_d > loss_stop:
                    if n >= config.EARLY_STEP:
                        should_stop = True
                    else:
                        n += 1
                else:
                    saver.save(sess, config.MODEL_PATH)
                    es_loss = loss_d
                    es_acc = acc_d
                    es_step = step
                    n = 0
                    loss_stop = loss_d
                step += 1

            if should_stop:
                print('Early Stop at Epoch{} acc:{} loss:{}'.format(es_step, es_loss, es_acc))

        plt.plot(acc_train)
        plt.plot(acc_dev)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(config.RESULT_ACC_PATH)
        plt.close()

        plt.plot(loss_train)
        plt.plot(loss_dev)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(config.RESULT_LOSS_PATH)
        plt.close()

    def predict(self, test_examples):
        self.mode = 'test'
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.latest_checkpoint(self.MODEL_DIC)  # 找到存储变量值的位置
            saver.restore(sess, ckpt)
            test_batch = [x.input for x in test_examples]
            predict = sess.run(self.score, {self.x: test_batch})
            return predict
