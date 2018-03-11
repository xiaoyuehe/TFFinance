# coding=utf-8
'''
模型（版本1）目标设定：
1，模型设计（准确率要达到95%）
2，模块化（数据录入模块，训练模块，测试模块，参数保存&模型还原）
'''

import math

import tensorflow as tf
from model01_input import Model01Input

MODEL_BASE = 'D:/StockData/11_MODEL_01/SNAP/'


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')


class Model01(object):
    def __init__(self):
        self.loss = None
        self.rate = None
        self.__model__()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        pass

    def batch_train(self, batch_x, batch_y):
        _, loss, rr = self.session.run([self.train_step, self.cross_entropy, self.accuracy],
                                       feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: 0.75})
        self.loss = loss
        self.rate = rr

    def test(self, x, y):
        rate = 0
        print(len(x))
        for i in range(math.ceil(len(x) / 100)):
            print("==> " + str(i) + " " + str(min((i + 1) * 100, len(x))))
            rate += self.session.run(self.accuracy, feed_dict={self.x: x[i * 100: min((i + 1) * 100, len(x)), ],
                                                               self.y: y[i * 100: min((i + 1) * 100, len(x)), ],
                                                               self.keep_prob: 1.0}) * 100 / len(x)
        return rate

    def snapshot(self, path):
        saver = tf.train.Saver()
        saver.save(self.session, path)

    def restore(self, path):
        saver = tf.train.Saver()
        saver.restore(self.session, path)

    def __model__(self):
        self.x = tf.placeholder(tf.float32, [None, 5, 5, 5], name="x")
        self.y = tf.placeholder(tf.float32, [None, 3], name="y")
        self.keep_prob = tf.placeholder(tf.float32)

        W1 = weight_variable([3, 3, 5, 32])
        b1 = bias_variable([32])
        h1 = tf.nn.relu(conv2d(self.x, W1) + b1)
        hp = max_pool_2x2(h1)

        W2 = weight_variable([3, 3, 32, 64])
        b2 = bias_variable([64])
        h2 = tf.nn.relu(conv2d(hp, W2) + b2)
        hp2 = max_pool_2x2(h2)

        W3 = weight_variable([3, 3, 64, 128])
        b3 = bias_variable([128])
        h3 = tf.nn.relu(conv2d(hp2, W3) + b3)
        hp3 = max_pool_2x2(h3)

        W4 = weight_variable([3, 3, 128, 64])
        b4 = bias_variable([64])
        h4 = tf.nn.relu(conv2d(hp3, W4) + b4)
        hp4 = max_pool_2x2(h4)

        W5 = weight_variable([5 * 5 * 64, 1024])
        b5 = bias_variable([1024])
        f5 = tf.reshape(hp4, [-1, 5 * 5 * 64])
        fc5 = tf.nn.relu(tf.matmul(f5, W5) + b5)
        fc5_drop = tf.nn.dropout(fc5, self.keep_prob)

        W6 = weight_variable([1024, 3])
        b6 = bias_variable([3])

        self.pred = tf.nn.softmax(tf.matmul(fc5_drop, W6) + b6)

        # 损失函数
        self.cross_entropy = tf.reduce_mean(
            -tf.reduce_sum(self.y * tf.log(tf.clip_by_value(self.pred, 0.05, 0.95)), reduction_indices=[1]))

        # 训练模型
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)

        # 准确率计算
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y, 1), tf.argmax(self.pred, 1)), tf.float32))


# ——————————————————导入数据——————————————————————

STEP_TIMES = 50000
BATCH_SIZE = 500

m = Model01()
m.restore(MODEL_BASE + "m1.cpt-32-0.802")
inp = Model01Input()
max_rate = 0
for i in range(STEP_TIMES):
    batch_x, batch_y = inp.next_train_batch(BATCH_SIZE)
    m.batch_train(batch_x, batch_y)
    if m.rate > max_rate:
        max_rate = m.rate
        m.snapshot(MODEL_BASE + "m1.cpt-" + str(i) + "-" + str(max_rate))
    if i % 20 == 0:
        print("%d --> %f : %f" % (i, m.loss, m.rate))
