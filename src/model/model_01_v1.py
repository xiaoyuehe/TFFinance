# coding=utf-8
'''
模型（版本1）目标设定：
1，模型设计（准确率要达到95%）
2，模块化（数据录入模块，训练模块，测试模块，参数保存&模型还原）
'''

import math

import numpy as np
import pandas as pd
import tensorflow as tf


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
        self._model_()
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

    def _model_(self):
        self.x = tf.placeholder(tf.float32, [None, 5, 5, 4], name="x")
        self.y = tf.placeholder(tf.float32, [None, 3], name="y")
        self.keep_prob = tf.placeholder(tf.float32)

        W1 = weight_variable([3, 3, 4, 32])
        b1 = bias_variable([32])
        h1 = tf.nn.relu(conv2d(self.x, W1) + b1)
        hp = max_pool_2x2(h1)

        W2 = weight_variable([3, 3, 32, 64])
        b2 = bias_variable([64])
        h2 = tf.nn.relu(conv2d(hp, W2) + b2)
        hp2 = max_pool_2x2(h2)

        W3 = weight_variable([3 * 3 * 64, 1024])
        b3 = bias_variable([1024])
        f3 = tf.reshape(hp2, [-1, 3 * 3 * 64])
        fc3 = tf.nn.relu(tf.matmul(f3, W3) + b3)
        fc3_drop = tf.nn.dropout(fc3, self.keep_prob)

        W4 = weight_variable([1024, 3])
        b4 = bias_variable([3])

        self.pred = tf.nn.softmax(tf.matmul(fc3_drop, W4) + b4)

        # 损失函数
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.pred), reduction_indices=[1]))

        # 训练模型
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)

        # 准确率计算
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y, 1), tf.argmax(self.pred, 1)), tf.float32))


# ——————————————————导入数据——————————————————————
f = open('601628_2.csv')
df = pd.read_csv(f)  # 读入股票数据
data = np.array(df.loc[:, ['ep', 'hp', 'lp', 'chr', 'exr']])  # 获取最高价序列
normalize_data = data[::-1]  # 反转，使数据按照日期先后顺序排列

train_x, train_y = [], []  # 训练集
for i in range(len(normalize_data) - 30):
    x = normalize_data[i:i + time_step]
    y = normalize_data[:, :1][i + 1:i + + 1]
    train_x.append(x.tolist())
    train_y.append(y.tolist())

STEP_TIMES = 100

m = Model01()
for i in range(STEP_TIMES):
    batch_x, batch_y
    m.batch_train(batch_x, batch_y)
    if i % 20 == 0:
        print("%d --> %f : %f" % (i, m.loss, m.rate))

print(m.test(mnist.test.images, mnist.test.labels))
