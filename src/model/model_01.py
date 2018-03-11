# coding=utf-8
'''
模型（版本1）目标设定：
1，模型设计（准确率要达到95%）
2，模块化（数据录入模块，训练模块，测试模块，参数保存&模型还原）
'''

import math

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


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
                                                               self.keep_prob: 1.0}) *100/ len(x)
        return rate

    def _model_(self):
        self.x = tf.placeholder(tf.float32, [None, 784], name="x")
        self.y = tf.placeholder(tf.float32, [None, 10], name="y")
        x_img = tf.reshape(self.x, [-1, 28, 28, 1])
        self.keep_prob = tf.placeholder(tf.float32)

        W1 = weight_variable([5, 5, 1, 32])
        b1 = bias_variable([32])
        h1 = tf.nn.relu(conv2d(x_img, W1) + b1)
        hp = max_pool_2x2(h1)

        W2 = weight_variable([5, 5, 32, 64])
        b2 = bias_variable([64])
        h2 = tf.nn.relu(conv2d(hp, W2) + b2)
        hp2 = max_pool_2x2(h2)

        W3 = weight_variable([7 * 7 * 64, 1024])
        b3 = bias_variable([1024])
        f3 = tf.reshape(hp2, [-1, 7 * 7 * 64])
        fc3 = tf.nn.relu(tf.matmul(f3, W3) + b3)
        fc3_drop = tf.nn.dropout(fc3, self.keep_prob)

        W4 = weight_variable([1024, 10])
        b4 = bias_variable([10])

        self.pred = tf.nn.softmax(tf.matmul(fc3_drop, W4) + b4)

        # 损失函数
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.pred), reduction_indices=[1]))

        # 训练模型
        self.train_step = tf.train.AdamOptimizer(LR).minimize(self.cross_entropy)

        # 准确率计算
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y, 1), tf.argmax(self.pred, 1)), tf.float32))


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(mnist.train.images[0], mnist.train.labels[0])
print(mnist.train.images.shape, mnist.train.labels.shape)
LR = 1e-4
BATCH_SIZE = 10
STEP_TIMES = 10
UNIT_NUM = 300
KEEP_PROB = 0.75

m = Model01()
for i in range(STEP_TIMES):
    batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
    print(batch_x[0])
    print(batch_y[0])
    m.batch_train(batch_x, batch_y)
    if i % 20 == 0:
        print("%d --> %f : %f" % (i, m.loss, m.rate))

print(m.test(mnist.test.images, mnist.test.labels))
