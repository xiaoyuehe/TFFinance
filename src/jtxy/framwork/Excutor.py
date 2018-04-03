# coding=utf-8
"""
模型执行器，有一些工具方法可以直接使用
"""

import math

import numpy as np
import tensorflow as tf


class Executor(object):
    def __init__(self, x, y, predict, loss, train_step, accuracy, keep_prob=1.0):
        self.x = x
        self.y = y
        self.predict = predict
        self.loss = loss
        self.accuracy = accuracy
        self.train_step = train_step
        self.keep_prob = keep_prob
        self.session = tf.Session()

        self.output_loss = None
        self.output_predict = None
        self.output_accuracy = None

    def begin(self):
        self.session.run(tf.global_variables_initializer())

    def end(self):
        self.session.close()

    def snapshot(self, path):
        saver = tf.train.Saver()
        saver.save(self.session, path)

    def restore(self, path):
        saver = tf.train.Saver()
        saver.restore(self.session, path)

    def batch_train(self, batch_x, batch_y, keep_prob=1.0):
        _, predict, loss, accuracy = self.session.run(
            [self.train_step, self.predict, self.loss, self.accuracy],
            feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: keep_prob})
        self.output_predict = predict
        self.output_loss = loss
        self.output_accuracy = accuracy

    def run_predict(self, x):
        return self.session.run(self.predict, feed_dict={self.x: x, self.keep_prob: 1.0})

    def run_loss(self, x, y):
        batch_size = 100
        count = math.ceil(len(x) / batch_size)
        loss = 0
        for i in range(count):
            cbz = batch_size if i < count else (len(x) - i * batch_size)
            cl, ca = self.session.run([self.loss],
                                      feed_dict={self.x: x[i * batch_size: min((i + 1) * batch_size,
                                                                               len(x)), ],
                                                 self.y: y[i * batch_size: min((i + 1) * batch_size,
                                                                               len(x)), ],
                                                 self.keep_prob: 1.0})
            loss += cl * cbz / len(x)
        return loss

    def run_test(self, x, y):
        batch_size = 100
        count = math.ceil(len(x) / batch_size)
        loss = 0
        rate = 0
        for i in range(count):
            cbz = batch_size if i < count else (len(x) - i * batch_size)
            cl, ca = self.session.run([self.loss, self.accuracy],
                                      feed_dict={self.x: x[i * batch_size: min((i + 1) * batch_size,
                                                                               len(x)), ],
                                                 self.y: y[i * batch_size: min((i + 1) * batch_size,
                                                                               len(x)), ],
                                                 self.keep_prob: 1.0})
            loss += cl * cbz / len(x)
            rate += ca * cbz / len(x)
        return loss, rate


def accr_rate(y, predict, pair_dict, result_dict, actual_dict, pred_dict, softmax=True):
    for j in range(len(y)):
        if softmax:
            actual_key = str(np.argmax(y[j]))
            predict_key = str(np.argmax(predict[j]))
        else:
            actual_key = str(y[j][0])
            predict_key = str(int(predict[j][0]))
        merge_key = actual_key + '_' + predict_key
        pair_dict[merge_key] = 1 if not pair_dict.get(merge_key) else pair_dict[merge_key] + 1
        actual_dict[actual_key] = 1 if not actual_dict.get(actual_key) else actual_dict[
                                                                                actual_key] + 1
        pred_dict[predict_key] = 1 if not pred_dict.get(predict_key) else pred_dict[predict_key] + 1

    for key in actual_dict:
        actual_number = 0 if not actual_dict.get(key) else float(actual_dict[key])
        predict_number = 0 if not pred_dict.get(key) else float(pred_dict[key])
        pair_number = 0 if not pair_dict.get(key + "_" + key) else float(pair_dict[key + "_" + key])
        result_dict["confidence_" + key] = 0 if actual_number == 0 else pair_number / actual_number
        result_dict["accuracy_" + key] = 0 if predict_number == 0 else pair_number / predict_number


class SoftMaxExecutor(Executor):
    def __init__(self, x, y, predict, loss, train_step, accuracy, keep_prob=1.0):
        super(SoftMaxExecutor, self).__init__(x, y, predict, loss, train_step, accuracy, keep_prob)

    def run_validation(self, x, y, softmax=True):
        pair_dict = {}
        result_dict = {}
        actual_dict = {}
        pred_dict = {}
        batch_size = 100
        count = int(math.ceil(len(x) / batch_size))
        loss = 0
        rate = 0
        for i in range(count):
            cbz = batch_size if i < count else (len(x) - i * batch_size)
            cx = x[i * batch_size: min((i + 1) * batch_size, len(x)), ]
            cy = y[i * batch_size: min((i + 1) * batch_size, len(x)), ]
            cl, ca, cp = self.session.run([self.loss, self.accuracy, self.predict],
                                          feed_dict={self.x: cx, self.y: cy, self.keep_prob: 1.0})
            loss += cl * cbz / len(x)
            rate += ca * cbz / len(x)
            accr_rate(cy, cp, pair_dict, result_dict, actual_dict, pred_dict, softmax)
        return loss, rate, result_dict, pair_dict


class RegressionExcutor(Executor):
    def __init__(self, x, y, predict, loss, train_step, accuracy, keep_prob=1.0):
        super(RegressionExcutor, self).__init__(x, y, predict, loss, train_step, accuracy,
                                                keep_prob)


def square_loss(y, predict):
    loss = tf.reduce_mean(tf.square(tf.reshape(y, [-1]) - tf.reshape(predict, [-1])))
    return loss
