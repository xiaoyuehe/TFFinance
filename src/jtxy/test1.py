# coding=utf-8
"""
模型执行器，有一些工具方法可以直接使用
"""

import tensorflow as tf
from datasource.ds1 import Ds1
from framwork.Excutor import SoftMaxExecutor
from models.resnet import resnet_v2_50

TEST_NO = '3'

MODEL_BASE = 'D:/StockData/11_MODEL_02/'

STEP_TIMES = 500000
BATCH_SIZE = 500


def model():
    x = tf.placeholder(tf.float32, [None, 5, 5, 5], name="x")
    y = tf.placeholder(tf.float32, [None, 3], name="y")
    keep_prob = tf.placeholder(tf.float32)
    a, b = resnet_v2_50(x, 3)
    # a, b = resnet_v2_101(x, 3)
    # a, b = resnet_v2_152(x, 3)
    predict = tf.reshape(b['predictions'], [-1, 3])
    loss_weight = tf.constant([0.49, 1.43, 3.73], dtype=tf.float32)
    loss = tf.reduce_mean(
        tf.reduce_sum((tf.multiply(loss_weight, tf.pow(tf.subtract(y, predict), 2)))
                      , reduction_indices=[1]))
    # 训练模型
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    # 准确率计算
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(predict, 1)), tf.float32))
    return SoftMaxExecutor(x, y, predict, loss, train_step, accuracy, keep_prob)


def xx_train2():
    m = model()
    m.begin()
    # m.restore(MODEL_BASE + "MM/m6.cpt-0312-83000")
    inp = Ds1(MODEL_BASE + "train_sh.csv")

    test_inp = Ds1(MODEL_BASE + 'test.csv')
    x, y = test_inp.next_train_batch(5000)

    yy_rate = 0.4
    for i in range(STEP_TIMES):
        batch_x, batch_y = inp.next_train_batch(BATCH_SIZE)
        m.batch_train(batch_x, batch_y)

        if i % 20 == 0:
            loss, rate, result_dict, pair_dict = m.run_validation(x, y)
            print("*" * 40)
            print("%d --> %f : %f \n test: %s\n%s" % (i, loss, rate, result_dict, pair_dict))
            print("*" * 40)

            if rate > yy_rate + 0.01:
                yy_rate = rate
                m.snapshot(MODEL_BASE + 'SNAP' + TEST_NO + '/m' + TEST_NO + '.cpt-' + str(i) + '-' + str(rate))

        if i % 5000 == 0:
            m.snapshot(MODEL_BASE + 'SNAP' + TEST_NO + '/ok_m' + TEST_NO + '.cpt-' + str(i))


xx_train2()
