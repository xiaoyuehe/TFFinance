# coding=utf-8
"""
模型执行器，有一些工具方法可以直接使用
"""

import tensorflow as tf
from datasource.ds2 import Ds2
from framwork.Excutor import RegressionExcutor
from models.lstm import lstm
from utils.data import cast_float

TEST_NO = 'lstm_01'
MODEL_BASE = '/Users/yaoli/StockData/11_MODEL_02/'
# MODEL_BASE = 'D:/StockData/11_MODEL_02/'

STEP_TIMES = 20000
BATCH_SIZE = 500


def model():
    x = tf.placeholder(tf.float32, [None, 25, 5], name="x")
    y = tf.placeholder(tf.float32, [None, 1], name="y")
    keep_prob = tf.placeholder(tf.float32)
    predict, _ = lstm(x, 5, 20, 1)
    loss = tf.reduce_mean(tf.square(tf.reshape(y, [-1]) - tf.reshape(predict, [-1])))
    # 训练模型
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    # 准确率计算
    accuracy = loss
    return RegressionExcutor(x, y, predict, loss, train_step, accuracy, keep_prob)


def xx_train2():
    m = model()
    m.begin()
    inp = Ds2(MODEL_BASE + "train_sh.csv")
    test_inp = Ds2(MODEL_BASE + 'test.csv')
    x, y = test_inp.next_train_batch_flat_lstm(10000)

    yy_rate = 0.4
    for i in range(STEP_TIMES):
        batch_x, batch_y = inp.next_train_batch_flat_lstm(BATCH_SIZE)
        m.batch_train(batch_x, batch_y)

        if i % 20 == 0:
            loss, rate, result_dict, pair_dict = m.run_validation(x, y, softmax=False)
            print("*" * 40)
            print("%s --> %s\n" % (m.output_loss, m.output_accuracy))
            print("%d --> %f : %f \n test: %s\n%s" % (i, loss, rate, result_dict, pair_dict))
            print("%.2f,%.2f" % (cast_float(result_dict.get("accuracy_10")),
                                 cast_float(result_dict.get("confidence_10"))))
            print("*" * 40)

            if rate > yy_rate + 0.01:
                yy_rate = rate
                m.snapshot(
                    MODEL_BASE + 'SNAP' + TEST_NO + '/m' + TEST_NO + '.cpt-' + str(i) + '-' + str(
                        rate))

        if i % 5000 == 0:
            m.snapshot(MODEL_BASE + 'SNAP' + TEST_NO + '/ok_m' + TEST_NO + '.cpt-' + str(i))


xx_train2()
