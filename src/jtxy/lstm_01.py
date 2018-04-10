# coding=utf-8
"""
模型执行器，有一些工具方法可以直接使用
"""

import tensorflow as tf
from datasource.lstm_ds import Ds
from framwork.Excutor import RegressionExcutor
from models.lstm import lstm
from utils.data import cast_float

TEST_NO = 'lstm_01'
MODEL_BASE = '/Users/yaoli/StockData/11_MODEL_01_mac/'
# MODEL_BASE = 'D:/StockData/11_MODEL_02/'

EPOCH_COUNT = 20000
STEP_TIMES = 20


def model():
    x = tf.placeholder(tf.float32, [None, 20, 5], name="x")
    y = tf.placeholder(tf.float32, [None, 20, 1], name="y")
    keep_prob = tf.placeholder(tf.float32)
    predict, _ = lstm(x, 5, 25, 1)
    loss = tf.reduce_mean(tf.square(tf.reshape(y[-1], [-1]) - tf.reshape(predict[-1], [-1])))
    # 训练模型
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    # 准确率计算
    accuracy = loss
    return RegressionExcutor(x, y, predict, loss, train_step, accuracy, keep_prob)


def xx_train2():
    m = model()
    m.begin()
    inp = Ds(MODEL_BASE + "train.csv")
    test_inp = Ds(MODEL_BASE + 'test.csv')
    x, y = test_inp.next_train_batch(20, batch_size=9)
    batch_x, batch_y = inp.next_train_batch(STEP_TIMES, batch_size=20)
    for i in range(EPOCH_COUNT):
        indx = i % 20
        m.batch_train(batch_x[indx:indx + 1], batch_y[indx:indx + 1])

        if i % 20 == 0:
            loss = m.run_loss(x, y, batch_size=1)
            print("*" * 40)
            print("%d --> %.6f" % (i, m.output_loss))
            print("%d --> %.6f" % (i, loss))
            print("*" * 40)

        if i % 500 == 0:
            m.snapshot(MODEL_BASE + 'SNAP_' + TEST_NO + '/' + str(i))


xx_train2()
