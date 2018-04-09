# coding=utf-8
"""
模型执行器，有一些工具方法可以直接使用
"""

import tensorflow as tf
from datasource.ds2 import Ds2
from framwork.Excutor import SoftMaxExecutor
from models.resnet import resnet_v2_10_half
from utils.data import cast_float

TEST_NO = '9'

# MODEL_BASE = '/Users/yaoli/StockData/11_MODEL_02/'
MODEL_BASE = 'D:/StockData/11_MODEL_02/'

STEP_TIMES = 5000000
BATCH_SIZE = 500


def model():
    x = tf.placeholder(tf.float32, [None, 5, 5, 5], name="x")
    y = tf.placeholder(tf.float32, [None, 10], name="y")
    keep_prob = tf.placeholder(tf.float32)
    a, b = resnet_v2_10_half(x, num_classes=10, global_pool=False)
    # a, b = resnet_v2_101(x, 3)
    # a, b = resnet_v2_152(x, 3)
    predict = tf.reshape(b['predictions'], [-1, 10])
    # loss_weight = tf.constant([0.49, 1.43, 3.73], dtype=tf.float32)
    # loss = tf.reduce_mean(
    #     tf.reduce_sum((tf.multiply(loss_weight, tf.pow(tf.subtract(y, predict), 2)))
    #                   , reduction_indices=[1]))
    loss = tf.reduce_mean(
        -tf.reduce_sum(y * tf.log(tf.clip_by_value(predict, 1e-4, 1.0)),
                       reduction_indices=[1]))
    # 训练模型
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    # 准确率计算
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(predict, 1)), tf.float32))
    return SoftMaxExecutor(x, y, predict, loss, train_step, accuracy, keep_prob)


def xx_train2():
    m = model()
    m.begin()
    m.restore(MODEL_BASE + "SNAP9/ok_m9.cpt-5000")
    inp = Ds2(MODEL_BASE + "train_sh.csv")

    test_inp = Ds2(MODEL_BASE + 'test.csv')
    x, y = test_inp.next_train_batch(5000)

    yy_rate = 0.4
    for i in range(STEP_TIMES):
        batch_x, batch_y = inp.next_train_batch(BATCH_SIZE)
        m.batch_train(batch_x, batch_y)

        if i % 20 == 0:
            loss, rate, result_dict, pair_dict = m.run_validation(x, y)
            print("*" * 40)
            print("%d --> %f : %f" % (i, m.output_loss, m.output_accuracy))
            print("%f : %f \n test: %s\n%s" % (loss, rate, result_dict, pair_dict))
            print("%.2f,%.2f" % (cast_float(result_dict.get("accuracy_9")),
                                 cast_float(result_dict.get("confidence_9"))))
            print("*" * 40)

            if rate > yy_rate + 0.01:
                yy_rate = rate
                m.snapshot(
                    MODEL_BASE + 'SNAP' + TEST_NO + '/m' + TEST_NO + '.cpt-' + str(i) + '-' + str(
                        rate))

        if i % 5000 == 0:
            m.snapshot(MODEL_BASE + 'SNAP' + TEST_NO + '/ok_m' + TEST_NO + '.cpt-' + str(i))


xx_train2()
