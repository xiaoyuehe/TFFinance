# coding=utf-8
"""
resnet implimention
"""

import tensorflow as tf
from model02_input import Model02Input
from model_02_v1 import Model02

TEST_NO = '4'

MODEL_BASE = 'D:/StockData/11_MODEL_02/'

STEP_TIMES = 500000
BATCH_SIZE = 500

slim = tf.contrib.slim


def xx_train2():
    m = Model02()
    # m.restore(MODEL_BASE + "MM/m6.cpt-0312-83000")
    inp = Model02Input(MODEL_BASE + "train_sh.csv")

    test_inp = Model02Input(MODEL_BASE + 'test.csv')
    x, y = test_inp.next_train_batch(15000)

    yy_rate = 0.4
    for i in range(STEP_TIMES):
        batch_x, batch_y = inp.next_train_batch(BATCH_SIZE)
        m.batch_train(batch_x, batch_y)

        if i % 20 == 0:
            rate = m.test(x, y)
            print("%d --> %f : %f ; test: %f:%f" % (i, m.loss, m.rate, rate, m.rate - rate))

            if rate > yy_rate + 0.01:
                yy_rate = rate
                m.snapshot(MODEL_BASE + 'SNAP' + TEST_NO + '/m' + TEST_NO + '.cpt-' + str(i) + '-' + str(rate))

        if i % 500 == 0:
            m.snapshot(MODEL_BASE + 'SNAP' + TEST_NO + '/ok_m' + TEST_NO + '.cpt-' + str(i))
            print('-'*30)
            print(m.validation(x, y))



xx_train2()

