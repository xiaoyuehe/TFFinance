# coding=utf-8
'''
resnet implimention
'''

import collections
import math

import numpy as np
import tensorflow as tf
from model01_input import Model01Input

MODEL_BASE = 'D:/StockData/11_MODEL_01/'

slim = tf.contrib.slim


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    'resnet block'


@slim.add_arg_scope
def subsample(inputs, factor, scope=None):
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


@slim.add_arg_scope
def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None):
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, padding='SAME', scope=scope)
    else:
        pad_total = kernel_size - 1
        pad_begin = pad_total // 2
        pad_end = pad_total - pad_begin
        inputs = tf.pad(inputs, [[0, 0], [pad_begin, pad_end], [pad_begin, pad_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride, padding='VALID', scope=scope)


@slim.add_arg_scope
def stack_block_dense(net, blocks, outputs_collections=None):
    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            for i, unit in enumerate(block.args):
                with tf.variable_scope('unit_%d' % (i - 1), values=[net]):
                    unit_dept, unit_dept_bottleneck, unit_stride = unit
                    net = block.unit_fn(net, depth=unit_dept, depth_bottleneck=unit_dept_bottleneck, stride=unit_stride)
                net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

    return net


def renet_arg_scope(is_training=True, weight_decay=0.001, batch_norm_decay=0.997, batch_norm_epsilon=1e-5,
                    batch_norm_scale=True):
    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'update_collections': tf.GraphKeys.UPDATE_OPS,
    }
    with slim.arg_scope([slim.conv2d], weight_regularizer=slim.l2_regularizer(weight_decay),
                        weight_initializer=slim.variance_scaling_initializer(),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_num,
                        normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as sc:
                return sc


@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, outputs_collections=None, scope=None):
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')

        if depth_in == depth:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride, normalizer_fn=None, activation_fn=None,
                                   scope='shortcut')

        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1, scope='conv1')
        residual = conv2d_same(residual, depth_bottleneck, 3, stride, scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1, normalizer_fn=None, activation_fn=None, scope='conv3')
        output = shortcut + residual
        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)


def resnet_v2(inputs, blocks, num_classes=None, global_pool=True, include_root_block=True, reuse=None, scope=None):
    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        # 定义reuse说明该scope下的变量是共享的
        end_points_collections = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck, stack_block_dense], outputs_collections=end_points_collections):
            net = inputs
            if include_root_block:
                with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
                    # net = conv2d_same(net, 64, 7, stride=2, scope='conv1')
                    net = conv2d_same(net, 64, 3, stride=2, scope='conv1')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')

            net = stack_block_dense(net, blocks)
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')

            if global_pool:
                net = tf.reduce_mean(net, [1, 2], name='pool5', keepdims=True)
            if num_classes is not None:
                net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits')

            end_points = slim.utils.convert_collection_to_dict(end_points_collections)
            if num_classes is not None:
                end_points['predictions'] = slim.softmax(net, scope='predictions')

            return net, end_points


def resnet_v2_50(inputs, num_classes=None, global_pool=True, reuse=None, scope='resnet_v2_50'):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 2 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)
    ]
    return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True, reuse=reuse, scope=scope)


def resnet_v2_101(inputs, num_classes=None, global_pool=True, reuse=None, scope='resnet_v2_101'):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)
    ]
    return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True, reuse=reuse, scope=scope)


def resnet_v2_152(inputs, num_classes=None, global_pool=True, reuse=None, scope='resnet_v2_152'):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)
    ]
    return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True, reuse=reuse, scope=scope)


class Model02(object):
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
        # print(len(x))
        for i in range(math.ceil(len(x) / 100)):
            # print("==> " + str(i) + " " + str(min((i + 1) * 100, len(x))))
            rate += self.session.run(self.accuracy, feed_dict={self.x: x[i * 100: min((i + 1) * 100, len(x)), ],
                                                               self.y: y[i * 100: min((i + 1) * 100, len(x)), ],
                                                               self.keep_prob: 1.0}) * 100 / len(x)
        return rate

    def validation(self, x, y):
        result_dict = {}
        for i in range(math.ceil(len(x) / 100)):
            # for i in range(1):
            # print("==> " + str(i) + " " + str(min((i + 1) * 100, len(x))))
            cx = x[i * 100: min((i + 1) * 100, len(x)), ]
            cy = y[i * 100: min((i + 1) * 100, len(x)), ]
            result = self.session.run(self.pred, feed_dict={self.x: cx, self.y: cy,
                                                            self.keep_prob: 1.0}) * 100 / len(x)

            for j in range(len(cy)):
                key = str(np.argmax(cy[j])) + '_' + str(np.argmax(result[j]))
                if key in result_dict:
                    result_dict[key] = result_dict[key] + 1
                else:
                    result_dict[key] = 1

        return result_dict

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

        # a, b = resnet_v2_50(self.x, 3)
        a, b = resnet_v2_152(self.x, 3)

        self.pred = tf.reshape(b['predictions'], [-1, 3])

        # 损失函数
        self.cross_entropy = tf.reduce_mean(
            -tf.reduce_sum(self.y * tf.log(self.pred), reduction_indices=[1]))
        # self.cross_entropy = tf.reduce_mean(
        #     -tf.reduce_sum(self.y * tf.log(tf.clip_by_value(self.pred, 0.05, 0.95)), reduction_indices=[1]))

        # 训练模型
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)

        # 准确率计算
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.y, 1), tf.argmax(tf.reshape(self.pred, [-1, 3]), 1)), tf.float32))


# ——————————————————导入数据——————————————————————


STEP_TIMES = 50000
BATCH_SIZE = 100


def xx_train():
    m = Model02()
    inp = Model01Input()
    max_rate = 0.7
    for i in range(STEP_TIMES):
        batch_x, batch_y = inp.next_train_batch(BATCH_SIZE)
        m.batch_train(batch_x, batch_y)
        if m.rate > max_rate:
            max_rate = m.rate
            m.snapshot(MODEL_BASE + "m2.cpt-" + str(i) + "-" + str(max_rate))
        if i % 20 == 0:
            print("%d --> %f : %f" % (i, m.loss, m.rate))


def yy_test():
    m = Model02()
    m.restore(MODEL_BASE + "SNAP4/m2.cpt-127-0.6673999997973442")
    inp = Model01Input(MODEL_BASE + 'data4.csv')
    x, y = inp.next_train_batch(10000)
    rate = m.test(x, y)
    print("-------------> %f" % (rate))


def xx_train2():
    m = Model02()
    inp = Model01Input()
    yy_rate = 0.64
    x, y = inp.next_train_batch(10000)
    for i in range(STEP_TIMES):
        batch_x, batch_y = inp.next_train_batch(BATCH_SIZE)
        m.batch_train(batch_x, batch_y)

        rate = m.test(x, y)
        if rate > yy_rate + 0.005:
            yy_rate = rate
            m.snapshot(MODEL_BASE + "m2.cpt-" + str(i) + "-" + str(rate))

        if i % 20 == 0:
            print("%d --> %f : %f" % (i, m.loss, rate))


def yy_validation():
    m = Model02()
    m.restore(MODEL_BASE + "SNAP4/m2.cpt-127-0.6673999997973442")
    # inp = Model01Input(MODEL_BASE + 'data4.csv')
    inp = Model01Input(MODEL_BASE + 'data3.csv')
    x, y = inp.next_train_batch(15000)
    print(m.validation(x, y))
    # print("-------------> %f" % (rate))


# xx_train2()
# xx_train()
# yy_test()
yy_validation()
