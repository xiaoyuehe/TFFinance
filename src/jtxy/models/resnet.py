# coding=utf-8
"""
resnet 实现方案
20180326
"""

import collections

import tensorflow as tf

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


def resnet_v2_35(inputs, num_classes=None, global_pool=True, reuse=None, scope='resnet_v2_35'):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 3 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)
    ]
    return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True, reuse=reuse, scope=scope)


def resnet_v2_26(inputs, num_classes=None, global_pool=True, reuse=None, scope='resnet_v2_26'):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 2)
    ]
    return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True, reuse=reuse, scope=scope)


def resnet_v2_26_half(inputs, num_classes=None, global_pool=True, reuse=None, scope='resnet_v2_26_half'):
    blocks = [
        Block('block1', bottleneck, [(128, 32, 1)] + [(128, 32, 2)]),
        Block('block2', bottleneck, [(256, 64, 1)] + [(256, 64, 2)]),
        Block('block3', bottleneck, [(512, 128, 1)] + [(512, 128, 2)]),
        Block('block4', bottleneck, [(1024, 256, 1)] * 2)
    ]
    return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True, reuse=reuse, scope=scope)


def resnet_v2_20_half(inputs, num_classes=None, global_pool=True, reuse=None, scope='resnet_v2_26_half'):
    blocks = [
        Block('block1', bottleneck, [(128, 32, 1)] + [(128, 32, 2)]),
        Block('block2', bottleneck, [(256, 64, 1)] + [(256, 64, 2)]),
        Block('block3', bottleneck, [(512, 128, 1)] * 2)
    ]
    return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True, reuse=reuse, scope=scope)


def resnet_v2_10_half(inputs, num_classes=None, global_pool=True, reuse=None, scope='resnet_v2_10_half'):
    blocks = [
        Block('block1', bottleneck, [(128, 32, 1)] + [(128, 32, 2)]),
        Block('block2', bottleneck, [(256, 64, 1)]  * 2)
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
