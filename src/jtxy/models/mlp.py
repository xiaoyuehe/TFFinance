# coding=utf-8
'''
accuracy:98%
'''

import tensorflow as tf


def layer(layer_x, input_size, output_size):
    w = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.1))
    b = tf.Variable(tf.zeros([output_size]))
    return tf.nn.relu(tf.matmul(layer_x, w) + b)


def mlp(input_x, layer_size, keep_prob, output_size):
    layer_x = input_x
    i = 1
    while i < len(layer_size):
        layer_x = layer(layer_x, layer_size[i - 1], layer_size[i])
        i += 1

    dropout = tf.nn.dropout(layer_x, keep_prob)
    if output_size:
        w = tf.Variable(tf.truncated_normal([layer_size[len(layer_size) - 1], output_size], stddev=0.1))
        b = tf.Variable(tf.zeros([output_size]))
        return tf.nn.softmax(tf.matmul(dropout, w) + b)
    else:
        return dropout
