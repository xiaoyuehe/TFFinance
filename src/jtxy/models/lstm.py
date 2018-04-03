# coding=utf-8
"""
lstm model
"""
import tensorflow as tf


def lstm(input_x, input_size, rnn_unit, output_size):
    batch_size = tf.shape(input_x)[0]
    time_step = tf.shape(input_x)[1]

    w_in = tf.Variable(tf.random_normal([input_size, rnn_unit]))
    b_in = tf.Variable(tf.constant(0.1, shape=[rnn_unit, ]))
    input = tf.reshape(input_x, [-1, input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入

    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state,
                                                 dtype=tf.float32)

    output = tf.reshape(output_rnn, [-1, rnn_unit])  # 作为输出层的输入
    w_out = tf.Variable(tf.random_normal([rnn_unit, output_size]))
    b_out = tf.Variable(tf.constant(0.1, shape=[output_size, ]))
    predict = tf.matmul(output, w_out) + b_out
    return predict, final_states
