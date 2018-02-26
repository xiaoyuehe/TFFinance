# coding=utf-8
'''
构造rnn网络基础方法
'''

import numpy as np
import pandas as pd
import tensorflow as tf


class RnnConfig(object):
    def __init__(self):
        pass


class SimpleLstmRnn(object):
    def __init__(self, rnn_config):
        self.rnn_config = rnn_config
        self.__build_rnn__()

    def __build_rnn__(self):
        batch_size = self.rnn_config.batch_size
        num_steps = self.rnn_config.num_steps
        input_size = self.rnn_config.input_size
        output_size = self.rnn_config.output_size
        lr = self.rnn_config.lr
        layer_nums = self.rnn_config.layer_nums

        # 处理输入数据
        self.input_holder = tf.placeholder(tf.float32, [batch_size, num_steps, input_size])
        w_in = tf.Variable(tf.random_normal([input_size, layer_nums[0]]))
        b_in = tf.Variable(tf.random_normal([layer_nums[0], ]))
        input_data = tf.reshape(self.input_holder, [-1, input_size])
        input_rnn = tf.matmul(input_data, w_in) + b_in
        input_rnn = tf.reshape(input_rnn, [-1, num_steps, layer_nums[0]])

        # 创建lstm层
        lcl = []
        for i in range(len(layer_nums)):
            size = layer_nums[i]
            cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
            lcl.append(cell)

        cell_layers = tf.nn.rnn_cell.MultiRNNCell(lcl, state_is_tuple=True)
        self.initial_state = cell_layers.zero_state(batch_size, tf.float32)

        inner_state = self.initial_state
        outputs = []
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                print(num_steps)
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, inner_state) = cell_layers(input_rnn[:, time_step, :], inner_state)
                outputs.append(cell_output)
        self.final_state = inner_state

        # 处理输出
        self.output_holder = tf.placeholder(tf.float32, [batch_size, num_steps, output_size])
        output = tf.reshape(outputs, [-1, layer_nums[-1]])  # 作为输出层的输入
        w_out = tf.Variable(tf.random_normal([layer_nums[-1], output_size]))
        b_out = tf.Variable(tf.random_normal([output_size, ]))
        pred = tf.matmul(output, w_out) + b_out

        # 损失函数
        self.loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(self.output_holder, [-1])))
        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)


# ——————————————————导入数据——————————————————————
f = open('601628_2.csv')
df = pd.read_csv(f)  # 读入股票数据
data = np.array(df.loc[:, ['chr', 'exr']])  # 获取最高价序列
normalize_data = data[::-1]  # 反转，使数据按照日期先后顺序排列

# 生成训练集
# 设置常量
rnn_config = RnnConfig()
rnn_config.batch_size = 100
rnn_config.num_steps = 25
rnn_config.input_size = 2
rnn_config.output_size = 1
rnn_config.lr = 0.05
rnn_config.layer_nums = [10, 10]

time_step = rnn_config.num_steps
batch_size = rnn_config.batch_size

train_x, train_y = [], []  # 训练集
for i in range(len(normalize_data) - time_step - 101):
    x = normalize_data[i:i + time_step]
    y = normalize_data[:, :1][i + 1:i + time_step + 1]
    train_x.append(x.tolist())
    train_y.append(y.tolist())

test_x, test_y = [], []  # 训练集
for i in range(len(normalize_data) - time_step - 101, len(normalize_data) - time_step - 1):
    x = normalize_data[i:i + time_step]
    y = normalize_data[:, :1][i + 1:i + time_step + 1]
    test_x.append(x.tolist())
    test_y.append(y.tolist())

rnn = SimpleLstmRnn(rnn_config)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    init_stat = sess.run(rnn.initial_state)

    print(init_stat)
    step = 0
    for i in range(10000):
        start = 0
        end = start + batch_size
        while (end < len(train_x)):
            feed_dict = {rnn.input_holder: train_x[start:end], rnn.output_holder: train_y[start:end]}
            for j, (c, h) in enumerate(rnn.initial_state):
                feed_dict[c] = init_stat[j].c
                feed_dict[h] = init_stat[j].h

            _,loss_value, init_stat = sess.run([rnn.train_op,rnn.loss, rnn.final_state], feed_dict=feed_dict)
            start += batch_size
            end = start + batch_size
            # 每10步保存一次参数
            if step % 5 == 0:
                print(i, step, loss_value)
            step += 1
