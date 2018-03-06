# coding=utf-8
'''
accuracy:98%
'''

import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def var_with_weight_loss(shape, stddev, wl):
    vv = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(vv), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return vv


WIDTH = 5
HEIGHT = 5
CHANNEL = 4
Y_SIZE = 1
LR = 1e-4
BATCH_SIZE = 100
STEP_TIMES = 8000
KEEP_PROB = 0.75

x = tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, CHANNEL])
y = tf.placeholder(tf.float32, [None, Y_SIZE])
keep_prob = tf.placeholder(tf.float32)

W1 = var_with_weight_loss(shape=[5, 5, 4, 64], stddev=5e-2, wl=0.0)
k1 = tf.nn.conv2d(x, W1, [1, 1, 1, 1], padding='SAME')
b1 = tf.Variable(tf.constant(0.0, shape=[64]))
h1 = tf.nn.relu(tf.nn.bias_add(k1, b1))
hp1 = tf.nn.max_pool(h1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
n1 = tf.nn.lrn(hp1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

W2 = var_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, wl=0.0)
k2 = tf.nn.conv2d(n1, W2, [1, 1, 1, 1], padding='SAME')
b2 = tf.Variable(tf.constant(0.1, shape=[64]))
h2 = tf.nn.relu(tf.nn.bias_add(k2, b2))
hp2 = tf.nn.max_pool(h2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

reshape=tf.reshape(hp2,[BATCH_SIZE,-1])
dim = reshape.get_shape()[1].value

W3 = var_with_weight_loss(shape=[dim, 125], stddev=0.04, wl=0.004)
b3 = tf.Variable(tf.constant(0.1, shape=[125]))
local3=tf.nn.relu(tf.matmul(reshape,W3)+b3)

W4 = var_with_weight_loss(shape=[125, 62], stddev=0.04, wl=0.004)
b4 = tf.Variable(tf.constant(0.1, shape=[62]))
local4=tf.nn.relu(tf.matmul(local3,W4)+b4)

W5 = var_with_weight_loss(shape=[62, 3], stddev=1/62.0, wl=0.0)
b5 = tf.Variable(tf.constant(0.0, shape=[3]))
logits=tf.nn.relu(tf.matmul(local4,W5)+b5)

W2 = weight_variable([5, 5, 32, 64])
b2 = bias_variable([64])
h2 = tf.nn.relu(conv2d(hp, W2) + b2)
hp2 = max_pool_2x2(h2)

W3 = weight_variable([7 * 7 * 64, 1024])
b3 = bias_variable([1024])
f3 = tf.reshape(hp2, [-1, 7 * 7 * 64])
fc3 = tf.nn.relu(tf.matmul(f3, W3) + b3)
fc3_drop = tf.nn.dropout(fc3, keep_prob)

W4 = weight_variable([1024, 10])
b4 = bias_variable([10])
pred = tf.nn.softmax(tf.matmul(fc3_drop, W4) + b4)

# 损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=[1]))

# 训练模型
train_step = tf.train.AdamOptimizer(LR).minimize(cross_entropy)

# 准确率计算
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1)), tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(STEP_TIMES):
        batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
        _, loss, rr = sess.run([train_step, cross_entropy, accuracy],
                               feed_dict={x: batch_x, y: batch_y, keep_prob: KEEP_PROB})

        if i % 20 == 0:
            print("%d --> %f : %f" % (i, loss, rr))

    # print(sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels,keep_prob:1.0}))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))
