# coding=utf-8
'''
accuracy:98%
'''

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print(mnist.train.images.shape, mnist.train.labels.shape)

X_SIZE = 784
Y_SIZE = 10
LR = 0.5
BATCH_SIZE = 100
STEP_TIMES = 4000
UNIT_NUM = 300
UNIT_NUM2 = 100
KEEP_PROB = 0.75

x = tf.placeholder(tf.float32, [None, X_SIZE])
y = tf.placeholder(tf.float32, [None, Y_SIZE])
keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.truncated_normal([X_SIZE, UNIT_NUM], stddev=0.1))
b1 = tf.Variable(tf.zeros([UNIT_NUM]))

W2 = tf.Variable(tf.truncated_normal([UNIT_NUM, UNIT_NUM2], stddev=0.1))
b2 = tf.Variable(tf.zeros([UNIT_NUM2]))

OW = tf.Variable(tf.zeros([UNIT_NUM2, Y_SIZE]))
Ob = tf.Variable(tf.zeros([Y_SIZE]))

# 模型定义
h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
h2_dropout = tf.nn.dropout(h2, keep_prob)
pred = tf.nn.softmax(tf.matmul(h2_dropout, OW) + Ob)

# 损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=[1]))

# 训练模型
train_step = tf.train.GradientDescentOptimizer(LR).minimize(cross_entropy)

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

    print(sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0}))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))
