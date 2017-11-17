# coding=utf-8
'''
accuracy:92%
'''

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print(mnist.train.images.shape, mnist.train.labels.shape)

X_SIZE = 784
Y_SIZE = 10
LR = 0.5
BATCH_SIZE = 300
STEP_TIMES = 30000

x = tf.placeholder(tf.float32, [None, X_SIZE])
y = tf.placeholder(tf.float32, [None, Y_SIZE])

W = tf.Variable(tf.zeros([X_SIZE, Y_SIZE]))
b = tf.Variable(tf.zeros([Y_SIZE]))

# y=softmax(x) Y(i)=exp(Xi)/sum(exp(Xj))
pred = tf.nn.softmax(tf.matmul(x, W) + b)

# y(i,j)*log(pred(i,j))，然后按照i汇总
# 得到N个数值，求平均值
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(LR).minimize(cross_entropy)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1)), tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(STEP_TIMES):
        batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
        _, loss, rr = sess.run([train_step, cross_entropy, accuracy], feed_dict={x: batch_x, y: batch_y})

        if i % 20 == 0:
            print("%d --> %f : %f" % (i, loss, rr))

    print(sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels}))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
