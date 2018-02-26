# coding=utf-8
'''
accuracy:98%
'''

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print(mnist.train.images.shape, mnist.train.labels.shape)

def weight_variable(shape):
    initial= tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial= tf.constant(0.1,shape=shape)
    return tf.Variable(initial)


def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

X_SIZE = 784
Y_SIZE = 10
LR = 1e-4
BATCH_SIZE = 100
STEP_TIMES = 8000
UNIT_NUM=300
KEEP_PROB = 0.75

x = tf.placeholder(tf.float32, [None, X_SIZE])
y = tf.placeholder(tf.float32, [None, Y_SIZE])
x_img = tf.reshape(x,[-1,28,28,1])
keep_prob = tf.placeholder(tf.float32)

W1 = weight_variable([5,5,1,32])
b1 = bias_variable([32])
h1 = tf.nn.relu(conv2d(x_img,W1)+b1)
hp = max_pool_2x2(h1)

W2 = weight_variable([5,5,32,64])
b2 = bias_variable([64])
h2 = tf.nn.relu(conv2d(hp,W2)+b2)
hp2 = max_pool_2x2(h2)

W3 = weight_variable([7*7*64,1024])
b3 = bias_variable([1024])
f3 = tf.reshape(hp2,[-1,7*7*64])
fc3 = tf.nn.relu(tf.matmul(f3,W3)+b3)
fc3_drop = tf.nn.dropout(fc3,keep_prob)

W4 = weight_variable([1024,10])
b4 = bias_variable([10])
pred = tf.nn.softmax(tf.matmul(fc3_drop,W4)+b4)

#损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=[1]))

#训练模型
train_step = tf.train.AdamOptimizer(LR).minimize(cross_entropy)

#准确率计算
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1)), tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(STEP_TIMES):
        batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
        _, loss, rr = sess.run([train_step, cross_entropy, accuracy], feed_dict={x: batch_x, y: batch_y,keep_prob:KEEP_PROB})

        if i % 20 == 0:
            print("%d --> %f : %f" % (i, loss, rr))

    # print(sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels,keep_prob:1.0}))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels,keep_prob:1.0}))
