import random
import tensorflow as tf

import numpy as np


print(random.randint(30,300))

s = [float(i) / 100 for i in
     range(32, 53)]
print(s)

x = np.array([1.0, 2, 3, 4])
print(x.shape)

y = np.zeros((2, 3, 4))
print(y)
a = tf.constant(y)
b = tf.unstack(a,axis=0)
c = tf.unstack(a,axis=1)
d = tf.unstack(a,axis=2)

aa = np.ones((2,4))
print(aa)
c_aa = tf.constant(aa)
c_x = tf.constant(x)
mul = tf.multiply(c_x,c_aa)

with tf.Session() as sess:
    # print(sess.run(b))
    # print(sess.run(c))
    # print(sess.run(d))
    print(sess.run(mul))
