import tensorflow as tf


def test1():
    print(tf.get_variable_scope().name)
    tf.get_variable('a', [1], dtype=tf.float64)


with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
    test1()
    print("*" * 20)
with tf.variable_scope("foo1"):
    v1 = tf.get_variable("v", [1])
    test1()
assert v1 == v
