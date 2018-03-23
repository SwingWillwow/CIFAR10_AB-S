import numpy as np
import tensorflow as tf


def _get_variable1(shape, initializer):
    return tf.get_variable("weight", shape, tf.float32, initializer=initializer)


def _get_variable2(shape, initializer, scope=None):
    return tf.get_variable(scope.name, shape, tf.float32, initializer=initializer)


with tf.Session() as sess:
    with tf.variable_scope('conv1') as scope:
        w = _get_variable1([5, 5, 3, 32], tf.constant_initializer(1.0))
        print(w.name)
        w2 = _get_variable2([1, 2, 3, 4], tf.constant_initializer(1.0), scope)
        print(w2.name)
