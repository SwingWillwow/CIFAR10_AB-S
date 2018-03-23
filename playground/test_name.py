import numpy as np
import tensorflow as tf


def _get_variable(shape, initializer):
    return tf.get_variable("weight", shape, tf.float32, initializer=initializer)


with tf.Session() as sess:
    with tf.variable_scope('conv1'):
        w = _get_variable([5, 5, 3, 32], tf.constant_initializer(1.0))
    print(w.name)