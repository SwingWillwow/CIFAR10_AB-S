import tensorflow as tf
import numpy as np


a = tf.Variable(1, dtype=tf.int32)
b = tf.Variable(2.0)


def train():
    clean_op = tf.cond(tf.equal(tf.mod(a, tf.constant(2))
                                , tf.constant(0)),
                       lambda: tf.assign_add(b, tf.constant(1.0),
                                         name='clean_op').op,
                       lambda: tf.no_op('clean_op'))
    with tf.control_dependencies([clean_op]):
        add_assignment = tf.assign_add(a, tf.constant(1, dtype=tf.int32))
        add_op = add_assignment.op
    with tf.control_dependencies([add_op]):
        train_op = tf.no_op('train')
    return train_op


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    train_op = train()
    for i in range(4):
        sess.run(train_op)
        print(sess.run(a))
        print(sess.run(b))
