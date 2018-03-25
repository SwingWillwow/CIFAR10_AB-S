import tensorflow as tf
import numpy as np


x = [[[0.61, 0.88, 0.91],
     [0.64, 0.06, 0.21],
     [0.91, 0.23, 0.35]],
     [[0.82, 0.38, 0.35],
      [0.99, 0.01, 0.28],
      [0.38, 0.94, 0.36]],
     [[0.07, 0.25, 0.13],
      [0.47, 0.49, 0.16],
      [0.82, 0.28, 0.31]]]
x = tf.constant(x, dtype=tf.float32)
mean = tf.reduce_mean(x, axis=0)
with tf.Session() as sess:
    print(sess.run(x[0][0][0]))
    print(sess.run(x[1][0][0]))
    print(sess.run(x[2][0][0]))
    print(sess.run(mean))
