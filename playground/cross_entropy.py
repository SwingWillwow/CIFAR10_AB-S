import tensorflow as tf
import numpy as np

# logit =[[1, 1, 2],
#         [1, 2, 1],
#         [1, 1, 1],
#         [1, 1, 1]]
softmax =[[0, 0.4, 0.6],
          [0, 1, 0],
          [0, 0.25, 0.75],
          [0, 0.3, 0.7]]
labels = [[0, 0, 1],
          [0, 1, 0],
          [0, 0, 1],
          [0, 0, 1]]
# softmax = tf.nn.softmax(logits=tf.cast(logit, tf.float32), axis=1)
softmax = tf.constant(softmax)
basic_cost = -(labels*tf.log(softmax+1e-8))
sum_cost = tf.reduce_sum(basic_cost, axis=1)
mean = tf.reduce_mean(sum_cost)
with tf.Session() as sess:
    print(sess.run(basic_cost))
    print(sess.run(sum_cost))
    print(sess.run(mean))
