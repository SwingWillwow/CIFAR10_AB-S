import tensorflow as tf
import numpy as np
logit =[[1, 1, 2],
        [1, 2, 1],
        [1, 1, 1],
        [1, 1, 1]]
labels = [2, 1, 2, 2]
# x = [0, 0, 0]
predict = tf.equal(tf.argmax(logit, axis=1), labels)
y = tf.reduce_sum(tf.cast(predict, tf.int32))
with tf.Session() as sess:
    print(sess.run(y))
    # predict = tf.nn.in_top_k(logit, labels, 1)
    # sum = np.sum(sess.run(predict))
    # print(sum)
