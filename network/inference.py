#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from net import squeeze_net

if __name__ == '__main__':

    face_input = tf.placeholder(tf.float32, [None, 200, 200, 3])
    face_data = np.load('test/test.npy').astype(float)

    with tf.variable_scope('squeeze_net'):
        face_output = squeeze_net(face_input)

    sess = tf.Session()
    saver = tf.train.Saver()

    saver.restore(sess, 'models/test/model.ckpt')

    face_id = sess.run(face_output, feed_dict = {face_input: face_data})

    np.save('test/face_id.npy', face_id)
