#!/usr/bin/env python3
import sys
import numpy as np
import tensorflow as tf

from net import mobile_net_v2

if __name__ == '__main__':

    model_path = '%s/model.ckpt' % sys.argv[1]

    face_input = tf.placeholder(tf.float32, [None, 200, 200, 3])
    face_data = np.load('test_old/test.npy').astype(float)

    with tf.variable_scope('mobile_net_v2'):
        face_output = mobile_net_v2(face_input, training = False)

    sess = tf.Session()
    saver = tf.train.Saver()

    saver.restore(sess, model_path)

    face_id = sess.run(face_output, feed_dict = {face_input: face_data})

    np.save('test_old/face_id.npy', face_id)
