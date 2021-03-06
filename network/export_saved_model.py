#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import sys

from net import mobile_net_v2

if __name__ == '__main__':

    import_dir = sys.argv[1]
    export_dir = sys.argv[2]
    step = sys.argv[3]

    with tf.variable_scope('mobile_net_v2'):
        face_input = tf.placeholder(tf.float32, [None, 200, 200, 3], name = 'face_input')
        face_output = mobile_net_v2(face_input, training = False)

    sess = tf.Session()
    saver = tf.train.Saver()

    saver.restore(sess, '%s/model.ckpt-%s' % (import_dir, step))

    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

    inputs = { 'face_input': tf.saved_model.utils.build_tensor_info(face_input) }
    outputs = { 'face_output': tf.saved_model.utils.build_tensor_info(face_output) }
    signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs, 'face_id')

    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], signature_def_map = { 'face_id_signature': signature })
    builder.save()

    print(face_input)
    print(face_output)
