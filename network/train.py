#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import random
from glob import glob
from pprint import pprint

from net import mobile_net_v2

def print_num_of_var():
    total_variables = 0
    for var in tf.trainable_variables():

        a = 1

        for p in var.get_shape():
            a *= int(p)

        total_variables += a

    print('Total variables:', total_variables)

def next_batch(batch_size):

    def parse_fn(example_proto):

        features = {
            'x1': tf.FixedLenFeature((), tf.string),
            'x2': tf.FixedLenFeature((), tf.string),
            'y': tf.FixedLenFeature((1, ), tf.float32),
        }

        parsed_features = tf.parse_single_example(example_proto, features)

        x1 = tf.decode_raw(parsed_features['x1'], out_type = tf.uint8)
        x2 = tf.decode_raw(parsed_features['x2'], out_type = tf.uint8)

        x1 = tf.cast(tf.reshape(x1, [200, 200, 3]), tf.float32)
        x2 = tf.cast(tf.reshape(x2, [200, 200, 3]), tf.float32)

        return x1, x2, parsed_features['y']

    filenames = glob('data3/*')
    random.shuffle(filenames)

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_fn, num_parallel_calls = 8)
    dataset = dataset.shuffle(5000).repeat().batch(batch_size)

    return dataset.make_one_shot_iterator().get_next()

if __name__ == '__main__':

    x1, x2, y = next_batch(128)
    y = tf.reshape(y, [-1])

    with tf.variable_scope('squeeze_net'):
        out1 = mobile_net_v2(x1, training = True)

    with tf.variable_scope('squeeze_net', reuse = True):
        out2 = mobile_net_v2(x2, training = True)

    print_num_of_var()

    euclidean_distance = tf.sqrt(tf.reduce_sum(tf.square(out1 - out2), axis = 1))

    loss = (1.0 - y) * tf.square(euclidean_distance) + y * tf.square(tf.maximum(0.0, 1.0 - euclidean_distance))

    global_step = tf.Variable(0, trainable = False)
    learning_rate = tf.train.exponential_decay(0.05, global_step, 200, 0.9, staircase = True)
    train_step = tf.train.RMSPropOptimizer(learning_rate, momentum = 0.9).minimize(loss, global_step = global_step)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    for i in range(1, 10001):

        loss_val, _  = sess.run([loss, train_step])

        if i % 100 == 0:
            print(i, np.mean(loss_val))

            saver.save(sess, 'models/alpha-2.0/model.ckpt')
