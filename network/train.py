#!/usr/bin/env python3
import sys
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

def online_batch(batch_size):

    def read_image(image1, image2, label):

        image1 = tf.images.decode_bmp(image1, channels = 3)
        image2 = tf.images.decode_bmp(image2, channels = 3)

        return image1, image2, label

    image1 = tf.placeholder(tf.string, [None])
    image2  = tf.placeholder(tf.string, [None])
    label  = tf.placeholder(tf.float32, [None])

    dataset = tf.data.Dataset.from_tensor_slices((image1, image2, label))
    dataset = dataset.map(read_image, num_parallel_calls = 8)
    dataset = dataset.shuffle(5000).repeat(1).batch(batch_size)

    iterator = dataset.make_initializable_iterator()

    return iterator, image1, image2, label

# Generate image pairs and bounding box index
#
# @input:
#   data_size(int): the count of image pairs
#
# @output:
#   image1(numpy, shape = (data_size, ), dtype = string): first image path
#   image2(numpy, shape = (data_size, ), dtype = string): second image path
#   label (numpy, shape = (data_size, ), dtype = float): match or mismatch of image paires
#   boxes1(numpy, shape = (data_size, ), dtype = int): bounding box index of first image
#   boxes2(numpy, shape = (data_size, ), dtype = int): bounding box index of second image
def generate_meta_data(data_size):
    image1 = []
    image2 = []
    label = []

    for i in range(data_size // 2):
        folder = random.randint(0, 25)

        candidate1 = random.randint(0, 50)
        candidate2 = random.randint(0, 50)

        while candidate2 == candidate1:
            candidate2 = random.randint(0, 50)

        image1.append('data4/%02d/%02d.bmp' % (folder, candidate1))
        image2.append('data4/%02d/%02d.bmp' % (folder, candidate2))
        label.append(0)

    for i in range(data_size - (data_size // 2)):
        folder1 = random.randint(0, 25)
        folder2 = random.randint(0, 25)

        while folder2 == folder1:
            folder2 = random.randint(0, 25)

        candidate1 = random.randint(0, 50)
        candidate2 = random.randint(0, 50)

        image1.append('data4/%02d/%02d.bmp' % (folder1, candidate1))
        image2.append('data4/%02d/%02d.bmp' % (folder2, candidate2))
        label.append(1)

    return (
        np.array(image1), np.array(image2), np.array(label).astype(float),
        np.floor(np.random.uniform(0, 27, data_size)).astype(int),
        np.floor(np.random.uniform(0, 27, data_size)).astype(int)
    )


if __name__ == '__main__':

    save_path = '%s/model.ckpt' % sys.argv[1]

    x1, x2, y = next_batch(200)
    y = tf.reshape(y, [-1])

    with tf.variable_scope('mobile_net_v2'):
        out1 = mobile_net_v2(x1, training = True)

    with tf.variable_scope('mobile_net_v2', reuse = True):
        out2 = mobile_net_v2(x2, training = True)

    print_num_of_var()

    euclidean_distance = tf.sqrt(tf.reduce_sum(tf.square(out1 - out2), axis = 1))

    loss = (1.0 - y) * tf.square(euclidean_distance) + y * tf.square(tf.maximum(0.0, 1.0 - euclidean_distance))

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    for i in range(1, 15001):

        loss_val, _  = sess.run([loss, train_step])

        if i % 100 == 0:
            print(i, np.mean(loss_val))

            saver.save(sess, save_path)
