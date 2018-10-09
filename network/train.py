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

def get_image_dataset():

    image = tf.placeholder(tf.string, [None])
    offset_x = tf.placeholder(tf.int32, [None])
    offset_y = tf.placeholder(tf.int32, [None])
    target_x = tf.placeholder(tf.int32, [None])
    target_y = tf.placeholder(tf.int32, [None])

    def map_fn(filename, offset_y, offset_x, target_y, target_x):
        img = tf.read_file(filename)
        img = tf.image.decode_bmp(img)
        img = tf.reshape(img, [240, 240, 3])
        img = tf.image.crop_to_bounding_box(img, offset_y, offset_x, target_y, target_x)
        img = tf.image.resize_images(img, tf.constant([200, 200]))

        return tf.cast(img, tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices((image, offset_y, offset_x, target_y, target_x))
    dataset = dataset.map(map_fn, num_parallel_calls = 4)
    # dataset = dataset.cache()

    return dataset, image, {'offset_x': offset_x, 'offset_y': offset_y, 'target_x': target_x, 'target_y': target_y}

def online_batch(batch_size):

    img_dataset1, img1, img1_crop_config = get_image_dataset()
    img_dataset2, img2, img2_crop_config = get_image_dataset()

    label = tf.placeholder(tf.float32, [None])
    label_dataset = tf.data.Dataset.from_tensor_slices(label)

    dataset = tf.data.Dataset.zip((img_dataset1, img_dataset2, label_dataset))
    dataset = dataset.shuffle(5000).batch(batch_size)

    iterator = dataset.make_initializable_iterator()

    return iterator, img1, img2, label, img1_crop_config, img2_crop_config

def generate_image_pairs(data_size):
    image1 = []
    image2 = []
    label = []

    for i in range(data_size // 2):
        folder = random.randint(0, 25)

        candidate1 = random.randint(0, 50)
        candidate2 = random.randint(0, 50)

        while candidate2 == candidate1:
            candidate2 = random.randint(0, 50)

        image1.append('data5/%02d/%02d.bmp' % (folder, candidate1))
        image2.append('data5/%02d/%02d.bmp' % (folder, candidate2))
        label.append(0)

    for i in range(data_size - (data_size // 2)):
        folder1 = random.randint(0, 25)
        folder2 = random.randint(0, 25)

        while folder2 == folder1:
            folder2 = random.randint(0, 25)

        candidate1 = random.randint(0, 50)
        candidate2 = random.randint(0, 50)

        image1.append('data5/%02d/%02d.bmp' % (folder1, candidate1))
        image2.append('data5/%02d/%02d.bmp' % (folder2, candidate2))
        label.append(1)

    # shuffle image1, image2, label with the same permutation
    pack = list(zip(image1, image2, label))
    random.shuffle(pack)
    image1, image2, label = zip(*pack)

    return np.array(image1), np.array(image2), np.array(label).astype(float)


if __name__ == '__main__':

    save_path = '%s/model.ckpt' % sys.argv[1]

    boxes = np.load('./tmp/bboxes.npy')

    iterator, image1_list, image2_list, label_list, image1_crop_config, image2_crop_config = online_batch(200)
    x1, x2, y = iterator.get_next()

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

    saver = tf.train.Saver(max_to_keep = 40)
    loss_val = None

    for epoch in range(5000):

        img1_list, img2_list, label = generate_image_pairs(10000)
        rand_pick1 = np.floor(np.random.uniform(0, 18, 10000)).astype(int)
        rand_pick2 = np.floor(np.random.uniform(0, 18, 10000)).astype(int)

        sess.run(iterator.initializer, feed_dict = {
            image1_list: img1_list,
            image2_list: img2_list,

            label_list: label,

            image1_crop_config['offset_y']: boxes[rand_pick1][:, 0],
            image1_crop_config['offset_x']: boxes[rand_pick1][:, 1],
            image1_crop_config['target_y']: boxes[rand_pick1][:, 2],
            image1_crop_config['target_x']: boxes[rand_pick1][:, 3],

            image2_crop_config['offset_y']: boxes[rand_pick2][:, 0],
            image2_crop_config['offset_x']: boxes[rand_pick2][:, 1],
            image2_crop_config['target_y']: boxes[rand_pick2][:, 2],
            image2_crop_config['target_x']: boxes[rand_pick2][:, 3],
        })

        while True:
            try:
                loss_val, _ = sess.run([loss, train_step])
            except tf.errors.OutOfRangeError:
                break

        print(epoch + 1, np.mean(loss_val))

        if (epoch + 1) % 100 == 0:
            saver.save(sess, save_path, global_step = epoch + 1)
