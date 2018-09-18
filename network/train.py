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

    def map_fn(filename):
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_bmp(img_string)
        img_resize = tf.reshape(img_decoded, [240, 240, 3])

        return tf.cast(img_resize, tf.float32) / 256

    image = tf.placeholder(tf.string, [None])

    dataset = tf.data.Dataset.from_tensor_slices((image,))
    dataset = dataset.map(map_fn, num_parallel_calls = 4)
    # dataset = dataset.cache()

    return dataset, image

def online_batch(batch_size):

    img_dataset1, img1 = get_image_dataset()
    img_dataset2, img2 = get_image_dataset()

    label = tf.placeholder(tf.float32, [None])
    label_dataset = tf.data.Dataset.from_tensor_slices(label)

    dataset = tf.data.Dataset.zip((img_dataset1, img_dataset2, label_dataset))
    dataset = dataset.shuffle(5000).batch(batch_size)

    iterator = dataset.make_initializable_iterator()

    return iterator, img1, img2, label

# Generate image pairs and bounding box index
#
# @input:
#   data_size(int): the count of image pairs
#
# @output:
#   image1(numpy, shape = (data_size, ), dtype = string): first image path
#   image2(numpy, shape = (data_size, ), dtype = string): second image path
#   label (numpy, shape = (data_size, ), dtype = float): match or mismatch of image paires
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

    # shuffle image1, image2, label with the same permutation
    pack = list(zip(image1, image2, label))
    random.shuffle(pack)
    image1, image2, label = zip(*pack)

    return (
        np.array(image1), np.array(image2), np.array(label).astype(float),
    )


if __name__ == '__main__':

    save_path = '%s/model.ckpt' % sys.argv[1]

    boxes = np.load('bboxes.npy')
    boxes_holder1 = tf.placeholder(tf.float32, [None, 4])
    boxes_holder2 = tf.placeholder(tf.float32, [None, 4])
    boxes_ind = [0]

    while len(boxes_ind) < 200:
        boxes_ind.append(boxes_ind[-1] + 1)

    iterator, image_list1, image_list2, label_list = online_batch(200)
    x1, x2, y = iterator.get_next()

    x1 = tf.image.crop_and_resize(x1, boxes_holder1, boxes_ind, tf.constant((200, 200)))
    x2 = tf.image.crop_and_resize(x2, boxes_holder2, boxes_ind, tf.constant((200, 200)))

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
    loss_val = None

    for epoch in range(300):

        img_list1, img_list2, label = generate_image_pairs(10000)
        sess.run(iterator.initializer, feed_dict = {image_list1: img_list1, image_list2: img_list2, label_list: label})

        while True:
            try:
                loss_val, _ = sess.run([loss, train_step], feed_dict = {
                    boxes_holder1: boxes[np.floor(np.random.uniform(0, 18, 200)).astype(int)],
                    boxes_holder2: boxes[np.floor(np.random.uniform(0, 18, 200)).astype(int)],
                })
            except tf.errors.OutOfRangeError:
                break

        print(epoch, np.mean(loss_val))
        saver.save(sess, save_path)
