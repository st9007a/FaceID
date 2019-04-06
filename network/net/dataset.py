#!/usr/bin/env python3
from glob import glob

import numpy as np
import tensorflow as tf

def get_file_tensor(directory):

    files = []

    for i in range(26):
        files.append(glob('%s/%02d/*.bmp' % (directory, i)))

    image_tensor = tf.constant(files)

    return image_tensor

def get_bbox_tensor():
    bbox = np.load('tmp/bboxes.npy')
    return tf.constant(bbox, dtype=tf.int32)

def get_randint_dataset(start, end, step=1, shuffle=10000):
    return tf.data.Dataset.range(start, end, step).repeat().shuffle(shuffle)

def get_dataset(directory):

    images = get_file_tensor(directory)
    bounding_boxes = get_bbox_tensor()
    label_padding = tf.constant(0)

    def _fix_pick_idx(anchor, anchor_pose, positive_pose, negative_image, negative_pose, anchor_box, positive_box, negative_box):
        negative_image = tf.where(negative_image == anchor,
                                  tf.mod(negative_image + 1, 51), negative_image)

        positvie_pose = tf.where(positive_pose == anchor_pose,
                                 tf.mod(positive_pose + 1, 26), positive_pose)

        return anchor, anchor_pose, positive_pose, negative_image, negative_pose, anchor_box, positive_box, negative_box

    def _read_image(image_id, pose_id, box_id):
        img = tf.read_file(images[image_id, pose_id])
        img = tf.image.decode_bmp(img)
        img = tf.reshape(img, [240, 240, 3])
        img = tf.image.crop_to_bounding_box(img,
                                            bounding_boxes[box_id, 0],
                                            bounding_boxes[box_id, 1],
                                            bounding_boxes[box_id, 2],
                                            bounding_boxes[box_id, 3])
        img = tf.image.resize_images(img, tf.constant([200, 200]))

        return img

    def _pack_images(anchor, anchor_pose, positive_pose, negative_image, negative_pose, anchor_box, positive_box, negative_box):
        return (_read_image(anchor, anchor_pose, anchor_box), \
                _read_image(anchor, positive_pose, positive_box), \
                _read_image(negative_image, negative_pose, negative_box)), \
                label_padding

    anchor = get_randint_dataset(0, 26)
    anchor_pose = get_randint_dataset(0, 51)
    positive_pose = get_randint_dataset(0, 51)
    negative_image = get_randint_dataset(0, 26)
    negative_pose = get_randint_dataset(0, 51)

    anchor_box = get_randint_dataset(0, 18)
    positive_box = get_randint_dataset(0, 18)
    negative_box = get_randint_dataset(0, 18)

    picker = tf.data.Dataset.zip((anchor,
                                  anchor_pose,
                                  positive_pose,
                                  negative_image,
                                  negative_pose,
                                  anchor_box,
                                  positive_box,
                                  negative_box,))

    picker = picker.map(_fix_pick_idx, num_parallel_calls=4)
    picker = picker.map(_pack_images, num_parallel_calls=4)

    return picker.batch(64)

if __name__ == '__main__':

    sess = tf.Session()
    dataset = get_dataset('data5')
    iterator = dataset.make_one_shot_iterator()
    next_el = iterator.get_next()

    for i in range(500):
        print(i)
        sess.run(next_el)
