#!/usr/bin/env python3
import sys
import numpy as np
import random
import tensorflow as tf
import random
import math

from glob import glob
from PIL import Image

export_dir = sys.argv[1]

scale = 0.5
resize = 220

height = 960
width = 1280

def get_image_nparray(file_name):

    resize_width = math.floor(width * scale)
    resize_height = math.floor(height * scale)

    img = Image.open(file_name)
    img.thumbnail((resize_width, resize_height))

    img = np.asarray(img)

    return img[(resize_height - resize) // 2: (resize_height + resize) // 2, (resize_width - resize) // 2: (resize_width + resize) // 2, 0:3]

def write_tfrecord(x1, x2, y, writer = None, size = 0, idx = 0):

    if writer == None:
        writer = tf.python_io.TFRecordWriter('%s/%d.tfrecord' % (export_dir, idx))

    elif size == 500:
        print('Done %d' % idx)
        writer.close()
        writer = tf.python_io.TFRecordWriter('%s/%d.tfrecord' % (export_dir, (idx + 1)))
        idx += 1
        size = 0

    x1 = np.reshape(x1, [resize * resize * 3]).tobytes()
    x2 = np.reshape(x2, [resize * resize * 3]).tobytes()

    example = tf.train.Example(features=tf.train.Features(feature = {
        'x1': tf.train.Feature(bytes_list = tf.train.BytesList(value = [x1])),
        'x2': tf.train.Feature(bytes_list = tf.train.BytesList(value = [x2])),
        'y': tf.train.Feature(float_list = tf.train.FloatList(value = [y])),
    }))

    writer.write(example.SerializeToString())
    size += 1

    return writer, size, idx

if __name__ == '__main__':

    file_dirs = glob('face_train/*')

    # build correct
    idx = 0
    size = 0
    writer = None

    for file_dir in file_dirs:

        data = []

        for i in range(1, 18):

            for l in range(1, 4):

                img1 = get_image_nparray('%s/%03d_%d_c.bmp' % (file_dir, i, l))

                for j in range(i + 1, 18):

                    if i == j:
                        continue

                    for k in range(1, 4):

                        img2 = get_image_nparray('%s/%03d_%d_c.bmp' % (file_dir, j, k))
                        writer, size, idx = write_tfrecord(img1, img2, 0, writer, size, idx)

    # build wrong
    couple = {}

    for file_dir in file_dirs:

        couple[file_dir] = set()

        for _ in range(8):

            while True:
                c = random.choice(file_dirs)

                if c != file_dir and c not in couple[file_dir]:

                    if c in couple and file_dir in couple[c]:
                        continue

                    couple[file_dir].add(c)
                    break

    for file_dir in couple:

        for i in range(1, 18):

            img1 = get_image_nparray('%s/%03d_1_c.bmp' % (file_dir, i))

            for c in couple[file_dir]:

                data = []

                for j in range(1, 18):

                    img2 = get_image_nparray('%s/%03d_1_c.bmp' % (c, j))
                    writer, size, idx = write_tfrecord(img1, img2, 1, writer, size, idx)

    writer.close()
