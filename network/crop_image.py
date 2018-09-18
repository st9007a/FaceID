#!/usr/bin/env python3
import sys
import os
import math
import numpy as np

from glob import glob
from PIL import Image

export_dir = sys.argv[1]

scale = 0.5
resize = 240

height = 960
width = 1280

def resize_and_save(file_name, save_name):

    resize_width = math.floor(width * scale)
    resize_height = math.floor(height * scale)

    img = Image.open(file_name)
    img.thumbnail((resize_width, resize_height))
    img = Image.fromarray(np.asarray(img)[:, :, 0:3])

    img = img.crop((
        (resize_width - resize) // 2,
        (resize_height - resize) // 2,
        (resize_width + resize) // 2,
        (resize_height + resize) // 2
    ))

    img.save(save_name, 'bmp')

if __name__ == '__main__':

    file_dirs = glob('face_train/*')

    for i, file_dir in enumerate(file_dirs):

        if not os.path.isdir('%s/%02d/' % (export_dir, i)):
            os.makedirs('%s/%02d/' % (export_dir, i))

        imgs = glob('%s/*.bmp' % file_dir)

        for j, img in enumerate(imgs):
            resize_and_save(img, '%s/%02d/%02d.bmp' % (export_dir, i, j))

        print('done %d' % i)

