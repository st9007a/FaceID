#!/usr/bin/env python3
import numpy as np
from glob import glob
from PIL import Image

def get_image_nparray(file_name):

    img = Image.open(file_name)
    img.thumbnail((640, 480))

    img = np.asarray(img)

    return img[140:340, 220:420, 0:3]

if __name__ == '__main__':

    file_dirs = glob('face_validation/*')

    data = []

    for file_dir in file_dirs:

        for i in range(1, 18):

            img = get_image_nparray('%s/%03d_1_c.bmp' % (file_dir, i))
            data.append(img)

    data = np.array(data)
    np.save('test/test.npy', data)
