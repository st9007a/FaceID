#!/usr/bin/env python3
import numpy as np
from PIL import Image

arr = np.load('test/test.npy')

i = 0
for img in arr:

    res = Image.fromarray(img)
    res.save('img/%d.bmp' % i)
    i += 1
