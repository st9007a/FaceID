#!/usr/bin/env python3
import numpy as np
from PIL import Image

arr = np.load('x1.npy')
print(arr.shape)
print(arr.dtype)

i = 0
for img in arr:

    res = Image.fromarray(img.astype(np.uint8))
    res.save('img/%d.bmp' % i)
    i += 1
