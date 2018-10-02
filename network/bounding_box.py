#!/usr/bin/env python3
import os
from pprint import pprint

import numpy as np

height = 240
width = 240

shift = 20
crop_size = [200, 166]
offset = [[0, 0], [-15, 0]]

if __name__ == '__main__':

    if not os.path.isdir('tmp'):
        os.makedirs('tmp')

    bboxes = []

    for i, s in enumerate(crop_size):
        center = [height // 2, width // 2]

        for v in [-shift, 0, shift]:
            for h in [-shift, 0, shift]:
                bboxes.append([
                    center[0] + v - s // 2 + offset[i][0],
                    center[1] + h - s // 2 + offset[i][1],
                    s,
                    s,
                    # center[0] + v + s // 2 + offset[i][0],
                    # center[1] + h + s // 2 + offset[i][1]
                ])

    pprint(bboxes)
    print('Generate bounding box and save to "./tmp/bboxes.npy"')
    np.save('./tmp/bboxes.npy', np.array(bboxes).astype(np.float32))
