#!/usr/bin/env python3
import numpy as np
from pprint import pprint

height = 240
width = 240

shift = 20
crop_size = [200, 166]
offset = [[0, 0], [-15, 0]]

if __name__ == '__main__':

    bboxes = []

    for i, s in enumerate(crop_size):
        center = [height // 2, width // 2]

        for v in [-shift, 0, shift]:
            for h in [-shift, 0, shift]:
                bboxes.append(
                    [center[0] + v - s // 2 + offset[i][0],
                     center[1] + h - s // 2 + offset[i][1],
                     center[0] + v + s // 2 + offset[i][0],
                     center[1] + h + s // 2 + offset[i][1]]
                )

    pprint(bboxes)
    np.save('./bboxes.npy', np.array(bboxes).astype(np.float32))
