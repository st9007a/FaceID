# Face ID

After reading [How I implemented iPhone X’s FaceID using Deep Learning in Python.](https://towardsdatascience.com/how-i-implemented-iphone-xs-faceid-using-deep-learning-in-python-d5dbaa128e1d), I try to implement a simple version of face id.

## Installation

`pip3 install -r network/requires.txt`

## Training Pipeline

```shell
$ cd network
$ python3 download_dataset.py
$ python3 crop_image.py data
$ python3 bounding_box.py
$ python3 train.py my_model data
```
