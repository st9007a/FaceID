# Face ID

After reading [How I implemented iPhone X’s FaceID using Deep Learning in Python.](https://towardsdatascience.com/how-i-implemented-iphone-xs-faceid-using-deep-learning-in-python-d5dbaa128e1d),
I try to implement a simple version of face id. I used MobileNet-v2 as basic model architecture.
It's so small that it can run on the browsers which support `tensorflow.js` and WebRTC API.

## Demo

[https://merry.ee.ncku.edu.tw/~st9007a/face/](https://merry.ee.ncku.edu.tw/~st9007a/face/)

## Installation

```shell
$ cd network
$ pip3 install -r requires.txt
$ cd ../web
$ yarn
```

## Training Pipeline

```shell
$ cd network
$ python3 download_dataset.py
$ python3 crop_image.py data
$ python3 bounding_box.py
$ python3 train.py my_model data
```

## Deploy

1. Export tensorflow saved model.

```shell
$ cd network
$ python3 export_saved_model.py [input model folder] [output model folder] [training step]
```

2. Use tensorflow-js-converter to export web model.

```shell
$ tensorflowjs_converter \
  --input_format=tf_saved_model \
  --signature_name=face_id_signature \
  --saved_model_tags=serve \
  [saved model folder] \
  [web model folder]
```

3. Go to `web/` folder and build web page.
```shell
$ cd ../web
$ yarn build
```

4. Make a soft link of your web model
```shell
$ cd dist
$ ln -s [your web model] model
```

5. Now, the folder `dist` is a simple web demo which displays a face id system.
