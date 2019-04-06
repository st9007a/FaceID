#!/usr/bin/env python3
from glob import glob
import json

import tensorflow as tf

from net.mobilenet import MobileNetV2
from net.dataset import get_dataset

def print_num_of_var():
    total_variables = 0
    for var in tf.trainable_variables():

        a = 1

        for p in var.get_shape():
            a *= int(p)

        total_variables += a

    print('Total variables:', total_variables)

def euclidean(a, b):
    return tf.reduce_sum(tf.square(a - b), axis=1)

def model_fn(features, labels, mode, params):

    if mode == tf.estimator.ModeKeys.TRAIN:
        with tf.variable_scope('mobile_net_v2'):
            positive = MobileNetV2(params, True)(features[1])

        with tf.variable_scope('mobile_net_v2', reuse=True):
            negative = MobileNetV2(params, True)(features[2])

        with tf.variable_scope('mobile_net_v2', reuse=True):
            anchor = MobileNetV2(params, True)(features[0])

        positive_distance = euclidean(positive, anchor)
        negative_distance = euclidean(negative, anchor)

        loss = tf.maximum(0., positive_distance - negative_distance + params['alpha'])

        negative_index = tf.argsort(negative_distance, direction='ASCENDING')[:40]

        gather_loss = tf.gather(loss, negative_index)

        optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate=1e-3)
        train_op = optimizer.minimize(gather_loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, loss=gather_loss, train_op=train_op)

    else:
        with tf.variable_scope('mobile_net_v2'):
            face_embedding = MobileNetV2(params, False)(features)

        return tf.estimator.EstimatorSpec(
            tf.estimator.ModeKeys.PREDICT,
            predictions=face_embedding,
            export_outputs={'face_embedding': tf.estimator.export.PredictOutput(face_embedding)})

def serving_input_fn():
    inputs = tf.placeholder(tf.float32, [None, 200, 200, 3])
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

if __name__ == '__main__':

    with open('params/test.json', 'r') as f:
        params = json.load(f)

    config = tf.estimator.RunConfig(save_checkpoints_steps=5000, model_dir='tensorboard/build1/')
    estimator = tf.estimator.Estimator(model_fn=model_fn, params=params, config=config)

    for i in range(1):
        estimator.train(lambda: get_dataset('data5'), steps=10)
        estimator.export_savedmodel(export_dir_base='saved_models/serve1/', serving_input_receiver_fn=serving_input_fn)
