#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

class InvertedResidual():

    def __init__(self, params, train):
        self.params = params
        self.train = train

    def __call__(self, x, name):

        expand_channels = self.params['expand_factor'] * int(x.get_shape()[3])
        strides = 2 if self.params['subsample'] else 1

        with tf.variable_scope(name, initializer=tf.glorot_normal_initializer()):

            depthwise_weight = tf.get_variable(name='depthwise_conv_weight', shape=(3, 3, expand_channels, 1))

            out = tf.layers.conv2d(x, filters=expand_channels, kernel_size=1, name='conv_1x1')
            out = tf.layers.batch_normalization(out, training=self.train, name='batch_norm1')
            out = tf.nn.relu6(out)

            out = tf.nn.depthwise_conv2d(out, filter=depthwise_weight, padding='SAME', strides=(1, strides, strides, 1), name='depth_conv_3x3')
            out = tf.layers.batch_normalization(out, training=self.train, name='batch_norm2')
            out = tf.nn.relu6(out)

            out = tf.layers.conv2d(x, filters=self.params['output_channels'], kernel_size=1, name='linear_conv_1x1')
            out = tf.layers.batch_normalization(out, training=self.train, name='batch_norm3')

        if int(x.get_shape()[3]) == int(out.get_shape()[3]) and not self.params['subsample']:
            out = out + x

        return out

class MobileNetV2():

    def __init__(self, params, train):
        self.params = params
        self.train = train

        self.bottleneck = []

        for params in self.params['bottleneck']:
            self.bottleneck.append(InvertedResidual(params, train))

    def __call__(self, x):

        out = tf.layers.conv2d(x, filters=16, kernel_size=5, strides=2, padding='same', name='conv1')
        out = tf.layers.batch_normalization(out, training=self.train, name='bn1')
        out = tf.nn.relu6(out)

        for i, layer in enumerate(self.bottleneck):
            out = layer(out, name='bottleneck%d' % i)

        out = tf.layers.conv2d(out, filters=1024, kernel_size=1, name='conv2')
        out = tf.layers.batch_normalization(out, training=self.train, name='bn2')
        out = tf.nn.relu6(out)

        out = tf.keras.layers.GlobalAveragePooling2D()(out)
        out = tf.layers.dense(out, self.params['embedding_dim'], name='dense1')

        out = tf.nn.l2_normalize(out, axis=1, name='face_output')

        return out

