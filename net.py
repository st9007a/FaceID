#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

def fire_module(x, name, squeeze, expand):

    with tf.variable_scope(name):
        x = tf.layers.conv2d(x, filters = squeeze, kernel_size = (1, 1), activation = tf.nn.relu, name = 'squeeze')

        expand_1x1 = tf.layers.conv2d(x, filters = expand, kernel_size = (1, 1), activation = tf.nn.relu, name = 'expand_1x1')
        expand_3x3 = tf.layers.conv2d(x, filters = expand, kernel_size = (3, 3), padding = 'same', activation = tf.nn.relu, name = 'expand_3x3')

    return tf.concat([expand_1x1, expand_3x3], axis = 3)

def squeeze_net(x):

    out = tf.layers.conv2d(x, filters = 16, kernel_size = (5, 5), strides = (2, 2), activation = tf.nn.relu, name = 'conv1')

    out = tf.layers.max_pooling2d(out, pool_size = 3, strides = 2)

    out = fire_module(out, squeeze = 32, expand = 32, name = 'fire1')
    out = fire_module(out, squeeze = 32, expand = 32, name = 'fire2')

    out = tf.layers.max_pooling2d(out, pool_size = 3, strides = 2)

    out = fire_module(out, squeeze = 64, expand = 64, name = 'fire3')
    out = fire_module(out, squeeze = 64, expand = 64, name = 'fire4')

    out = tf.layers.max_pooling2d(out, pool_size = 3, strides = 2)

    out = fire_module(out, squeeze = 96, expand = 96, name = 'fire5')
    out = fire_module(out, squeeze = 96, expand = 96, name = 'fire6')

    out = fire_module(out, squeeze = 128, expand = 128, name = 'fire7')
    out = fire_module(out, squeeze = 128, expand = 128, name = 'fire8')

    out = tf.layers.conv2d(out, filters = 512, kernel_size = (1, 1), activation = tf.nn.relu, name = 'conv2')

    out = tf.nn.pool(out, window_shape = (11, 11), strides = (1, 1), pooling_type = 'AVG', padding = 'VALID')

    out = tf.reshape(out, [-1, 512])
    out = tf.layers.dense(out, 256, activation = tf.nn.relu, name = 'dense1')

    out = tf.layers.dense(out, 128, name = 'dense2')

    out = tf.nn.l2_normalize(out, axis = 1)

    return out

