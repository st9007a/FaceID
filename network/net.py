#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

def fire_module(x, name, squeeze, expand):

    with tf.variable_scope(name):
        x = tf.layers.conv2d(x, filters = squeeze, kernel_size = (1, 1), activation = tf.nn.relu, name = 'squeeze')

        expand_1x1 = tf.layers.conv2d(x, filters = expand, kernel_size = (1, 1), activation = tf.nn.relu, name = 'expand_1x1')
        expand_3x3 = tf.layers.conv2d(x, filters = expand, kernel_size = (3, 3), padding = 'same', activation = tf.nn.relu, name = 'expand_3x3')

    return tf.concat([expand_1x1, expand_3x3], axis = 3)

def squeeze_net(x, training):

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
    out = tf.layers.dropout(out, rate = 0.2, name = 'dropout1', training = training)

    out = tf.layers.dense(out, 256, activation = tf.nn.relu, name = 'dense1')
    out = tf.layers.dense(out, 128, name = 'dense2')

    out = tf.nn.l2_normalize(out, axis = 1, name = 'face_output')

    return out


def inverted_residual(x, name, expand_factor, output_channels, subsample = False):

    expand_channels = expand_factor * int(x.get_shape()[3])
    strides = 2 if subsample else 1

    with tf.variable_scope(name):

        out = tf.layers.conv2d(
            x,
            filters =  expand_channels,
            kernel_size = 1,
            activation = tf.nn.relu6,
            name = 'conv_1x1',
        )

        out = tf.layers.batch_normalization(out, training = True)

        out = tf.layers.separable_conv2d(
            out,
            filters = expand_channels,
            kernel_size = 3,
            activation = tf.nn.relu6,
            padding = 'same',
            strides = strides,
            name = 'depth_conv_3x3'
        )

        out = tf.layers.batch_normalization(out, training = True)

        out = tf.layers.conv2d(
            out,
            filters = output_channels,
            kernel_size = 1,
            name = 'linear_conv_1x1'
        )

        out = tf.layers.batch_normalization(out, training = True)

        if int(x.get_shape()[3]) == int(out.get_shape()[3]):
            out = out + x

    return out

def mobile_net_v2(x, training):

    out = tf.layers.conv2d(x, filters = 32, kernel_size = 3, strides = 2, padding = 'same', activation = tf.nn.relu, name = 'conv1')
    out = tf.layers.batch_normalization(out, training = True)

    out = inverted_residual(out, name = 'bottleneck1', expand_factor = 1, output_channels = 16)

    out = inverted_residual(out, name = 'bottleneck2', expand_factor = 6, output_channels = 24, subsample = True)
    out = inverted_residual(out, name = 'bottleneck3', expand_factor = 6, output_channels = 24)

    out = inverted_residual(out, name = 'bottleneck4', expand_factor = 6, output_channels = 32, subsample = True)
    out = inverted_residual(out, name = 'bottleneck5', expand_factor = 6, output_channels = 32)
    out = inverted_residual(out, name = 'bottleneck6', expand_factor = 6, output_channels = 32)

    out = inverted_residual(out, name = 'bottleneck7', expand_factor = 6, output_channels = 64, subsample = True)
    out = inverted_residual(out, name = 'bottleneck8', expand_factor = 6, output_channels = 64)
    out = inverted_residual(out, name = 'bottleneck9', expand_factor = 6, output_channels = 64)
    out = inverted_residual(out, name = 'bottleneck10', expand_factor = 6, output_channels = 64)

    out = inverted_residual(out, name = 'bottleneck11', expand_factor = 6, output_channels = 96)
    out = inverted_residual(out, name = 'bottleneck12', expand_factor = 6, output_channels = 96)
    out = inverted_residual(out, name = 'bottleneck13', expand_factor = 6, output_channels = 96)

    out = inverted_residual(out, name = 'bottleneck14', expand_factor = 6, output_channels = 160, subsample = True)
    out = inverted_residual(out, name = 'bottleneck15', expand_factor = 6, output_channels = 160)
    out = inverted_residual(out, name = 'bottleneck16', expand_factor = 6, output_channels = 160)

    out = inverted_residual(out, name = 'bottleneck17', expand_factor = 6, output_channels = 320)

    out = tf.layers.conv2d(out, filters = 1280, kernel_size = 1, activation = tf.nn.relu, name = 'conv2')
    out = tf.layers.batch_normalization(out, training = True)

    out = tf.nn.pool(out, window_shape = (4, 4), strides = (1, 1), pooling_type = 'AVG', padding = 'VALID')

    out = tf.reshape(out, [-1, 1280])
    out = tf.layers.dense(out, 128, name = 'dense1')

    out = tf.nn.l2_normalize(out, axis = 1, name = 'face_output')
    return out
