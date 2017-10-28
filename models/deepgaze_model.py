import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from tensorflow.python.ops import clip_ops
import os
import sys


VGG_MEAN = [103.939, 116.779, 123.68]


def gauss_filter(size, sigma):
    # make sure size is odd
    x = int(size)
    if x % 2 == 0:
        x = x + 1
    x_zeros = np.zeros((size, size))

    center = int(np.floor(x / 2.))

    x_zeros[center, center] = 1
    y = gaussian_filter(x_zeros, sigma=sigma)
    min_y = np.min(y)
    new_y = y - min_y
    max_y = np.max(new_y)
    y = new_y / (max_y * 1.0)
    y = np.expand_dims(y, axis=2)
    y = np.expand_dims(y, axis=3)

    return y


def get_cb_matrix(size, sigma):
    center_zeros = np.zeros((size, size))
    center_pixel = size / 2
    center_zeros[center_pixel, center_pixel] = 1
    y = gaussian_filter(center_zeros, sigma=sigma)

    y = np.expand_dims(y, axis=2)

    yy = np.expand_dims(y, axis=0)

    yyy = yy
    for ii in range(0, 15):
        yyy = np.concatenate((yyy, yy), axis=0)

    return yyy


def expand_cb_filter(cb_filter, batch_size):
    cb_filter_3d = np.expand_dims(cb_filter, axis=2)
    cb_filter_4d = np.expand_dims(cb_filter, axis=0)
    expanded_filter = cb_filter_4d

    for ii in range(0, batch_size - 1):
        expanded_filter = np.concatenate(
            (expanded_filter, cb_filter_4d), axis=0)

    expanded_filter = np.expand_dims(expanded_filter, axis=3)
    return expanded_filter


def get_hard_center_bias():
    whole = np.ones([112, 112]) * 0.01
    inside = np.ones([90, 90])
    whole[11:101, 11:101] = inside
    return whole

def normalize_tensor(tensor):
    tensor_norm = tf.div(
        tf.subtract(
            tensor,
            tf.reduce_min(tensor)
            ),
        tf.subtract(
            tf.reduce_max(tensor),
            tf.reduce_min(tensor)
            ) + 1e-6
        )
    return tensor_norm


class model_struct:
    """
    A trainable version VGG16.
    """

    def __init__(
            self, vgg16_npy_path=None, trainable=True,
            fine_tune_layers=None):
        if vgg16_npy_path is not None:
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
            # pop the specified keys from the weights that will be loaded
            if fine_tune_layers is not None:
                for key in fine_tune_layers:
                    del self.data_dict[key]
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable

    def build(
            self, rgb, output_categories=None,
            train_mode=None, batchnorm=None,
            resize_layer='conv2_1', trainable_layers=None, config=None, shuffled=False):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder:
        :if True, dropout will be turned on
        """
        self.trainable_layers = trainable_layers
        self.rgb_orig = rgb

        if output_categories is None:
            output_categories = 2  # len(config.labels)

        self.rgb = tf.cast(rgb, tf.float32)  # Scale up to imagenet's uint8

        # Convert RGB to BGR
        blue, green, red = tf.split(self.rgb, 3, 3)

        self.vgg_mean = VGG_MEAN

        bgr = tf.concat([
            blue, # - VGG_MEAN[0],
            green, # - VGG_MEAN[1],
            red, # - VGG_MEAN[2],
        ], 3)

        self.bgr = bgr

        assert self.rgb.get_shape().as_list()[1:] == [224, 224, 3]

        # VGG19 starts here
        self.relu1_1, self.conv1_1 = self.conv_layer(
            self.rgb, 3, 64, "conv1_1")
        self.relu1_2, self.conv1_2 = self.conv_layer(
            self.relu1_1, 64, 64, "conv1_2")
        self.pool1 = self.max_pool(self.relu1_2, 'pool1')

        self.relu2_1, self.conv2_1 = self.conv_layer(
            self.pool1, 64, 128, "conv2_1")
        self.relu2_2,  self.conv2_2 = self.conv_layer(
            self.relu2_1, 128, 128, "conv2_2")
        self.pool2 = self.max_pool(self.relu2_2, 'pool2')

        self.relu3_1, self.conv3_1 = self.conv_layer(
            self.pool2, 128, 256, "conv3_1")
        self.relu3_2, self.conv3_2 = self.conv_layer(
            self.relu3_1, 256, 256, "conv3_2")
        self.relu3_3, self.conv3_3 = self.conv_layer(
            self.relu3_2, 256, 256, "conv3_3")
        self.relu3_4, self.conv3_4 = self.conv_layer(
            self.relu3_3, 256, 256, "conv3_4")
        self.pool3 = self.max_pool(self.relu3_4, 'pool3')

        self.relu4_1, self.conv4_1 = self.conv_layer(
            self.pool3, 256, 512, "conv4_1")
        self.relu4_2, self.conv4_2 = self.conv_layer(
            self.relu4_1, 512, 512, "conv4_2")
        self.relu4_3, self.conv4_3 = self.conv_layer(
            self.relu4_2, 512, 512, "conv4_3")
        self.relu4_4, self.conv4_4 = self.conv_layer(
            self.relu4_3, 512, 512, "conv4_4")
        self.pool4 = self.max_pool(self.relu4_4, 'pool4')

        self.relu5_1, self.conv5_1 = self.conv_layer(
            self.pool4, 512, 512, "conv5_1", batchnorm)
        self.relu5_2, self.conv5_2 = self.conv_layer(
            self.relu5_1, 512, 512, "conv5_2", batchnorm)
        self.relu5_3, self.conv5_3 = self.conv_layer(
            self.relu5_2, 512, 512, "conv5_3", batchnorm)
        self.relu5_4, self.conv5_4 = self.conv_layer(
            self.relu5_3, 512, 512, "conv5_4", batchnorm)

        self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        resize_size = [int(x) for i, x in enumerate(
            self.conv2_1.get_shape()) if i != 0]
        new_size = np.asarray([resize_size[0], resize_size[1]])

        # these are the vgg layers we're pulling features from -- this can be
        # # played around with
        l1 = tf.image.resize_images(
            self.conv5_1, new_size)

        l2 = tf.image.resize_images(
            self.relu5_1, new_size)

        l3 = tf.image.resize_images(
            self.relu5_2, new_size)

        l4 = tf.image.resize_images(
            self.conv5_3, new_size)

        l5 = tf.image.resize_images(
            self.relu5_4, new_size)

        self.feature_encoder = tf.concat(
            [l1, l2, l3, l4, l5], 3)  # channelwise

        # Add 1x1 convs to vgg features
        # Conv 1 - 1x1/relu/dropout/batchnorm
        self.fc_relu1, self.fc_conv1 = self.conv_layer(
            self.feature_encoder,
            int(self.feature_encoder.get_shape()[-1]), 16,
            "fc_conv1", filter_size=1, batchnorm=batchnorm)

        if train_mode is not None:
            # Add dropout during training
            self.fc_relu1 = tf.cond(
                train_mode,
                lambda: tf.nn.dropout(
                    self.fc_relu1, 0.5), lambda: self.fc_relu1)

        # Conv 2 - 1x1/relu/dropout/batchnorm
        self.fc_relu2, self.fc_conv2 = self.conv_layer(
            self.fc_relu1, int(self.fc_relu1.get_shape()[-1]), 32,
            "fc_conv2", filter_size=1, batchnorm=batchnorm)

        if train_mode is not None:
            # Add dropout during training
            self.fc_relu2 = tf.cond(
                train_mode,
                lambda: tf.nn.dropout(
                    self.fc_relu2, 0.5), lambda: self.fc_relu2)

        # Conv 3 - 1x1/relu/dropout/batchnorm
        self.fc_relu3, self.fc_conv3 = self.conv_layer(
            self.fc_relu2, int(
                self.fc_relu2.get_shape()[-1]), 4,
            "fc_conv3", filter_size=1, batchnorm=batchnorm)

        if train_mode is not None:
            # Add dropout during training
            self.fc_relu3 = tf.cond(
                train_mode,
                lambda: tf.nn.dropout(
                    self.fc_relu3, 0.5), lambda: self.fc_relu3)

        # image-sized output
        self.logits_relu, self.logits = self.conv_layer(
            self.fc_relu3, int(
                self.fc_relu3.get_shape()[-1]), 1,
            "fc_conv4", filter_size=1, batchnorm=batchnorm)

        self.leaky_relu_logits = lrelu(self.logits)

        gauss_blur_filter = gauss_filter(21, 7)
        gb_filter_max = np.max(gauss_blur_filter)
        self.gauss_blur_filter = gauss_blur_filter / gb_filter_max
        # self.gauss_blur_filter = tf.Variable(initial_value=tf.cast(self.gauss_blur_filter, tf.float32), trainable=True, name='gauss_blur_filter')

        # self.logits = normalize_tensor(self.logits)

        # apply gaussian blur
        self.center_bias_logits = tf.nn.conv2d(tf.cast(self.logits, tf.float32),
                                              self.gauss_blur_filter, [1, 1, 1, 1], padding='SAME')

        center_bias_filter = np.load(config.mean_fixmap)
        cb_max = np.max(center_bias_filter)
        center_bias_filter = center_bias_filter / cb_max
        center_bias_filter = expand_cb_filter(center_bias_filter, config.train_batch)
        self.center_bias_logits = tf.multiply(self.center_bias_logits, center_bias_filter)

        # center_bias_mat = np.ones([config.label_size, config.label_size])
        # center_bias_mat = tf.expand_dims(tf.expand_dims(tf.cast(center_bias_mat, tf.float32), 2), 0)
        # self.center_bias_mat = tf.Variable(initial_value=center_bias_mat, trainable=True, name='center_bias_mat')

        # self.center_bias_logits = tf.multiply(self.center_bias_logits, self.center_bias_mat)

        # self.prediction = self.center_bias_logits
        self.prediction_presoft = tf.reshape(normalize_tensor(self.center_bias_logits), [config.train_batch, config.label_size * config.label_size, 1])
        self.prediction = normalize_tensor(tf.reshape(tf.nn.softmax(self.prediction_presoft, dim=1), [config.train_batch, config.label_size, config.label_size, 1]))
        # self.prediction = normalize_tensor(tf.reshape(self.prediction_presoft, [config.train_batch, config.label_size, config.label_size, 1]))

        # Finishing touches
        self.data_dict = None

    def batchsoftmax(self, layer, name=None, axis=3):
        exp_layer = tf.exp(layer)
        exp_sums = tf.expand_dims(
            tf.reduce_sum(exp_layer, reduction_indices=[axis]), dim=axis)
        return tf.div(exp_layer, exp_sums, name=name)

    def batchnorm(self, layer):
        m, v = tf.nn.moments(layer, [0])
        return tf.nn.batch_normalization(layer, m, v, None, None, 1e-3)

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(
            bottom, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(
            bottom, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(
            self, bottom, in_channels,
            out_channels, name, batchnorm=None, filter_size=3):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(
                filter_size, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            # if batchnorm is not None:
            #     if name in batchnorm:
            #         relu = self.batchnorm(relu)

            return relu, bias

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal(
            [filter_size, filter_size, in_channels, out_channels], 0.0, .001)  # 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")
        initial_value = tf.truncated_normal(
            [out_channels], 0.0, .001)  # .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            # get_variable, change the boolean to numpy

            if self.trainable_layers is not None: # and name in self.trainable_layers:
                var = tf.get_variable(
                    name=var_name, initializer=value, trainable=True)
                print name, 'IS TRAINABLE'
            else:
                var = tf.get_variable(
                    name=var_name, initializer=value, trainable=False)
            # var = tf.get_variable(name=var_name, initializer=value, trainable=True)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        assert var.get_shape() == initial_value.get_shape()

        return var

    def get_center_bias_filter(self, name, size):
        initial_value = tf.truncated_normal(
            [size, size, 1, 1], 0.0, .001)
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.get_variable(
                name=var_name, initializer=value, trainable=True)
        else:
            var = tf.get_variable(
                name=var_name, initializer=value, trainable=False)

        self.var_dict[(name, idx)] = var
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./vgg16-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in self.var_dict.items():
            var_out = sess.run(var)
            if name not in data_dict.keys():
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print("file saved", npy_path)
        return npy_path

    def get_var_count(self):
        count = 0
        for v in self.var_dict.values():
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)
