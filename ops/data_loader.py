import tensorflow as tf
import numpy as np
from scipy import misc
from glob import glob
import matplotlib.pyplot as plt


def get_image_size(config):
    im_size = misc.imread(
      glob(config.train_directory + '*' + config.im_ext)[0]).shape
    if len(im_size) == 2:
        im_size = np.hstack((im_size, 3))
    return im_size


def repeat_elements(x, rep, axis):
    '''Repeats the elements of a tensor along an axis, like np.repeat
    If x has shape (s1, s2, s3) and axis=1, the output
    will have shape (s1, s2 * rep, s3)
    This function is taken from keras backend
    '''
    x_shape = x.get_shape().as_list()
    splits = tf.split(axis, x_shape[axis], x)
    x_rep = [s for s in splits for i in range(rep)]
    return tf.concat(axis, x_rep)


def repeat_reshape_2d(
        image, im_size, num_channels, tf_dtype=tf.float32,
        img_mean_value=None):
    res_image = tf.reshape(image, np.asarray(im_size)[:num_channels])
    image = tf.cast(repeat_elements(tf.expand_dims(
        res_image, 2), 3, axis=2), tf_dtype)
    if img_mean_value is not None:
        image -= img_mean_value
    return image


def clip_to_value(data, low, high, val, tf_dtype=tf.float32):
    hmask = tf.cast(tf.greater(data, high), tf_dtype)
    lmask = tf.cast(tf.less(data, low), tf_dtype)
    bmask = tf.cast(tf.equal(hmask + lmask, False), tf_dtype)
    return data * bmask


def read_and_decode_single_example(
                    filename, im_size, model_input_shape, train,
                    img_mean_value=None, feat_mean_value=None, num_channels=2, img_mean_file=None):
    """first construct a queue containing a list of filenames.
    this lets a user split up there dataset in multiple files to keep
    size down"""
    filename_queue = tf.train.string_input_producer([filename],
                                                    num_epochs=None)
    # Unlike the TFRecordWriter, the TFRecordReader is symbolic
    reader = tf.TFRecordReader()
    # One can read a single serialized example from a filename
    # serialized_example is a Tensor of type string.
    _, serialized_example = reader.read(filename_queue)
    # The serialized example is converted back to actual values.
    # One needs to describe the format of the objects to be returned
    features = tf.parse_single_example(
        serialized_example,
        features={
          'label': tf.FixedLenFeature([], tf.string),
          'image': tf.FixedLenFeature([], tf.string)
                }
        )


    label = tf.decode_raw(features['label'], tf.uint8)
    image = tf.decode_raw(features['image'], tf.uint8)

    raw_im_shape = [480, 640]

    label = tf.reshape(label, np.asarray([112, 112]))
    image = tf.reshape(image, np.asarray([224, 224, 3]))

    return label, tf.cast(image, tf.uint8)

    # Process features specially

    # To support augmentations we have to convert data to 3D
    if num_channels == 2:
        image = repeat_reshape_2d(
            image, im_size, num_channels, img_mean_value=img_mean_value)
    else:
        # Need to reconstruct channels first then transpose channels
        # import pdb; pdb.set_trace()
        res_image = tf.reshape(image, np.asarray(im_size)[[2, 0, 1]])
        if img_mean_value is not None:
            res_image -= img_mean_value
        image = tf.transpose(res_image, [2, 1, 0])

    # Insert augmentation and preprocessing here

    # And finally handle the labels
    label = tf.reshape(label, np.asarray(im_size[0:2]))

    # Set means
    if img_mean_value is None:
        img_mean_value = 0
    elif img_mean_value == 'within':
        img_mean_value = tf.reduce_mean(image)
    return label, image


def read_and_decode(
                    filename_queue, im_size, model_input_shape,
                    train, img_mean_value=[None],
                    num_channels=3, img_mean_file=None):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
          'label': tf.FixedLenFeature([], tf.string),
          'image': tf.FixedLenFeature([], tf.string),
                }
        )

    # Convert from a scalar string tensor (whose single string has
    # import pdb; pdb.set_trace()
    label = tf.decode_raw(features['label'], tf.uint8)
    label_float = tf.decode_raw(features['label'], tf.float64)
    image = tf.decode_raw(features['image'], tf.int8)
    label = tf.reshape(label, np.asarray([112, 112]))
    image = tf.reshape(image, np.asarray([224, 224, 3]))

    orig_image = image

    minl = tf.reduce_min(label)
    label = tf.cast(label, tf.float32) - tf.cast(minl, tf.float32)
    maxl = tf.reduce_max(label)
    label = tf.div(tf.cast(label, tf.float32),tf.cast(maxl, tf.float32))

    # if img_mean_value != None:
    #     # import pdb; pdb.set_trace()
    #     mean = np.load(img_mean_value)['im_list'].astype(int)

    #     image = tf.cast(image, tf.int64) - tf.convert_to_tensor(mean)

    # if train is not None:
        
    #     image = augment_data(image, model_input_shape, im_size, ['up_down'])
    #     label = tf.expand_dims(label, dim=2)
    #     label = augment_data(label, [112, 112], im_size, ['up_down'])
    #     label = tf.squeeze(label)

    return label, tf.cast(image, tf.uint8)


def augment_data(image, model_input_shape, im_size, train):
    if train is not None:
        if 'left_right' in train:
            image = tf.image.flip_left_right(image)
        if 'up_down' in train:
            image = tf.image.flip_up_down(image)
        if 'random_contrast' in train:
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        if 'random_brightness' in train:
            image = tf.image.random_brightness(image, max_delta=32./255.)
        if 'rotate' in train:
            image = tf.image.rot90(image, k=np.random.randint(4))
        if 'random_crop' in train:
            image = tf.random_crop(
                image,
                [model_input_shape[0], model_input_shape[1], im_size[2]])
        else:
            image = tf.image.resize_image_with_crop_or_pad(
                image, model_input_shape[0], model_input_shape[1])
    else:
        image = tf.image.resize_image_with_crop_or_pad(
            image, model_input_shape[0], model_input_shape[1])
    return image


def inputs(
        tfrecord_file, batch_size, im_size, model_input_shape,
        train=None, num_epochs=None, use_features=False,
        img_mean_value=None):

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [tfrecord_file], num_epochs=num_epochs)

        # Even when reading in multiple threads, share the filename
        # queue.
        label, image = read_and_decode(
            filename_queue=filename_queue,
            im_size=im_size,
            model_input_shape=model_input_shape,
            train=None,
            img_mean_value=img_mean_value)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        if use_features:
            input_data = feat
        else:
            input_data = image

        data, labels = tf.train.shuffle_batch(
            [input_data, label], batch_size=batch_size, num_threads=2,
            capacity=1000 + 3 * batch_size, min_after_dequeue=1000)
            # allow_smaller_final_batch=True)

        # Finally, have to reshape label -> 1d matrix
        label_size = [112, 112]
        labels = tf.reshape(labels, [batch_size, np.prod(np.asarray(label_size))])
    return data, labels
