import cv2
import numpy as np
import tensorflow as tf
from scipy import misc
# from tqdm import tqdm
from scipy.spatial.distance import cdist


def load_raw_im(im):
    """Loads a RAW image."""
    with open(im) as fd:
        img_str = fd.read()
    return img_str


def decode_image(im_str):
    """Decodes a RAW image to uint8."""
    nparr = np.fromstring(im_str, np.uint8)
    return cv2.imdecode(nparr, 0)


def clip_df(df, background_constant):
    adj_constant = background_constant - 100
    df[df > adj_constant] = 0
    df[df < 0] = 0
    return df


def get_depth(
    depth, cte_depth, background_constant,
        resize=None, raw_or_enc='enc'):
    """docstring for get_depth"""
    # dd values in [0,1]
    if raw_or_enc == 'enc':
        dd = misc.imread(depth)[:, :, 0]  # .astype('float32')/255.
    elif raw_or_enc == 'raw':
        dd = decode_image(load_raw_im(depth))

    if resize is not None:
        dd = misc.imresize(dd, resize)

    dd = (dd).astype('float32')
    if np.sum(dd) == 0:
        dd = None
    else:
        dd[dd <= 0] = background_constant
        dd[dd != background_constant] =\
            cte_depth - dd[dd != background_constant]
    return dd
    # return np.repeat(dd[:, :, None], 3, axis=-1)


def create_depth_graph(all_xy, theta, hw, resize):
    """docstring for depth_features"""

    # xy = tf.get_variable('xy', initializer=all_xy)
    # offsets = tf.get_variable('offsets', initializer=theta)
    depth_image = tf.placeholder(
        tf.float32, shape=[hw[0], hw[1]], name='depth_image')
    # total_obs = hw[0] * hw[1]

    # Create difference maps
    print 'Creating feature extraction graph'
    # for i, th in tqdm(enumerate(theta), total=len(theta)):
    for i, th in enumerate(theta):
        im1 = pad_image(depth_image, th[0], hw)
        # im2 = pad_image(depth_image, th[1], hw)
        # sel_slice = tf.expand_dims(im1 - im2, 2)  # image
        sel_slice = tf.expand_dims(depth_image - im1, 2)  # image
        # sel_slice = tf.reshape(depth_image - im1, [total_obs, 1])  # column
        if i == 0:
            dms = sel_slice
        else:
            dms = tf.concat(2, [dms, sel_slice])
    return dms, depth_image


def pad_image(depth_image, offset, hw):
    offset *= -1
    if offset[0] < 0 and offset[1] < 0:
        # +, +
        crop_im = depth_image[:offset[0], :offset[1]]
        pad_im = tf.concat(
            1, [tf.zeros([hw[0],
                np.abs(offset[1])]), tf.concat(
                0, [tf.zeros(
                    [np.abs(offset[0]), hw[1] - np.abs(offset[1])]),
                    crop_im])])  # r_zeros + c_zeros + image
        # image + r_zeros + c_zeros
    elif offset[0] < 0 and offset[1] >= 0:
        # +, -
        crop_im = depth_image[:offset[0], offset[1]:]
        pad_im = tf.concat(
            1, [tf.concat(0,
                [tf.zeros([
                    np.abs(offset[0]), hw[1] - np.abs(offset[1])]), crop_im]),
                tf.zeros([hw[0], offset[1]])])  # r_zeros + image + c_zeros
    elif offset[0] >= 0 and offset[1] < 0:
        # -, +
        crop_im = depth_image[offset[0]:, :offset[1]]
        pad_im = tf.concat(
            1, [tf.zeros([hw[0],
                np.abs(offset[1])]), tf.concat(0,
                [crop_im, tf.zeros(
                    [offset[0], hw[1] - np.abs(
                        offset[1])])])])  # c_zeros + image + r_zeros
    elif offset[0] >= 0 and offset[1] >= 0:
        # -, -
        crop_im = depth_image[offset[0]:, offset[1]:]
        pad_im = tf.concat(
            1, [tf.concat(0,
                [crop_im, tf.zeros(
                    [offset[0], hw[1] - np.abs(offset[1])])]), tf.zeros(
                [hw[0], offset[1]])])  # image + r_zeros + c_zeros
    return pad_im


def random_offsets(offset_max):
    return [np.random.randint(
        -offset_max, offset_max), np.random.randint(-offset_max, offset_max)]


def get_labels(raw_labels, background_constant, label_values):
    """get_labels in a loop"""
    h, w, d = raw_labels.shape
    lab_arr = np.zeros((h, w)).astype('float32')

    # show_colors(raw_labels)
    for v, label in enumerate(label_values.values()):
        ind = ((raw_labels == label).prod(2)).astype('bool')
        lab_arr[ind] = v + 1
    ind = ((raw_labels == 0).prod(2)).astype('bool')
    lab_arr[ind] = background_constant
    return lab_arr


def get_label(raw_labels, background_constant, label_values, resize=None):
    """get_label image"""
    label_im = misc.imread(raw_labels)
    if resize is not None:
        label_im = misc.imresize(label_im, resize)

    bg_im = np.sum(
        label_im, axis=-1).reshape(label_im.shape[0] * label_im.shape[1])

    label_matrix = np.asarray(label_values.values())
    res_im = label_im.reshape(
        label_im.shape[0] * label_im.shape[1], label_im.shape[2])
    dists = np.argmin(cdist(res_im, label_matrix, 'euclidean'), axis=1)
    dists[bg_im == 0] = background_constant
    return dists.reshape(label_im.shape[0], label_im.shape[1])
