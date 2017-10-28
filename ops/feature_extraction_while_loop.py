import numpy as np
import tensorflow as tf
from scipy import misc
from tqdm import tqdm
from scipy.spatial.distance import cdist


def get_depth(depth, cte_depth, background_constant, resize=None, raw_or_enc='enc'):
    """docstring for get_depth"""
    # dd values in [0,1]
    if raw_or_enc == 'enc':
        dd = misc.imread(depth)[:, :, 0].astype('float32')/255.
    elif raw_or_enc == 'raw':
        dd = decode_image(load_raw_im(depth))

    if resize is not None:
        dd = misc.imresize(dd, resize)

    if np.sum(dd) == 0:
        dd = None
    else:
        dd[dd == 0] = background_constant
        dd[dd != background_constant] =\
            cte_depth - dd[dd != background_constant]
    return dd
    # return np.repeat(dd[:, :, None], 3, axis=-1)


def create_depth_graph(all_xy, theta, hw, resize):
    """docstring for depth_features"""

    # Declare tf variables
    xy = tf.get_variable('xy', initializer=all_xy)
    thetas = tf.get_variable('thetas', initializer=theta.astype(np.int32))
    depth_image = tf.placeholder(
        tf.float32, shape=[hw[0], hw[1]], name='depth_image')

    # Create difference maps
    print 'Creating feature extraction graph'
    dms = []
    i0 = tf.constant(0)
    theta_shape = tf.constant(theta.shape)
    total_obs = hw[0] * hw[1]

    # Prepare while loop
    elems = [
        i0, theta_shape, total_obs, hw, depth_image, thetas, dms 
    ]
   
    returned = tf.while_loop(
        condition, body, loop_vars=elems, back_prop=False, swap_memory=False)

    dms = returned[-1]

    # for i, th in tqdm(enumerate(theta), total=len(theta)):
    #     im1 = pad_image(depth_image, th[0], hw)
    #     im2 = pad_image(depth_image, th[1], hw)
    #     diff_feat = im1 - im2
    #     sel_slice = tf.reshape(diff_feat, [total_obs, 1])  # ravel into a feature column
    #     if i == 0:
    #         dms = sel_slice
    #     else:
    #         dms = tf.concat(1, [dms, sel_slice])
    return dms, depth_image


def condition(i0, theta_shape, total_obs, hw, depth_image, thetas, dms):
    return i0 < theta_shape[0]


def body(i0, theta_shape, total_obs, hw, depth_image, thetas, dms):
    th = tf.slice(
        thetas, [i0, 2, 2],
        [2, 2, 1])
    im1 = pad_image(depth_image, th[0], hw)
    im2 = pad_image(depth_image, th[1], hw)
    diff_feat = im1 - im2
    dms.append(diff_feat)
    i0 += 1
    return i0, num_theta, total_obs, hw, depth_image, offsets, dms


def pad_image(depth_image, offset, hw):
    offset *= -1
    zero_lt = tf.less(offset[0], tf.constant(0))
    zero_gt = tf.greater_equal(offset[0], tf.constant(0))
    one_lt = tf.less(offset[1], tf.constant(0))
    one_gt = tf.greater_equal(offset[1], tf.constant(0))

    if zero_lt is not None and one_lt is not None:
        # +, +
        crop_im = depth_image[:offset[0], :offset[1]]
        pad_im = tf.concat(1,
            [tf.zeros([hw[0],
                np.abs(offset[1])]), tf.concat(0,
                [tf.zeros(
                    [np.abs(offset[0]), hw[1] - np.abs(offset[1])]), crop_im])])  # r_zeros + c_zeros + image
        # image + r_zeros + c_zeros
    elif zero_lt is not None and one_gt is not None:
        # +, -
        crop_im = depth_image[:offset[0], offset[1]:]
        pad_im = tf.concat(1,
            [tf.concat(0,
                [tf.zeros([
                    np.abs(offset[0]), hw[1] - np.abs(offset[1])]), crop_im]),
                tf.zeros([hw[0], offset[1]])]) # r_zeros + image + c_zeros
    elif zero_gt is not None and one_lt is not None:
        # -, +
        crop_im = depth_image[offset[0]:, :offset[1]]
        pad_im = tf.concat(1,
            [tf.zeros([hw[0],
                np.abs(offset[1])]), tf.concat(0,
                [crop_im, tf.zeros([offset[0], hw[1] - np.abs(offset[1])])])])  # c_zeros + image + r_zeros
    elif zero_gt is not None and one_gt is not None:
        # -, -
        crop_im = depth_image[offset[0]:, offset[1]:]
        pad_im = tf.concat(1,
            [tf.concat(0,
                [crop_im, tf.zeros([offset[0], hw[1] - np.abs(offset[1])])]), tf.zeros(
                [hw[0], offset[1]])])  # image + r_zeros + c_zeros
    return pad_im


def random_offsets(offset_max):
    return [np.random.randint(-offset_max, offset_max),
               np.random.randint(
                   -offset_max, offset_max)]


def get_labels(raw_labels, background_constant, label_values):
    """get_labels in a loop"""
    ii = 0
    h,w, d = raw_labels.shape
    lab_arr = sp.zeros((h,w)).astype('float32')

    #show_colors(raw_labels)
    for v, label in enumerate(label_values.values()):
        ind = ((raw_labels == label).prod(2)).astype('bool')
        lab_arr[ind] = v + 1
    ind = ((raw_labels == 0).prod(2)).astype('bool')
    lab_arr[ind] = background_constant
    return lab_arr


def get_label(raw_labels, background_constant, label_values, resize=None):
    """docstring for get_labels"""
    label_im = misc.imread(raw_labels)
    if resize is not None:
        label_im = misc.imresize(label_im, resize)

    hw = label_im.shape
    lab_arr = np.zeros((hw[0], hw[1])).astype('float32')
    bg_im = np.sum(label_im,axis=-1).reshape(label_im.shape[0] * label_im.shape[1])

    #show_colors(raw_labels)
    label_matrix = np.asarray(label_values.values())
    res_im = label_im.reshape(label_im.shape[0] * label_im.shape[1], label_im.shape[2])
    dists = np.argmin(cdist(res_im, label_matrix, 'euclidean'), axis=1)
    dists[bg_im == 0] = background_constant
    dists = dists.reshape(label_im.shape[0], label_im.shape[1])
    return lab_arr

