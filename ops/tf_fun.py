import re
import os
import numpy as np
import tensorflow as tf
from glob import glob


def make_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def fine_tune_prepare_layers(tf_vars, finetune_vars):
    ft_vars = []
    other_vars = []
    for v in tf_vars:
        ss = [v.name.find(x) != -1 for x in finetune_vars]
        if True in ss:
            ft_vars.append(v)
        else:
            other_vars.append(v)
    return other_vars, ft_vars


def ft_optimized(cost, var_list_1, var_list_2, optimizer, lr_1, lr_2):
    """Applies different learning rates to specified layers."""
    opt1 = optimizer(lr_1)
    opt2 = optimizer(lr_2)
    grads = tf.gradients(cost, var_list_1 + var_list_2)
    grads1 = grads[:len(var_list_1)]
    grads2 = grads[len(var_list_1):]
    train_op1 = opt1.apply_gradients(zip(grads1, var_list_1))
    train_op2 = opt2.apply_gradients(zip(grads2, var_list_2))
    return tf.group(train_op1, train_op2)


def ft_non_optimized(cost, other_opt_vars, ft_opt_vars, optimizer, lr_1, lr_2):
    op1 = tf.train.AdamOptimizer(lr_1).minimize(cost, var_list=other_opt_vars)
    op2 = tf.train.AdamOptimizer(lr_2).minimize(cost, var_list=ft_opt_vars)
    return tf.group(op1, op2)  # ft_optimize is more efficient.


def class_accuracy(pred, targets):
    return tf.reduce_mean(
        tf.to_float(
            tf.equal(tf.argmax(pred, 1), tf.cast(
                targets, dtype=tf.int64))))  # assuming targets is an index


def count_nonzero(data):
    return tf.reduce_sum(tf.cast(tf.not_equal(data, 0), tf.float32))


def fscore(pred, targets):
    predicted = tf.argmax(pred, dimension=3)

    predicted = tf.reshape(predicted, [16, 12544])

    # import pdb; pdb.set_trace()

    # Count true +, true -, false + and false -.
    targets = tf.cast(targets, tf.int64)
    tp = count_nonzero(predicted * targets)
    fp = count_nonzero(predicted * (targets - 1))
    fn = count_nonzero((predicted - 1) * targets)

    # Calculate accuracy, precision, recall and F1 score.
    # accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fmeasure = (2 * precision * recall) / (precision + recall)
    return fmeasure, targets, predicted


def tf_confusion_matrix(pred, targets):
    return tf.contrib.metrics.confusion_matrix(pred, targets)


def softmax_cost(logits, labels, ratio=None):
    if ratio != None:
        reversed_labels = tf.sub(1, tf.cast(labels, tf.int32))
        weights = tf.mul(tf.cast(reversed_labels, tf.float32), 1.)
        weights = tf.sub(1.0, weights)

        # one_hot_labels = tf.one_hot(tf.cast(labels, tf.uint8), 2)
        # reversed_one_hot_labels = tf.sub(1, tf.cast(one_hot_labels, tf.int32))
        # one_hot_weights = tf.mul(tf.cast(reversed_one_hot_labels, tf.float32), .5)
        # one_hot_weights = tf.sub(1.0, one_hot_weights)

        logits_softmax = tf.nn.softmax(logits)
        weighted_entropy = tf.nn.weighted_cross_entropy_with_logits(logits_softmax, tf.one_hot(tf.cast(labels, tf.uint8), 2), tf.constant([0.1, 0.9]))
        return tf.reduce_mean(weighted_entropy)


        # weighted_logits = logits * [.1, .9]
        # import pdb; pdb.set_trace()
        # return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(weighted_logits, tf.cast(labels, tf.int32)))
        # ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, tf.cast(labels, tf.int32))

        
        weighted_entropy = tf.mul(ce, weights)
        # return tf.reduce_mean(tf.clip_by_value(weighted_entropy,1e-4,1e4))
        return tf.reduce_mean(tf.clip_by_value(weighted_entropy, 1e-4, 1e4)), weights, labels
    else:
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, tf.cast(labels, tf.int32)))

def salicon_cost(logits, labels):
    predicted = tf.argmax(logits, dimension=2)
    # weights = np.ones([16, 12544, 1]) * .3
    tf_weights = tf.convert_to_tensor(weights)
    weighted_predicted = tf.matmul(tf.cast(predicted, tf.float64), tf_weights)
    # import pdb; pdb.set_trace()
    return tf.nn.sparse_softmax_cross_entropy_with_logits(weighted_predicted, labels)

def euclidean_loss(pred, labels, batch_size):

    # pred_normalized = tf.nn.l2_normalize(tf.reshape(pred, [16, 12544, 1]), dim=2)

    reshaped_predictions = tf.reshape(pred, [batch_size, 12544, 1])
    # pred_normalized = tf.nn.l2_normalize(reshaped_predictions, dim=1) + 1e-6 # this is [16, 12544, 1])

    # min_pred = tf.reduce_min(reshaped_predictions)
    # pred = tf.sub(reshaped_predictions, min_pred)
    # max_pred = tf.reduce_max(pred) + 1e-6
    # pred_normalized = tf.div(pred, max_pred)

    # # reshaped_labels = tf.reshape(labels, [16, 12544, 1])
    # min_label = tf.reduce_min(labels)
    # lab = tf.sub(labels, min_label)
    # max_label = tf.reduce_max(lab) + 1e-6
    # lab_normalized = tf.div(lab, max_label)

    pred_reshaped = tf.reshape(pred, [batch_size, 12544, 1])
    # pred_normalized = pred_reshaped
    # out_tensor = []
    # for i in range(0, batch_size):
    #     curr_tensor = pred_reshaped[i]
        
    #     curr_tensor_min = tf.reduce_min(curr_tensor)
    #     zeroed_tensor = curr_tensor - curr_tensor_min + 1e-6
    #     curr_tensor_max = tf.reduce_max(zeroed_tensor)
    #     norm_tensor = zeroed_tensor / curr_tensor_max
    #     out_tensor.append(norm_tensor)

    # pred_normalized = tf.stack(out_tensor)
    # pred_normalized = tf.nn.l2_normalize(pred_reshaped, 0)


    lab_reshaped = tf.reshape(labels, [batch_size, 12544, 1])
    # out_tensor = []
    # for i in range(0, batch_size):
    #     curr_tensor = lab_reshaped[i]
        
    #     curr_tensor_min = tf.reduce_min(curr_tensor)
    #     zeroed_tensor = curr_tensor - curr_tensor_min + 1e-6
    #     curr_tensor_max = tf.reduce_max(zeroed_tensor)
    #     norm_tensor = zeroed_tensor / curr_tensor_max
    #     out_tensor.append(norm_tensor)

    # lab_normalized = tf.stack(out_tensor)
    # lab_normalized = tf.nn.l2_normalize(lab_reshaped, 0)   

    # pred_normalized = tf.div(pred_normalized, tf.reduce_max(pred_normalized))
    # lab_normalized = tf.div(lab_normalized, tf.reduce_max(lab_normalized)) 

    # error_matrix = tf.square(tf.subtract(pred_normalized, tf.cast(lab_normalized, tf.float32)))
    error_matrix = tf.square(tf.subtract(pred_reshaped, tf.cast(lab_reshaped, tf.float32)))

    # l2diff = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(pred_normalized, tf.cast(lab_normalized, tf.float32)))))
    l2diff = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(pred_reshaped, tf.cast(lab_reshaped, tf.float32)))))
    
    return l2diff, error_matrix

def special_sauce_loss(pred, labels):

    reshaped_predictions = tf.reshape(pred, [16, 12544, 1])
    pred_normalized = tf.nn.l2_normalize(reshaped_predictions, dim=1) + 1e-6 # this is [16, 12544, 1])

    l2diff = tf.square(tf.sub(tf.squeeze(pred_normalized), tf.cast(labels, tf.float32)))

    denominator = 1 - labels + 0.001

    loss = tf.sqrt(tf.reduce_sum(tf.div(l2diff, denominator)))

    return loss
    

    # reshaped_predictions = tf.squeeze(tf.reshape(pred, [16, 12544, 1]))
    # # pred_normalized = tf.nn.l2_normalize(reshaped_predictions, dim=1) + 1e-6 # this is [16, 12544, 1])

    # min_pred = tf.reduce_min(reshaped_predictions)
    # reshaped_predictions = tf.sub(reshaped_predictions, min_pred)
    # max_pred = tf.reduce_max(reshaped_predictions)
    # reshaped_predictions = tf.div(reshaped_predictions, max_pred)

    # import pdb; pdb.set_trace()
    # dist_pred_labels = tf.pow(tf.sub(reshaped_predictions, labels), 2)
    # loss = tf.reduce_sum(tf.sqrt(tf.div(dist_pred_labels, (1 - labels + .001))))
    # return loss

def normal_log(X, Y, sigma, left=-np.inf, right=np.inf):
    pred_normalized = tf.nn.l2_normalize(tf.reshape(X, [16, 12544, 1]), dim=2)
    pred_normalized = tf.squeeze(pred_normalized)

    N = 16. * 12544.

    import pdb; pdb.set_trace()

    beta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(pred_normalized), pred_normalized)), tf.transpose(pred_normalized)), Y)

    sigma = tf.reduce_sum(tf.square(tf.sub(Y, tf.matmul(pred_normalized, beta)))) / N
    
    val = -tf.log(tf.constant(np.sqrt(2 * np.pi), dtype=tf.float32)) - \
        (N / 2) * tf.log(tf.pow(sigma, 2)) - \
        (tf.div(1., (2 * tf.pow(sigma, 2)))) * (tf.reduce_sum(tf.square(tf.sub(Y, tf.matmul(pred_normalized, beta)))) / N)

    return -val
    #  * sigma) \
    #             - tf.pow(tf.squeeze(pred_normalized) - mu, 2) / (tf.constant(2, dtype=tf.float32) \
    #                 * tf.pow(sigma, 2))

    # return -tf.reduce_sum(val)


def find_ckpts(config, dirs=None):
    if dirs is None:
        dirs = sorted(  
            glob(
                config.train_checkpoint + config.which_dataset + '*'),
            reverse=True)[0]  # only the newest model run
    ckpts = sorted(glob(dirs + '/*.ckpt*'))
    ckpts = [ck for ck in ckpts if '.meta' not in ck]  # Don't include metas
    ckpt_names = [re.split('-', ck)[-1] for ck in ckpts]
    return np.asarray(ckpts), np.asarray(ckpt_names)
