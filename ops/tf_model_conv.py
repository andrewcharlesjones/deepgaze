import os
import re
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
import sys
from ops.data_loader import inputs
from ops.tf_fun import softmax_cost, fscore, make_dir, count_nonzero, salicon_cost, euclidean_loss, normal_log, special_sauce_loss
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.misc import imread
from scipy.misc import imsave

sys.path.append(
    '/media/data_cifs/ajones/smart_playroom/smart_playroom/deepgaze/deepgaze_generic/models')

from deepgaze_model import model_struct


def train_and_eval(config):
    """Train and evaluate the model."""

    # Prepare model training
    dt_stamp = re.split(
        '\.', str(datetime.now()))[0].\
        replace(' ', '_').replace(':', '_').replace('-', '_')
    dt_dataset = config.model_type + '_' + dt_stamp + '/'
    config.train_checkpoint = os.path.join(
        config.model_output, dt_dataset)  # timestamp this run
    config.summary_dir = os.path.join(
        config.train_summaries, config.model_output, dt_dataset)
    dir_list = [config.train_checkpoint, config.summary_dir]
    [make_dir(d) for d in dir_list]

    # Prepare model inputs
    train_data = config.train_data
    validation_data = config.val_data

    # Prepare data on CPU
    with tf.device('/cpu:0'):
        train_images, train_labels = inputs(
            tfrecord_file=train_data,
            batch_size=config.train_batch,
            im_size=config.resize,
            model_input_shape=config.resize,
            train=None,
            img_mean_value='train_mean_big.npz',
            num_epochs=config.num_epochs)
        val_images, val_labels = inputs(
            tfrecord_file=validation_data,
            batch_size=config.train_batch,
            im_size=config.resize,
            model_input_shape=config.resize,
            train=None,
            img_mean_value='train_mean_big.npz',
            num_epochs=config.num_epochs)
        tf.summary.image('train images', tf.cast(train_images, tf.uint8))
        tf.summary.image('validation images', tf.cast(val_images, tf.uint8))
        tf.summary.image('train labels', tf.cast(tf.reshape(
            train_labels, [config.train_batch, 112, 112, 1]), tf.float32))
        tf.summary.image('validation labels', tf.cast(tf.reshape(
            val_labels, [config.train_batch, 112, 112, 1]), tf.float32))

    num_train_imgs = 0
    for record in tf.python_io.tf_record_iterator(train_data):
        num_train_imgs += 1

    num_val_imgs = 0
    for record in tf.python_io.tf_record_iterator(validation_data):
        num_val_imgs += 1

    print 'Number of training images', num_train_imgs
    print 'Number of validation images', num_val_imgs

    # Prepare model on GPU
    with tf.device('/gpu:0'):
        with tf.variable_scope('cnn') as scope:

            model = model_struct(config.vgg16_npy_path)
            train_mode = tf.get_variable(name='training', initializer=True)
            model.build(
                train_images,
                train_mode=train_mode, batchnorm=config.batch_norm,
                trainable_layers=config.trainable_layers,
                config=config, shuffled=False)

            # Prepare the cost function
            cost, train_error_matrix = euclidean_loss(
                model.prediction, train_labels, config.train_batch)

            tf.summary.scalar("train cost", cost)

            train_op = tf.train.AdamOptimizer(config.lr).minimize(cost)

            tf.summary.image("prediction", model.prediction)

            # Setup validation op
            if validation_data is not False:
                scope.reuse_variables()
                # Validation graph is the same as training except no batchnorm
                val_model = model_struct(
                    vgg16_npy_path='/media/data_cifs/ajones/deepgaze/salicon_prep_g11/vgg19.npy')
                val_model.build(
                    val_images, train_mode=train_mode,
                    batchnorm=config.batch_norm,
                    trainable_layers=config.trainable_layers,
                    config=config, shuffled=True)

                # Calculate validation accuracy
                val_cost, val_error_matrix = euclidean_loss(
                    val_model.prediction, val_labels, config.train_batch)

                tf.summary.scalar("validation cost", val_cost)
                tf.summary.image("val prediction", val_model.prediction)

    # Set up summaries and saver
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    summary_op = tf.summary.merge_all()

    # Initialize the graph
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    # Need to initialize both of these if supplying num_epochs to inputs
    sess.run([tf.group(tf.global_variables_initializer()),
              tf.local_variables_initializer()])
    sess.run([tf.local_variables_initializer()])
    summary_writer = tf.summary.FileWriter(config.summary_dir, sess.graph)

    # Set up exemplar threading
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Start training loop
    np.save(config.train_checkpoint, config)
    step, epoch_no, val_max, losses = 0, 0, 0, []

    epoch_loss_values = []
    best_val_loss = float('inf')

    try:
        while not coord.should_stop():
            start_time = time.time()
            halt = False

            if step % 20 == 0:

                if config.show_output:
                    _, loss_value, val_loss, summary_str, imgs, labs, predictions, cb_logits, logits, te_mat, preds_presoft = sess.run(
                        [train_op, cost, val_cost, summary_op,
                         tf.reshape(train_images[0:3], [3, 224, 224, 3]),
                         tf.reshape(train_labels[0:3], [3, 112, 112]),
                         tf.reshape(model.prediction[0:3], [3, 112, 112]),
                         tf.reshape(model.center_bias_logits[
                                    0:3], [3, 112, 112]),
                         tf.reshape(model.logits[0:3], [3, 112, 112]),
                         tf.reshape(train_error_matrix[0:3], [3, 112, 112]),
                         tf.reshape(model.prediction_presoft[0:3], [3, 112, 112])])
                    # plt.imshow(np.mean(sess.run(model.feature_encoder)[0], 2))
                    # plt.show()
                    # import ipdb; ipdb.set_trace()
                else:
                    _, loss_value, val_loss, summary_str, imgs, labs, predictions, logits = sess.run(
                        [train_op, cost, val_cost, summary_op,
                         tf.reshape(train_images[0:3], [3, 224, 224, 3]),
                         tf.reshape(train_labels[0:3], [3, 112, 112]),
                         tf.reshape(model.prediction[0:3], [3, 112, 112]),
                         tf.reshape(model.logits[0:3], [3, 112, 112])])

            if step % 60 == 0:
                np.save('running_imgs', imgs)
                np.save('running_labs', labs)
                np.save('running_preds', predictions)
                np.save('running_logits', logits)

                summary_writer.add_summary(summary_str, step)

                duration = time.time() - start_time

                # Training status
                format_str = ('%s: step %d, loss = %.2f, val loss = %.2f (%.1f examples/sec; '
                              '%.3f sec/batch) | logdir = %s\n')
                print(format_str % (datetime.now(), step, loss_value, val_loss,
                                    config.train_batch / duration,
                                    float(duration), config.summary_dir))

                if val_loss < best_val_loss:
                    saver.save(
                        sess, os.path.join(
                            config.train_checkpoint,
                            'model_' + str(step) + '.ckpt'), global_step=step)

                if config.show_output:  # and step > 300 and step % 20 == 0:
                    num_columns = 6
                    num_imgs_plot = 3

                    for i in range(num_imgs_plot):

                        plt.subplot(num_imgs_plot, num_columns,
                                    i * num_columns + 1)
                        im = plt.imshow(imgs[i])
                        plt.colorbar(im)
                        plt.subplot(num_imgs_plot, num_columns,
                                    i * num_columns + 2)
                        lab = plt.imshow(labs[i])
                        plt.colorbar(lab)
                        plt.subplot(num_imgs_plot, num_columns,
                                    i * num_columns + 3)
                        logits_out = plt.imshow(logits[i])
                        plt.colorbar(logits_out)
                        plt.subplot(num_imgs_plot, num_columns,
                                    i * num_columns + 4)
                        cbl = plt.imshow(cb_logits[i])
                        plt.colorbar(cbl)
                        plt.subplot(num_imgs_plot, num_columns,
                                    i * num_columns + 5)
                        preds = plt.imshow(preds_presoft[i])
                        plt.colorbar(preds)
                        plt.subplot(num_imgs_plot, num_columns,
                                    i * num_columns + 6)
                        preds = plt.imshow(predictions[i])
                        plt.colorbar(preds)
                    plt.show()

            else:
                _, loss_value = sess.run([train_op, cost])

            # assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            if np.isnan(loss_value):
                print 'yikes. nan loss -- check your cost function.'
                import pdb
                pdb.set_trace()

            # End iteration
            step += 1

    except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps.' % (epoch_no, step))
    finally:
        coord.request_stop()
        np.save(os.path.join(config.tfrecord_dir, 'training_loss'), losses)
    coord.join(threads)
    sess.close()
