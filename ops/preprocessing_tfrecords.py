import re
import os
import sys
import traceback
import shutil
import numpy as np
import tensorflow as tf
from scipy import misc
from glob import glob
from tqdm import tqdm


# # # For prepare tf records
def flatten_list(l):
    return [item for sublist in l for item in sublist]


def get_file_list(GEDI_path, label_directories, im_ext):
    files = []
    for idx in label_directories:
        print 'Getting files from: %s' % (os.path.join(
            GEDI_path, idx, '*' + im_ext))
        dir_files = glob(os.path.join(GEDI_path, idx, '*' + im_ext))
        files += dir_files
        print 'Found %s files' % len(dir_files)
    return files


def write_label_list(files, label_list):
    with open(label_list, "w") as f:
        f.writelines([ln + '\n' for ln in files])


def split_files(files, train_proportion, tvt_flags):
    num_files = len(files)
    all_labels = find_label(files)
    files = np.asarray(files)
    rand_order = np.random.permutation(num_files)
    split_int = int(np.round(num_files * train_proportion))
    train_inds = rand_order[:split_int]
    new_files = {}
    new_files['train'] = files[train_inds]
    new_files['train_labels'] = all_labels[train_inds]
    if 'test' in tvt_flags and 'val' in tvt_flags:
        hint = int(np.round(num_files * (1-train_proportion)))
        val_inds = rand_order[split_int:split_int + hint]
        test_inds = rand_order[split_int + hint:]
        new_files['val'] = files[val_inds]
        new_files['val_labels'] = all_labels[val_inds]
        new_files['test'] = files[test_inds]
        new_files['test_labels'] = all_labels[test_inds]
    elif 'test' in tvt_flags and 'val' not in tvt_flags:
        val_inds = rand_order[split_int:]
        new_files['test'] = files[test_inds]
        new_files['test_labels'] = all_labels[test_inds]
    elif 'val' in tvt_flags and 'test' not in tvt_flags:
        val_inds = rand_order[split_int:]
        new_files['val'] = files[val_inds]
        new_files['val_labels'] = all_labels[val_inds]
    return new_files


def move_files(files, target_dir):
    for idx in files:
        shutil.copyfile(idx, target_dir + re.split('/', idx)[-1])


def load_im_batch(files, hw, normalize):
    labels = []
    images = []
    if len(hw) == 2:
        rep_channel = True
    else:
        rep_channel = False
    for idx in files:
        # the parent directory is the label
        labels.append(re.split('/', idx)[-2])
        if rep_channel:
            images.append(np.repeat(misc.imread(idx)[:, :, None], 3, axis=-1))
        else:
            images.append(misc.imread(idx))
    if normalize is not None:
        images = [im.astype(np.float32)/255 for im in images]
    # transpose images to batch,ch,h,w
    return np.asarray(images).transpose(0, 3, 1, 2), np.asarray(labels)


def find_label(files):
    _, c = np.unique(
        [re.split('/', l)[-2] for l in files], return_inverse=True)
    return c


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def image_to_tfexample(image_data, image_format, height, width, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/class/label': int64_feature(class_id),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
    }))


def _add_to_tfrecord(
        images, labels, tfrecord_writer, im_ext, hw, tf_im_conv, offset=0):
    num_images = images.shape[0]
    with tf.Graph().as_default():
        image_placeholder = tf.placeholder(dtype=tf.uint8)
        encoded_image = tf_im_conv(image_placeholder)
        with tf.Session('') as sess:
            for j in range(num_images):
                # Have to do below to translate between tf and byte encodigns
                image = np.squeeze(images[j]).transpose((1, 2, 0))
                label = labels[j]
                im_string = sess.run(
                    encoded_image, feed_dict={image_placeholder: image})
                example = image_to_tfexample(
                    im_string, im_ext, hw[0], hw[1], label)
                tfrecord_writer.write(example.SerializeToString())
    return offset + num_images


def image_converter(im_ext):
    if im_ext == '.jpg' or im_ext == '.jpeg' or im_ext == '.JPEG':
        out_fun = tf.image.encode_jpeg
    elif im_ext == '.png':
        out_fun = tf.image.encode_png
    else:
        print '-'*60
        traceback.print_exc(file=sys.stdout)
        print '-'*60
    return out_fun


def process_image_data(
        im_key, im_dict, output_file, im_ext, train_shards, hw, normalize):
    print 'Building: %s' % output_file
    files = im_dict[im_key]
    all_labels = im_dict[im_key + '_labels']
    output_pointer = output_file + im_key + '.tfrecords'
    tf_im_conv = image_converter(im_ext)
    with tf.python_io.TFRecordWriter(output_pointer) as tfrecord_writer:
        num_shards = np.round(len(files)/train_shards)
        batch_idx = np.repeat(np.arange(num_shards), train_shards)
        print 'Storing data in {} shards. Using {}/{} training images \
            (adjust shard size to change proportion).'.format(
                num_shards, len(batch_idx), len(files))
        batch_idx = np.concatenate((
            batch_idx, np.ones((len(files) - len(batch_idx))) * -1))
        for idx in tqdm(range(num_shards)):
            # load_im_batch also returns a string array of labels
            images, _ = load_im_batch(files[batch_idx == idx], hw, normalize)
            labels = all_labels[batch_idx == idx]
            _add_to_tfrecord(
                images, labels, tfrecord_writer, im_ext, hw, tf_im_conv)


def simple_tf_records(
        im_key, im_dict, output_file, im_ext, train_shards, hw, normalize):
    print('Building', output_file)
    files = im_dict[im_key]
    all_labels = im_dict[im_key + '_labels']
    output_pointer = output_file + im_key + '.tfrecords'
    with tf.python_io.TFRecordWriter(output_pointer) as tfrecord_writer:
        for idx in tqdm(range(len(files))):
            image, _ = load_im_batch([files[idx]], hw, normalize)
            label = all_labels[idx]
            # construct the Example proto boject
            example = tf.train.Example(
                # Example contains a Features proto object
                features=tf.train.Features(
                    # Features has a map of string to Feature proto objects
                    feature={
                        # A Feature contains one of either a int64_list,
                        # float_list, or bytes_list
                        'label': int64_feature(label),
                        'image': bytes_feature(image.tostring()),
                        # tf.train.Feature(int64_list=tf.train.Int64List(value=image.astype('int64'))),
                    }
                )
            )
            # use the proto object to serialize the example to a string
            serialized = example.SerializeToString()
            # write the serialized object to disk
            tfrecord_writer.write(serialized)


def extract_to_tf_records(files, label_list, output_pointer, config, k):
    print 'Building: %s' % config.tfrecord_dir
    max_array = np.zeros(len(files))
    with tf.python_io.TFRecordWriter(output_pointer) as tfrecord_writer:
        for idx, f in tqdm(enumerate(files), total=len(files)):
            image = produce_patch(
                f, config.channel, config.panel,
                divide_panel=config.divide_panel).astype(np.float32)
            max_array[idx] = np.max(image)
            label = label_list[idx]
            # construct the Example proto boject
            example = tf.train.Example(
                # Example contains a Features proto object
                features=tf.train.Features(
                    # Features has a map of string to Feature proto objects
                    feature={
                        # A Feature contains one of either a int64_list,
                        # float_list, or bytes_list
                        'label': int64_feature(label),
                        'image': bytes_feature(image.tostring()),
                        # tf.train.Feature(int64_list=tf.train.Int64List(value=image.astype('int64'))),
                    }
                )
            )
            # use the proto object to serialize the example to a string
            serialized = example.SerializeToString()
            # write the serialized object to disk
            tfrecord_writer.write(serialized)
    # Calculate ratio of +:-
    lab_counts = np.asarray([np.sum(label_list == 0), np.sum(label_list == 1)]).astype(float)
    ratio = lab_counts / np.asarray((len(label_list))).astype(float)
    print 'Data ratio is %s' % ratio
    np.savez(
        os.path.join(
            config.tfrecord_dir, k + '_' + config.max_file),
        max_array=max_array, ratio=ratio)
    return max_array


def write_label_file(labels_to_class_names, dataset_dir,
                     filename='labels.txt'):
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'w') as f:
        for label in labels_to_class_names:
            class_name = labels_to_class_names[label]
            f.write('%d:%s\n' % (label, class_name))
