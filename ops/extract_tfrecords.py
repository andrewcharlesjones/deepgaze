from PIL import Image
import numpy as np
import tensorflow as tf

def read_and_decode(filename_queue):
   reader = tf.TFRecordReader()
   _, serialized_example = reader.read(filename_queue)
   features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'image': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
          'feat': tf.FixedLenFeature([], tf.string)
      }
   )
   image = tf.decode_raw(features['image'], tf.int32)
   label = tf.cast(features['label'], tf.int32)
   feat = tf.decode_raw(features['feat'], tf.float32)
   return image, label, feat


def get_records(FILE, config):
   ims = []
   labs = []
   feats = []
   with tf.Session() as sess:
       filename_queue = tf.train.string_input_producer([ FILE ])
       image, label, feat = read_and_decode(filename_queue)
       image = tf.reshape(image, tf.pack(config.resize))
       init_op = tf.initialize_all_variables()
       sess.run(init_op)
       coord = tf.train.Coordinator()
       threads = tf.train.start_queue_runners(coord=coord)
       while 1:
           im, lab, fe = sess.run([image, label, feat])
           import ipdb;ipdb.set_trace()
           im = Image.fromarray(im, 'RGB')[:, :, :, None]
           ims = np.append(ims, im, axis=-1)
           labs = np.append(labs, lab[:, :, :, None], axis=-1)
           feats = np.append(feats, fe[:, :, :, None], axis=-1)
       coord.request_stop()
       coord.join(threads)
   return ims, labs, feats

