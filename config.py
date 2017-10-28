from os.path import join as pjoin


#  Main configuration file for monkey tracking
class deepgazeConfig(object):

    def __init__(self):

        # Directory settings
        self.base_dir = '/media/data_cifs/ajones/smart_playroom/smart_playroom/deepgaze/deepgaze_generic'

        # USE FOR LARGE DATASET
        self.image_dir = '/media/data_cifs/ajones/salicon_data/images/train/'
        self.label_dir = '/media/data_cifs/ajones/salicon_data/annotations/train2014examples/'
        self.label_json = '/media/data_cifs/ajones/salicon_data/annotations/fixations_train2015r1.json'
        self.num_images = 10000
        self.label_size = 112

        # USE FOR SMALL DATASET
        # self.image_dir = '/media/data_cifs/ajones/salicon_data/images/train2014examples/'
        # self.label_dir = '/media/data_cifs/ajones/salicon_data/annotations/train2014examples/'
        # self.label_json = '/media/data_cifs/ajones/salicon_data/annotations/fixations_train2014examples.json'
        # self.num_images = 10.

        self.vgg16_npy_path = '/media/data_cifs/clicktionary/pretrained_weights/vgg16.npy'

        self.mit1003_image_dir = '/media/data_cifs/ajones/deepgaze/salicon_prep_g11/mit1003_data/ALLSTIMULI'
        self.mit1003_fixmap_dir = '/media/data_cifs/ajones/deepgaze/salicon_prep_g11/mit1003_data/ALLFIXATIONMAPS'

        self.image_regex = '*.jpg'
        self.model_output = pjoin(self.base_dir, 'model_output')
        self.tfrecord_dir = '/media/data_cifs/ajones/smart_playroom/smart_playroom/deepgaze/contextual_circuit_saliency/salicon_prep/tfrecords'
        self.train_summaries = self.base_dir
        self.train_checkpoint = pjoin(self.base_dir, 'train_checkpoint/')

        self.train_data = pjoin(
            self.tfrecord_dir, 'train_salicon.tfrecords')
        self.val_data = pjoin(self.tfrecord_dir, 'val_salicon.tfrecords')
        self.resize = [224, 224]

        # Model settings
        self.num_epochs = None
        self.model_type = 'vgg_19_feature_model'
        # , 'fc_conv1', 'fc_conv2', 'fc_conv3']  # layers for batchnorm
        self.batch_norm = ['conv1', 'conv2',
                           'conv3', 'conv5_1', 'conv5_2', 'conv5_3']
        self.data_augmentations = ['left_right']
        self.trainable_layers = ['fc_conv1',
                                 'fc_conv2', 'fc_conv3', 'fc_conv4']
        # ['left_right, up_down, random_crop,
        # random_brightness, random_contrast, rotate']
        self.train_batch = 8
        self.validation_batch = 8
        self.ratio = 0.5
        self.lr = 1e-4
        self.tf_summary_dir = pjoin(self.tfrecord_dir, 'summaries')
        self.keep_checkpoints = 100
        # for a weighted cost. First entry = background.

        # For the rf if you choose to use it (not implemented)
        self.show_output = False
        # self.tree_depth = 20
        # self.num_trees = 3
        # self.max_nodes = 1000

        # Training settings
        self.batch_size = 16
        self.train_steps = 1000
        self.num_classes = 2
        self.use_training_loss = False  # early stopping based on loss
        self.early_stopping_rounds = 100
        self.test_proprtion = 0.1  # TEST_RATIO
        self.mean_file = 'mean_file.npy'
