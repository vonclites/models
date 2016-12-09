from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = 'casia_all_%s.tfrecord'

SPLITS_TO_SIZES = {'train': 16000, 'test': 4000}

_NUM_CLASSES = 2

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'NIR ocular image.',
    'label': '1 for fake iris or 0 for real iris.',
}

_COARSE_LABEL_FILENAME = 'coarse_labels.txt'

def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    if split_name not in SPLITS_TO_SIZES:
      raise ValueError('split name %s was not recognized.' % split_name)

    if not file_pattern:
      file_pattern = _FILE_PATTERN

    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/class/label': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        'image/class/coarse_label': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        'image/filepath': tf.FixedLenFeature((), tf.string, default_value=''),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(shape=[224, 224, 1], channels=1),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
        'coarse_label': slim.tfexample_decoder.Tensor('image/class/coarse_label'),
        'filepath': slim.tfexample_decoder.Tensor('image/filepath'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None
    if dataset_utils.has_labels(dataset_dir):
      labels_to_names = dataset_utils.read_label_file(dataset_dir)

    dataset = slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=SPLITS_TO_SIZES[split_name],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=_NUM_CLASSES,
        labels_to_names=labels_to_names)

    coarse_labels_to_names = None
    if dataset_utils.has_labels(dataset_dir, _COARSE_LABEL_FILENAME):
      coarse_labels_to_names = dataset_utils.read_label_file(
          dataset_dir, _COARSE_LABEL_FILENAME)
    dataset.coarse_labels_to_names = coarse_labels_to_names

    return dataset
