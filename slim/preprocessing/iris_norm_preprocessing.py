from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

_IMAGE_HEIGHT = 64
_IMAGE_WIDTH = 512


def preprocess_for_train(image, output_height, output_width):
  image = tf.image.grayscale_to_rgb(image)
  image.set_shape([output_height, output_width, 3])
  if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  image = tf.image.random_flip_left_right(image)
  return image


def preprocess_for_eval(image, output_height, output_width):
  image = tf.image.grayscale_to_rgb(image)
  image.set_shape([output_height, output_width, 3])
  if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  return image


def preprocess_image(image, output_height, output_width, is_training=False):
  # train_image_classifier.py cannot handle non-square sizes
  # Must overwrite output shapes with hard-coded values
  output_height = _IMAGE_HEIGHT
  output_width = _IMAGE_WIDTH
  if is_training:
    return preprocess_for_train(image, output_height, output_width)
  else:
    return preprocess_for_eval(image, output_height, output_width)