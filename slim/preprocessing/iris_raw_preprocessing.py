from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def common_preprocess(image, output_height, output_width):
  image = tf.image.grayscale_to_rgb(image)
  image = tf.expand_dims(image, 0)
  image = tf.image.resize_bicubic(image, [output_height, output_width])
  image = tf.squeeze(image)
  image.set_shape([output_height, output_width, 3])
  return image


def preprocess_image(image, output_height, output_width, is_training=False):
  image = common_preprocess(image, output_height, output_width)
  if is_training:
    return tf.image.random_flip_left_right(image)
  else:
    return image