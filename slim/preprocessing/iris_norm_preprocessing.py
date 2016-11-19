from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

_IMAGE_HEIGHT = 32
_IMAGE_WIDTH = 256


def preprocess_for_train(image, output_height, output_width):
  """Preprocesses the given image for training.

  Note that the actual resizing scale is sampled from
    [`resize_size_min`, `resize_size_max`].

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    resize_side_min: The lower bound for the smallest side of the image for
      aspect-preserving resizing.
    resize_side_max: The upper bound for the smallest side of the image for
      aspect-preserving resizing.

  Returns:
    A preprocessed image.
  """
  image = tf.image.grayscale_to_rgb(image)
  image.set_shape([output_height, output_width, 3])
  image = tf.to_float(image)
  image = tf.image.random_flip_left_right(image)
  return image


def preprocess_for_eval(image, output_height, output_width):
  image = tf.image.grayscale_to_rgb(image)
  image.set_shape([output_height, output_width, 3])
  image = tf.to_float(image)
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