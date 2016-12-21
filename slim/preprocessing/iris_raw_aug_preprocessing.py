from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from preprocessing.inception_preprocessing import distorted_bounding_box_crop
from preprocessing.inception_preprocessing import apply_with_random_selector
from preprocessing.inception_preprocessing import distort_color

slim = tf.contrib.slim


def common_preprocess(image, output_height, output_width):
  image = tf.image.grayscale_to_rgb(image)
  image.set_shape([output_height, output_width, 3])
  if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  return image


def preprocess_image(image, height, width, is_training=False):
  fast_mode=True
  image = common_preprocess(image, height, width)
  if is_training:
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                       dtype=tf.float32,
                       shape=[1, 1, 4])
    image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                  bbox)
    tf.image_summary('image_with_bounding_boxes', image_with_box, 3)
    distorted_image, _ = distorted_bounding_box_crop(image, bbox,
                                                     area_range=(0.4,1.0))
    distorted_image.set_shape([None, None, 3])
    image_with_distorted_box = tf.image.draw_bounding_boxes(
        tf.expand_dims(image, 0), distorted_bbox)
    tf.image_summary('images_with_distorted_bounding_box',
                     image_with_distorted_box)
    # This resizing operation may distort the images because the aspect
    # ratio is not respected. We select a resize method in a round robin
    # fashion based on the thread number.
    # Note that ResizeMethod contains 4 enumerated resizing methods.

    # We select only 1 case for fast_mode bilinear.
    num_resize_cases = 1 if fast_mode else 4
    distorted_image = apply_with_random_selector(
        distorted_image,
        lambda x, method: tf.image.resize_images(x, [height, width], method=method),
        num_cases=num_resize_cases)
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = apply_with_random_selector(
        distorted_image,
        lambda x, ordering: distort_color(x, ordering, fast_mode),
        num_cases=4)
    tf.image_summary('final_distorted_image',
                     tf.expand_dims(distorted_image, 0), 3)
    return distorted_image
  else:
    return image