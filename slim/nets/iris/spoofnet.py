from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

def spoofnet_v1(inputs,
                num_classes=2,
                is_training=True,
                dropout_keep_prob=0.5,
                spatial_squeeze=True,
                scope='spoofnet'):
  with tf.variable_scope(scope, 'spoofnet', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.conv2d(inputs, 16, [5, 5], scope='conv1')
      net = slim.max_pool2d(net, [3, 3], scope='pool1')
      net = slim.conv2d(net, 64, [5, 5], scope='conv2')
      net = slim.max_pool2d(net, [3, 3], scope='pool2')
      net = slim.flatten(net, scope='flatten2')
      net = slim.fully_connected(net, num_classes,
                                 activation_fn=None,
                                 normalizer_fn=None,
                                 scope='fc3')
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      return net, end_points
spoofnet_v1.default_image_size = 224

def spoofnet_3x3(inputs,
                num_classes=2,
                is_training=True,
                dropout_keep_prob=0.5,
                spatial_squeeze=True,
                scope='spoofnet'):
  with tf.variable_scope(scope, 'spoofnet', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.conv2d(inputs, 16, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.conv2d(net, 64, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.flatten(net, scope='flatten2')
      net = slim.fully_connected(net, num_classes,
                                 activation_fn=None,
                                 normalizer_fn=None,
                                 scope='fc3')
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      return net, end_points
spoofnet_3x3.default_image_size = 224

def spoofnet_5(inputs,
               num_classes=2,
               is_training=True,
               dropout_keep_prob=0.5,
               spatial_squeeze=True,
               scope='spoofnet'):
  with tf.variable_scope(scope, 'spoofnet', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 1, slim.conv2d, 16, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      # 112
      net = slim.repeat(inputs, 1, slim.conv2d, 32, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      # 56
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      # 28
      net = slim.flatten(net, scope='flatten3')
      net = slim.fully_connected(net, num_classes,
                                 activation_fn=None,
                                 normalizer_fn=None,
                                 scope='fc4')
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      return net, end_points
spoofnet_5.default_image_size = 224

def spoofnet_4xk(inputs,
                       num_classes=2,
                       is_training=True,
                       dropout_keep_prob=0.5,
                       spatial_squeeze=True,
                       scope='spoofnet'):
  with tf.variable_scope(scope, 'spoofnet', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.conv2d(inputs, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.conv2d(net, 256, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.flatten(net, scope='flatten2')
      net = slim.fully_connected(net, num_classes,
                                 activation_fn=None,
                                 normalizer_fn=None,
                                 scope='fc3')
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      return net, end_points
spoofnet_4xk.default_image_size = 224

def vgg_16_1(inputs,
          num_classes=2,
          is_training=True,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          scope='vgg_16'):
  with tf.variable_scope(scope, 'spoofnet', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [4, 4], stride=4, scope='pool1')
      net = slim.flatten(net, scope='flatten1')
      net = slim.fully_connected(net, num_classes,
                                 activation_fn=None,
                                 normalizer_fn=None,
                                 scope='fc2')
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      return net, end_points
vgg_16_1.default_image_size = 224

def vgg_16_3(inputs,
          num_classes=2,
          is_training=True,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          scope='vgg_16'):
  with tf.variable_scope(scope, 'spoofnet', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(inputs, 2, slim.conv2d, 128, [3, 3], 2, scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.flatten(net, scope='flatten2')
      net = slim.fully_connected(net, num_classes,
                                 activation_fn=None,
                                 normalizer_fn=None,
                                 scope='fc3')
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      return net, end_points
vgg_16_3.default_image_size = 224