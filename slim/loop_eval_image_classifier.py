
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
from datasets import dataset_utils

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 64, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'eval_interval_secs', 60*1,
    'The frequency with which the model is evaluated, in seconds.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_boolean(
    'recall', False,
    'Whether or not to compute the Recall@5 metric.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

FLAGS = tf.app.flags.FLAGS


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=False)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size)
    [image, label, coarse_label] = provider.get(
        ['image', 'label', 'coarse_label'])
    label -= FLAGS.labels_offset

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

    images, labels, coarse_labels = tf.train.batch(
        [image, label, coarse_label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)
    coarse_labels = tf.cast(coarse_labels, tf.int32)
    tf.image_summary('image', images, max_images=5)

    ####################
    # Define the model #
    ####################
    logits, _ = network_fn(images)

    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()

    one_hot_labels = slim.one_hot_encoding(labels, 2)
    loss = slim.losses.softmax_cross_entropy(logits, one_hot_labels)

    predictions = tf.argmax(logits, 1)
    labels = tf.squeeze(labels)

    # Define the metrics:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Total_Loss': slim.metrics.streaming_mean(loss),
        'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
    })

  with tf.variable_scope('coarse_label_accuracy',
                         values=[predictions, labels, coarse_labels]):
    totals = tf.Variable(
        initial_value=tf.zeros([len(dataset.coarse_labels_to_names)]),
        trainable=False,
        collections=[tf.GraphKeys.LOCAL_VARIABLES],
        dtype=tf.float32,
        name='totals')

    counts = tf.Variable(
        initial_value=tf.zeros([len(dataset.coarse_labels_to_names)]),
        trainable=False,
        collections=[tf.GraphKeys.LOCAL_VARIABLES],
        dtype=tf.float32,
        name='counts')

    correct = tf.cast(tf.equal(predictions, labels), tf.int32)
    accuracy_ops = []
    for index, coarse_key in list(enumerate(dataset.coarse_labels_to_names)):
      label_correct = tf.boolean_mask(correct, tf.equal(coarse_key, coarse_labels))
      sum_correct = tf.reduce_sum(label_correct)
      sum_correct = tf.cast(tf.expand_dims(sum_correct, 0), tf.float32)
      delta_totals = tf.SparseTensor([[index]], sum_correct, totals.get_shape())
      label_count = tf.cast(tf.shape(label_correct), tf.float32)
      delta_counts = tf.SparseTensor([[index]], label_count, counts.get_shape())

      totals_compute_op = tf.assign_add(
          totals,
          tf.sparse_tensor_to_dense(delta_totals),
          use_locking=True)
      counts_compute_op = tf.assign_add(
          counts,
          tf.sparse_tensor_to_dense(delta_counts),
          use_locking=True)

      accuracy_ops.append(totals_compute_op)
      accuracy_ops.append(counts_compute_op)
    with tf.control_dependencies(accuracy_ops):
      update_op = tf.select(tf.equal(counts, 0),
                            tf.zeros_like(counts, tf.float32),
                            tf.div(totals, counts))
      names_to_updates['Coarse_Label_Accuracy'] = update_op

    if FLAGS.recall:
      recall_value, recall_update = slim.metrics.streaming_recall_at_k(
          logits, labels, 5)
      names_to_values['Recall@5'] = recall_value
      names_to_updates['Recall@5'] = recall_update

    # Print the summaries to screen.
    # TODO(vonclites) list(d.items()) is for Python 3... check compatibility
    for name, value in list(names_to_values.items()):
      summary_name = 'eval/%s' % name
      op = tf.scalar_summary(summary_name, value, collections=[])
      op = tf.Print(op, [value], summary_name)
      tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    for index, label_name in list(enumerate(dataset.coarse_labels_to_names.values())):
      summary_name = 'eval/%s' % label_name
      tf.scalar_summary(summary_name, update_op[index],
                        collections=[tf.GraphKeys.SUMMARIES])

    # TODO(sguada) use num_epochs=1
    if FLAGS.max_num_batches:
      num_batches = FLAGS.max_num_batches
    else:
      # This ensures that we make a single pass over all of the data.
      num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

#    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
#      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
#    else:
#      checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Evaluating %s' % FLAGS.checkpoint_path)

    slim.evaluation.evaluation_loop(
        master=FLAGS.master,
        checkpoint_dir=FLAGS.checkpoint_path,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        eval_op=list(names_to_updates.values()),
        eval_interval_secs=FLAGS.eval_interval_secs,
        variables_to_restore=slim.get_variables_to_restore())


if __name__ == '__main__':
  tf.app.run()
