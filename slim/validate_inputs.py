from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

num_clones = 1
clone_on_cpu = False
worker_replicas = 1
num_ps_tasks = 0
num_readers = 4
num_preprocessing_threads = 8
dataset_name = 'iris_raw'
dataset_split_name = 'train'
dataset_dir = '/mnt/data1/slim/iris_raw'
labels_offset = 0
preprocessing_name = 'vgg_iris_raw'
batch_size = 64
train_image_size = None
task = 0
weight_decay = 1

def main():
  with tf.Graph().as_default():
    if not dataset_dir:
      raise ValueError('You must supply the dataset directory with --dataset_dir')

    deploy_config = model_deploy.DeploymentConfig(
        num_clones=num_clones,
        clone_on_cpu=clone_on_cpu,
        replica_id=task,
        num_replicas=worker_replicas,
        num_ps_tasks=num_ps_tasks)

    dataset = dataset_factory.get_dataset(
        dataset_name, dataset_split_name, dataset_dir)

    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=True)

    with tf.device(deploy_config.inputs_device()):
      with tf.name_scope('inputs'):
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=num_readers,
            common_queue_capacity=20 * batch_size,
            common_queue_min=10 * batch_size)
        [image, label, fp] = provider.get(['image', 'label', 'filepath'])
        label -= labels_offset

    train_image_size = 224

    image = image_preprocessing_fn(image, train_image_size,
                                   train_image_size)

    images, labels, fps = tf.train.batch(
        [image, label, fp],
        batch_size=batch_size,
        num_threads=num_preprocessing_threads,
        capacity=5 * batch_size)
    tf.image_summary('image', images, max_images=5)
    labels = slim.one_hot_encoding(
        labels, dataset.num_classes - labels_offset)
    batch_queue = slim.prefetch_queue.prefetch_queue(
        [images, labels, fps], capacity=2 * deploy_config.num_clones)

    images, labels, fps = batch_queue.dequeue()

    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    image_data, label_data, fp_data = sess.run([images, labels, fps])

    coord.request_stop()
    coord.join(threads)
    sess.close()
    return image_data, label_data, fp_data

imgs, labels, fps = main()