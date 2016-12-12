# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A factory-pattern class which returns classification image/label pairs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets import cifar10
from datasets import cifar100
from datasets import flowers
from datasets import imagenet
from datasets import mnist
from datasets.iris import casia_livdet_tb
from datasets.iris import casia_livdet_all
from datasets.iris import casia
from datasets.iris import casia_norm
from datasets.iris import livdet
from datasets.iris import livdet_norm
from datasets.iris import casia_livdet_norm
from datasets.iris import clarkson_live_printed
from datasets.iris import clarkson_live_printed_norm
from datasets.iris import warsaw
from datasets.iris import warsaw_norm
from datasets.iris import testing
from datasets.iris import iris_norm

datasets_map = {
    'cifar10': cifar10,
    'cifar100': cifar100,
    'flowers': flowers,
    'imagenet': imagenet,
    'mnist': mnist,
    'casia_livdet_tb': casia_livdet_tb,
    'casia_livdet': casia_livdet_all,
    'casia': casia,
    'casia_norm': casia_norm,
    'livdet': livdet,
    'livdet_norm': livdet_norm,
    'casia_livdet_norm': casia_livdet_norm,
    'clarkson_live_printed': clarkson_live_printed,
    'clarkson_live_printed_norm': clarkson_live_printed_norm,
    'warsaw': warsaw,
    'warsaw_norm': warsaw_norm,
    'testing': testing,
    'iris_norm': iris_norm,
}


def get_dataset(name, split_name, dataset_dir, file_pattern=None, reader=None):
  """Given a dataset name and a split_name returns a Dataset.

  Args:
    name: String, the name of the dataset.
    split_name: A train/test split name.
    dataset_dir: The directory where the dataset files are stored.
    file_pattern: The file pattern to use for matching the dataset source files.
    reader: The subclass of tf.ReaderBase. If left as `None`, then the default
      reader defined by each dataset is used.

  Returns:
    A `Dataset` class.

  Raises:
    ValueError: If the dataset `name` is unknown.
  """
  if name not in datasets_map:
    raise ValueError('Name of dataset unknown %s' % name)
  return datasets_map[name].get_split(
      split_name,
      dataset_dir,
      file_pattern,
      reader)
