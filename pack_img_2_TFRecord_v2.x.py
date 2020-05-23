import os
import glob
import tensorflow as tf
import numpy as np
import cv2 

import time

start_time = time.time()

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# cifar10 的 特征
classification = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# 常用路径
base_path = 'cifar10_data'
image_path = 'cifar10_data/image'
train_image_path = 'cifar10_data/image/train'
test_image_path = 'cifar10_data/image/test'
dataset_path = 'datasets'

# 图片 path_list 和 label_list
image_paths = []
image_labels = []
index = 0

# 遍历 image 目录 获取 path 和 lable 两个 list
for index, path in enumerate(classification):
  image_path_list = glob.glob(train_image_path + '/' + path + '/*')
  image_paths.extend(image_path_list) 

  image_label_list = [index for i in range(image_path_list.__len__())]
  image_labels.extend(image_label_list)

# 定义 record 存放路径 及 record_writer 实例
record_file = 'train_images.tfrecord'
record_path = dataset_path + '/' + record_file

print(os.path.exists(record_path))

image_paths_length = image_paths.__len__()
# shuffle 图片顺序
shulled_index_list = [i for i in range(image_paths_length)]
np.random.shuffle(shulled_index_list)

def image_example(image_string, label):
  image_shape = tf.image.decode_jpeg(image_string).shape

  feature = {
    'height': _int64_feature(image_shape[0]),
    'width': _int64_feature(image_shape[1]),
    'depth': _int64_feature(image_shape[2]),
    'label': _int64_feature(label),
    'image': _bytes_feature(image_string),
  }

  return tf.train.Example(features=tf.train.Features(feature=feature))

with tf.io.TFRecordWriter(record_path) as writer:
  for i in range(image_paths_length):
    path = image_paths[shulled_index_list[i]]
    label = image_labels[shulled_index_list[i]]

    # 二进制 图片
    image_string = open(path, 'rb').read()
    
    tf_example = image_example(image_string, label)
    writer.write(tf_example.SerializeToString())

end_time = time.time() - start_time
print(end_time)
