import os
import glob
import tensorflow as tf
import numpy as np
import pathlib
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

# 常用路径
base_path = 'cifar10_data'
image_path = base_path + '/image'
train_image_path = image_path + '/train'
test_image_path = image_path + '/test'
dataset_path = 'datasets'

# 获取 训练图片 目录
data_root = pathlib.Path(test_image_path)
image_paths = list(data_root.glob('*/*'))
image_paths = [str(path) for path in image_paths]
# 打乱图片路径
np.random.shuffle(image_paths)
# 获取 图片总量
image_count = len(image_paths)
# cifar10 的 特征
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))
# 获取 图片 对应 的 label list
image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in image_paths]

# 定义 record 存放路径 及 record_writer 实例
record_file = 'test_images.tfrecord'
record_path = dataset_path + '/' + record_file

print(os.path.exists(record_path))

# shuffle 图片顺序
shulled_index_list = [i for i in range(image_count)]
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
  for i in range(image_count):
    path = image_paths[shulled_index_list[i]]
    label = image_labels[shulled_index_list[i]]

    # 二进制 图片
    image_string = open(path, 'rb').read()
    
    tf_example = image_example(image_string, label)
    writer.write(tf_example.SerializeToString())

end_time = time.time() - start_time
print(end_time)
