import os
import glob
import tensorflow as tf
import numpy as np
import cv2 

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

# 图片 path_list 和 label_list
image_paths = []
image_labels = []
index = 0

# 遍历 image 目录 获取 path 和 lable 两个 list
for path in classification:
  image_path_list = glob.glob(train_image_path + '/' + path + '/*')
  image_paths.extend(image_path_list) 

  image_label_list = [index for i in range(image_path_list.__len__())]
  image_labels.extend(image_label_list)

  index += 1

# 定义 record 存放路径 及 record_writer 实例
record_file = 'images.tfrecords'

record_path = base_path + '/' + record_file

if os.path.exists(record_path):
  print('删除了')
  os.remove(record_path)

print(os.path.exists(record_path))

# tensorflow 1.x
# writer = tf.python_io.TFRecordWriter(record_path)
# tensorflow 2.x
writer = tf.io.TFRecordWriter(record_path)

image_paths_length = image_paths.__len__()
# shuffle 图片顺序
shulled_index_list = [i for i in range(image_paths_length)]
np.random.shuffle(shulled_index_list)

for i in range(image_paths_length):
  path = image_paths[shulled_index_list[i]]
  label = image_labels[shulled_index_list[i]]

  # data = cv2.imread(path)

  image_string = open(path, 'rb').read()

  # 图片大小不一致的化 可以 cv2 读取的时候 resize 成 同一个size

  # 将数据处理成二进制
  image_shape = tf.image.decode_jpeg(image_string).shape

  example = tf.train.Example(features = tf.train.Features(feature = {
    'label': _int64_feature(label),
    'image': _bytes_feature(image_string),

    # 'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [data.tobytes()])),
    # 'label': tf.train.Feature(int64_list = tf.train.Int64List(value = [label])),
    # 'width': tf.train.Feature(int64_list = tf.train.Int64List(value = [image_shape[0]])),
    # 'height': tf.train.Feature(int64_list = tf.train.Int64List(value = [image_shape[1]]))

    # 'height': _int64_feature(image_shape[0]),
    # 'width': _int64_feature(image_shape[1]),
  }))

  # 写入文件的时候不能直接处理对象，需要将其转化为字符串才能处理
  # 反序列化到对象的方法，该方法是FromString()
  # example 序列化后 写入
  writer.write(example.SerializeToString())

writer.close()
