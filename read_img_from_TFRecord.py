import os
import tensorflow as tf
import numpy as np
import time

start_time = time.time()

def load_data(tf_record_path):
  if not(os.path.exists(tf_record_path)):
    print('不存在的tfrecord')
    os._exit(0)

  # 读取 TFRecord
  raw_dataset = tf.data.TFRecordDataset(tf_record_path)

  # feature 解释 字典
  image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image': tf.io.FixedLenFeature([], tf.string)
  }

  length = 0
  parsed_dataset = []

  for example_proto in raw_dataset:
    length += 1

    parsed_example = tf.io.parse_single_example(example_proto, image_feature_description)
    parsed_dataset.append(parsed_example)

  def preprocess_image(image):
    # image = tf.image.cov
    image = tf.image.decode_jpeg(image, channels=3)
    # TensorFlow的函数处理图片后存储的数据是float32格式
    # image = tf.image.resize(image, [32, 32])
    # image /= 255.0  # normalize to [0,1] range
    return image

  x_train = np.empty(shape=(length, 32, 32, 3))
  y_train = np.empty(shape=(length, 1))

  for index, image_features in enumerate(parsed_dataset):
    # 二进制 图片
    image = image_features['image']
    label = image_features['label']

    image = preprocess_image(image)

    x_train[index] = image
    y_train[index] = label
  
  print(x_train.shape)
  print(y_train.shape)

  return x_train, y_train
  