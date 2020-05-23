import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
import time

AUTOTUNE = tf.data.experimental.AUTOTUNE

now_time = time.time()  ##2##

# 常用路径
base_path = 'cifar10_data'
image_path = base_path + '/image'
train_image_path = image_path + '/train'
test_image_path = image_path + '/test'

import pathlib
# 获取 训练图片 目录
data_root = pathlib.Path(train_image_path)

import random
all_image_paths = list(data_root.glob('*/*'))
# print(all_image_paths)

all_image_paths = [str(path) for path in all_image_paths]

# 打乱图片路径
random.shuffle(all_image_paths)

# 获取 图片总量
image_count = len(all_image_paths)
# cifar10 的 特征
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))
# 获取 图片 对应 的 label list
all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths] 

def preprocess_image(image):
  # image = tf.image.cov
  image = tf.image.decode_jpeg(image, channels=3)
  # TensorFlow的函数处理图片后存储的数据是float32格式
  # image /= 255.0  # normalize to [0,1] range
  return image

def load_and_preprocess_image(path):
  # image tensor
  image = tf.io.read_file(path)
  # <tf.Tensor: id=1, shape=(), dtype=string, numpy=b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00...
  return preprocess_image(image)

x_train = np.empty(shape=(50000, 32, 32, 3))

# plt.imshow(load_and_preprocess_image(all_image_paths[0]))  # 绘制图片
# plt.show()

for index, path in enumerate(all_image_paths):
  image = load_and_preprocess_image(path)
  x_train[index] = image

print("x_train.shape", x_train.shape)

print(type(x_train))

# 构建 标签数据表
# label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)  # 给数据增加一个维度，使数据和网络结构匹配

# # x_train = tf.convert_to_tensor(x_train)

# print("x_train.shape", x_train.shape)
# print("label_ds.shape", label_ds.shape)

total_time = time.time() - now_time  ##3##
print("total_time", total_time)  ##4##