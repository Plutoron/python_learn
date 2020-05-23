import random
import pathlib
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
import time

AUTOTUNE = tf.data.experimental.AUTOTUNE

now_time = time.time()  # 2##

# 常用路径
base_path = 'cifar10_data'
image_path = base_path + '/image'
train_image_path = image_path + '/train'
test_image_path = image_path + '/test'

# 获取 训练图片 目录
data_root = pathlib.Path(train_image_path)
all_image_paths = list(data_root.glob('*/*'))
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

def load_data():
    def preprocess_image(image):
        # image = tf.image.cov
        image = tf.image.decode_jpeg(image, channels=3)
        # TensorFlow的函数处理图片后存储的数据是float32格式
        image = tf.image.resize(image, [32, 32])
        image /= 255.0  # normalize to [0,1] range
        return image

    def load_and_preprocess_image(path):
        image = tf.io.read_file(path)
        return preprocess_image(image)

    # x_train = np.empty(shape=(50000, 32, 32, 3))
    x_train = np.empty(shape=(50000, 32, 32, 3))

    # y_train = np.empty(shape=(50000, 1))
    y_train = np.empty(shape=(50000, 1))

    for index, path in enumerate(all_image_paths[0:50000]):
        image = load_and_preprocess_image(path)
        x_train[index] = image

    for index, label in enumerate(all_image_labels[0:50000]):
        y_train[index] = label

    return x_train, y_train
