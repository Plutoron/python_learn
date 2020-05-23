import os
import glob
import tensorflow as tf
import numpy as np
import cv2 

# cifar10 的 特征
classification = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# 常用路径
base_path = 'cifar10_data'
image_path = 'cifar10_data/image'
train_image_path = 'cifar10_data/image/train'
test_image_path = 'cifar10_data/image/test'

images = ['1.jpg', '2.jpg', '3.jpg']
labels = [1, 2, 3]

# num_epochs = None 文件队列有文件就读取，不限制次数
# [images, labels] = tf.train.slice_input_producer([images, labels], num_epochs = 2, shuffle = True) 
# [images, labels] = tf.train.slice_input_producer([images, labels], num_epochs = 2, shuffle = True) 

dataset = tf.data.Dataset.from_tensor_slices((images, labels))
dataset = dataset.shuffle(buffer_size = 3, reshuffle_each_iteration = True)
dataset = dataset.repeat(2)

# 创建Iterator读取数据
it = dataset.__iter__()
for i in range(5):
    x, y = it.next()
    print(it.next())
