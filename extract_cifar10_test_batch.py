import os
import sys
import glob
import tensorflow as tf
import numpy as np
import cv2 

# cifar10 的 特征
classification = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 
def unpickle(file):
  import pickle
  with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
  return dict

# cifar10_floder = '/Users/suyunlong/python/cifar10_data/cifar-10-batches-py'
cifar10_floder = 'cifar10_data/cifar-10-batches-py'

train_files = glob.glob(cifar10_floder + '/test_batch*')

data = []
labels = []

# 提取 data 和 labels
for file in train_files:
  _file = unpickle(file)

  data.extend(_file[b"data"]) # list 类型
  labels.extend(_file[b"labels"])
  # data += list(_file[b"data"])
  # labels += list(_file[b"labels"])

# -1 是 data 长度的 占位 ， 3 是 图片通道数 ， 32 图片为 32 * 32
imgs = np.reshape(data, [-1, 3, 32, 32])

# imgs.shape[0] = 50000
for i in range(imgs.shape[0]):
  img_data = imgs[i, ...]

  # 1 2 0 是 上面 32 32 -1 图片数据维度转换
  img_data = np.transpose(img_data, [1, 2, 0])

  # # 转换 图片 格式 
  img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

  # 存放路径
  folder = '{}/{}'.format('cifar10_data/image/test', classification[labels[i]])

  if not os.path.exists(folder):
    os.mkdir(folder)

  cv2.imwrite('{}/{}.jpg'.format(folder, str(i)), img_data)
