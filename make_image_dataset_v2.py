from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

# 常用路径
base_path = 'cifar10_data'
image_path = base_path + '/image'
train_image_path = image_path + '/train'
test_image_path = image_path + '/test'

import pathlib
# 获取 训练图片 目录
data_root = pathlib.Path(train_image_path)
# for item in data_root.iterdir():
#   print(item)

# cifar10_data/image/train/cat
# cifar10_data/image/train/dog
# cifar10_data/image/train/truck
# cifar10_data/image/train/bird
# cifar10_data/image/train/airplane
# cifar10_data/image/train/ship
# cifar10_data/image/train/frog
# cifar10_data/image/train/horse
# cifar10_data/image/train/deer
# cifar10_data/image/train/automobile

import random
all_image_paths = list(data_root.glob('*/*'))
# print(all_image_paths)

all_image_paths = [str(path) for path in all_image_paths]

# 打乱图片路径
random.shuffle(all_image_paths)
# print(all_image_paths[:10])

# 获取 图片总量
image_count = len(all_image_paths)
# print(image_count)
# 50000

# cifar10 的 特征
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())

# print(label_names)
# ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

label_to_index = dict((name, index) for index, name in enumerate(label_names))
# print(label_to_index)
# {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

# 获取 图片 对应 的 label list
all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths] 
# print("First 10 labels indices: ", all_image_labels[:10])

def preprocess_image(image):
  img_tensor = tf.image.decode_jpeg(image, channels=3)
  # print(img_tensor.shape)
  # print(img_tensor.dtype)
  img_tensor = tf.image.resize(img_tensor, [32, 32])
  img_tensor /= 255.0  # normalize to [0,1] range

  return image

def load_and_preprocess_image(path):
  # image tensor
  image = tf.io.read_file(path)
  # <tf.Tensor: id=1, shape=(), dtype=string, numpy=b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00...
  return preprocess_image(image)
  
# other
ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))

# 元组被解压缩到映射函数的位置参数中
def load_and_preprocess_from_path_label(path, label):
  return load_and_preprocess_image(path), label

image_label_ds = ds.map(load_and_preprocess_from_path_label)

print(image_label_ds)