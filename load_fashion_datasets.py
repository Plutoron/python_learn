import numpy as np
import os
import tensorflow as tf

# mnist = np.load('./datasets/mnist.npz')

# print(mnist)

# for item in mnist:
#   print(item)

# print(mnist['x_test'])

import pathlib
import gzip

def load_localData(path):
  if os.path.exists(path):
    files = [
      'train-labels-idx1-ubyte.gz', 
      'train-images-idx3-ubyte.gz',
      't10k-labels-idx1-ubyte.gz', 
      't10k-images-idx3-ubyte.gz'
    ]

    paths = []

    for _path in files:
      paths.append(path + '/' + _path)
    
    # for _path in gz_paths: 
    with gzip.open(paths[0], 'rb') as lbpath:
      y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
 
    with gzip.open(paths[1], 'rb') as imgpath:
      x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
 
    with gzip.open(paths[2], 'rb') as lbpath:
      y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
 
    with gzip.open(paths[3], 'rb') as imgpath:
      x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)