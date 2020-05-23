import os
import sys
import tarfile

from six.moves import urllib

# import tensorflow as tf

DEST_DIR = 'cifar10_data/'
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

# Download and extract the tarball from Alex's website.


def download_and_extract(DATA_URL, DEST_DIR):
  if not os.path.exists(DEST_DIR):
    os.makedirs(DEST_DIR)

  filename = DATA_URL.split('/')[-1]  # 文件名
  filepath = os.path.join(DEST_DIR, filename)

  if not os.path.exists(filepath):
    # 文件下载函数
    def _progress(count, block_size, total_size):
        # %.1f%% -> 实数后面输出1个 %
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                         float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()  # 更新stdout

    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

  # 提取 bin 文件路径
  extracted_dir_path = os.path.join(DEST_DIR, 'cifar-10-batches-py')

  if not os.path.exists(extracted_dir_path):
    # Read from and write to tar format archives
    tarfile.open(filepath, 'r:gz').extractall(DEST_DIR)


download_and_extract(DATA_URL, DEST_DIR)
