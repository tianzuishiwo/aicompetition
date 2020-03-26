import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from dl.practise.gesture.dl_utils import *
from dl.practise.gesture.config import *


class DataLoader(object):
    def __init__(self, batch_size, img_size):
        self.batch_size = batch_size
        self.img_size = img_size
        self.x_train, self.y_train, self.x_test, self.y_test = self._load_data()
        pass

    def get_train_sequence(self):
        return MySequence(self.x_train, self.y_train, self.batch_size, self.img_size, True)

    def get_test_sequence(self):
        return MySequence(self.x_test, self.y_test, self.batch_size, self.img_size, False)

    def _load_data(self):
        label_list = None
        train_filenames = None
        for i in range(0, 10):
            small_list = read_and_restrict_imagepaths(ROOT_DATA_PATH + str(i) + '/')

            if train_filenames is None:
                train_filenames = tf.constant(small_list)
                label_list = tf.ones(shape=(len(small_list),)) * i
                continue

            train_filenames = tf.concat([train_filenames, tf.constant(small_list)], axis=-1)
            label_list = tf.concat([label_list, tf.ones(shape=(len(small_list),)) * i], axis=-1)

        x_train, x_test, y_train, y_test = train_test_split(train_filenames.numpy().tolist(),
                                                            label_list.numpy().tolist(), test_size=0.20, shuffle=True)

        print('train_filenames.shape: ', train_filenames.shape)
        print('label_list.shape: ', label_list.shape)
        return x_train, y_train, x_test, y_test


class MySequence(tf.keras.utils.Sequence):

    def __init__(self, img_paths, labels, batch_size, img_size, use_aug):
        # 1、获取训练特征与目标值的合并结果 [batch_size, 1],  [batch_size, 40]   [batch_size, 41]
        self.x_y = x_y_hstack(img_paths, labels)
        self.batch_size = batch_size
        self.img_size = img_size  # (300, 300)
        self.is_training = use_aug

    def __len__(self):
        return math.ceil(len(self.x_y) / self.batch_size)

    def preprocess_img(self, img_path):
        """处理每张图片，大小， 数据增强"""
        # 1、读取图片对应内容，做形状，内容处理, (h, w)
        img = read_image_and_resize(img_path, self.img_size[0])
        # 2、数据增强：如果是训练集进行数据增强操作
        if self.is_training:
            # 1、随机擦处
            img = image_eraser(img)
            # 2、数据增强：翻转
            img = image_enhance(img)
        # 4、处理一下形状 【300， 300， 3】
        # 改变到[300, 300] 建议不要进行裁剪操作，变形操作，保留数据增强之后的效果，填充到300x300
        img = image_fill_padding(img, self.img_size[0])
        return img

    def __getitem__(self, index):
        # 1、获取当前批次idx对应的特征值和目标值
        batch_x = get_batch_data(self.x_y[:, 0], index, self.batch_size)
        batch_y = get_batch_data(self.x_y[:, 1], index, self.batch_size)

        batch_x = np.array([self.preprocess_img(img_path) for img_path in batch_x])
        batch_y = np.array(batch_y).astype(np.float32)

        # 2、mixup
        # batch_x, batch_y = mixup(batch_x, batch_y)   # 警告：这个方法有问题，暂时别使用
        # 3、归一化处理
        batch_x = normalization(batch_x)
        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.x_y)


def main():
    batch_size = 32
    img_size = (227, 227, 3)
    DataLoader(batch_size, img_size)
    pass


if __name__ == '__main__':
    main()
