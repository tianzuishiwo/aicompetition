import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os
import math
from PIL import Image
from dl.practise.gesture.dl_utils import *
from dl.practise.gesture.config import *

ROOT_PATH = '/Users/wushaohua/my/hm/数据集/深度学习/Sign-Language-Digits-Dataset-master/Dataset/'




class DataLoader(object):
    def __init__(self, batch_size, img_size):
        self.batch_size = batch_size
        self.img_size = img_size
        self.x_train, self.y_train, self.x_test, self.y_test = self._load_data()
        pass

    def get_train_sequence(self):
        # img_paths, labels, batch_size, img_size, use_aug
        return MySequence(self.x_train, self.y_train, self.batch_size, self.img_size, True)

    def get_test_sequence(self):
        return MySequence(self.x_test, self.y_test, self.batch_size, self.img_size, False)

    def _load_data(self):
        label_list = None
        train_filenames = None
        for i in range(0, 10):
            path = ROOT_PATH + str(i) + '/'
            small_list = [path + filename for filename in os.listdir(path)]
            small_list = self.set_retain_data(small_list)
            # print(f'当前手势：{i}  图片数量： {len(small_list)}')

            if train_filenames is None:
                train_filenames = tf.constant(small_list)
                label_list = tf.ones(shape=(len(small_list),)) * i
                continue

            train_filenames = tf.concat([train_filenames, tf.constant(small_list)], axis=-1)
            label_list = tf.concat([label_list, tf.ones(shape=(len(small_list),)) * i], axis=-1)

        x_train, x_test, y_train, y_test = train_test_split(train_filenames.numpy().tolist(),
                                                            label_list.numpy().tolist(), test_size=0.20)

        print('train_filenames.shape: ', train_filenames.shape)
        print('label_list.shape: ', label_list.shape)
        return x_train, y_train, x_test, y_test

    def set_retain_data(self, small_list):
        len1 = len(small_list)
        retain_count = math.floor(len1 * DATA_RATE)
        print('保留数据量： ', retain_count, ' 总共数据量： ', len1)
        return small_list[:retain_count]


class MySequence(tf.keras.utils.Sequence):

    def __init__(self, img_paths, labels, batch_size, img_size, use_aug):

        # 1、获取训练特征与目标值的合并结果 [batch_size, 1],  [batch_size, 40]   [batch_size, 41]
        self.x_y = np.hstack((np.array(img_paths).reshape(len(img_paths), 1),
                              np.array(labels).reshape(len(labels), 1)))
        self.batch_size = batch_size
        self.img_size = img_size  # (300, 300)
        self.use_aug = use_aug
        self.alpha = 0.2
        # 随机擦出方法
        self.eraser = get_random_eraser(s_h=0.3, pixel_level=True)

    def __len__(self):
        return math.ceil(len(self.x_y) / self.batch_size)

    @staticmethod
    def center_img(img, size=None, fill_value=255):
        """改变图片尺寸到300x300，并且做填充使得图像处于中间位置
        """
        h, w = img.shape[:2]
        if size is None:
            size = max(h, w)
        shape = (size, size) + img.shape[2:]
        background = np.full(shape, fill_value, np.uint8)
        center_x = (size - w) // 2
        center_y = (size - h) // 2
        background[center_y:center_y + h, center_x:center_x + w] = img
        return background

    def preprocess_img(self, img_path):
        """处理每张图片，大小， 数据增强
        :param img_path:
        :return:
        """
        # print('img_path: ',img_path)
        # 1、读取图片对应内容，做形状，内容处理, (h, w)
        img = Image.open(img_path)
        # [180, 200, 3]
        scale = self.img_size[0] / max(img.size[:2])
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))
        img = img.convert('RGB')
        img = np.array(img)

        # 2、数据增强：如果是训练集进行数据增强操作
        if self.use_aug:
            # 1、随机擦处
            img = self.eraser(img)

            # 2、翻转
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                width_shift_range=0.05,
                height_shift_range=0.05,
                horizontal_flip=True,
                vertical_flip=True,
            )
            img = datagen.random_transform(img)

        # 4、处理一下形状 【300， 300， 3】
        # 改变到[300, 300] 建议不要进行裁剪操作，变形操作，保留数据增强之后的效果，填充到300x300
        img = self.center_img(img, self.img_size[0])
        return img

    def __getitem__(self, idx):

        # 1、获取当前批次idx对应的特征值和目标值
        batch_x = self.x_y[idx * self.batch_size: self.batch_size * (idx + 1), 0]
        batch_y = self.x_y[idx * self.batch_size: self.batch_size * (idx + 1), 1:]

        batch_x = np.array([self.preprocess_img(img_path) for img_path in batch_x])
        batch_y = np.array(batch_y).astype(np.float32)

        # 2、mixup
        # batch_x, batch_y = self.mixup(batch_x, batch_y)

        # 3、归一化处理
        batch_x = self.preprocess_input(batch_x)

        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.x_y)

    def mixup(self, batch_x, batch_y):
        """
        数据混合mixup
        :param batch_x: 要mixup的batch_X
        :param batch_y: 要mixup的batch_y
        :return: mixup后的数据
        """
        size = self.batch_size
        l = np.random.beta(self.alpha, self.alpha, size)

        X_l = l.reshape(size, 1, 1, 1)
        y_l = l.reshape(size, 1)

        X1 = batch_x
        Y1 = batch_y
        X2 = batch_x[::-1]
        Y2 = batch_y[::-1]

        X = X1 * X_l + X2 * (1 - X_l)
        Y = Y1 * y_l + Y2 * (1 - y_l)

        return X, Y

    def preprocess_input(self, x):
        """归一化处理样本特征值
        :param x:
        :return:
        """
        assert x.ndim in (3, 4)
        assert x.shape[-1] == 3

        MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]

        x = x - np.array(MEAN_RGB)
        x = x / np.array(STDDEV_RGB)

        return x


def main():
    batch_size = 32
    img_size = (227, 227, 3)
    DataLoader(batch_size, img_size)
    pass


if __name__ == '__main__':
    main()
